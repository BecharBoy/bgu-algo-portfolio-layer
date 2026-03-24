import asyncio
import logging
import uuid
from ib_async import IB, Stock, LimitOrder, MarketOrder

logger = logging.getLogger(__name__)

# How long we wait for a full fill before cancelling the order.
ORDER_FILL_TIMEOUT_SECONDS = 15

# Limit price buffer: BUY gets +0.5% above reference, SELL gets -0.5%.
# Ensures we are competitive at open without chasing price.
LIMIT_PRICE_BUFFER = 0.005


class IB_Connect:
    def __init__(self, host: str, port: int, client_id: int):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    async def connect(self) -> None:
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()

    async def reconnect_if_needed(self) -> None:
        if not self.ib.isConnected():
            await self.connect()

    # ── Account & Positions ───────────────────────────────────────────────────

    async def get_account_summary(self) -> dict:
        account_values = await self.ib.reqAccountSummaryAsync()
        result = {}
        for av in account_values:
            if av.tag == "NetLiquidation":
                result["nlv"] = float(av.value)
            elif av.tag == "TotalCashValue":
                result["cash"] = float(av.value)
        if "nlv" not in result or "cash" not in result:
            raise RuntimeError(f"Incomplete account summary from TWS: {result}")
        return result

    async def get_positions(self) -> dict:
        """
        Returns live positions directly from IB/TWS.
        Used for end-of-cycle reconciliation against DB.
        """
        positions = await self.ib.reqPositionsAsync()
        return {
            pos.contract.symbol: {
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
            }
            for pos in positions
            if pos.position != 0
        }

    # ── Order Helpers ─────────────────────────────────────────────────────────

    def _limit_price(self, action: str, reference_price: float) -> float:
        """
        Compute limit price from reference (last close).
        BUY:  +0.5% above reference — willing to pay slightly above close.
        SELL: -0.5% below reference — willing to accept slightly below close.
        Rounded to 2 decimal places (cent precision; IB rejects sub-cent prices).
        """
        if action.upper() == "BUY":
            return round(reference_price * (1 + LIMIT_PRICE_BUFFER), 2)
        return round(reference_price * (1 - LIMIT_PRICE_BUFFER), 2)

    def get_fills_for_order(self, order_id: int) -> list:
        """
        Returns IB fill objects for a specific order_id.
        Encapsulates the ib_async internal .fills() call so callers
        never need to touch self.ib.ib directly.
        """
        return [f for f in self.ib.fills() if f.execution.orderId == order_id]

    # ── Order Execution ───────────────────────────────────────────────────────

    async def place_limit_order(
        self, action: str, symbol: str, quantity: int, reference_price: float
    ) -> dict:
        """
        Places a DAY limit order at reference_price +/- LIMIT_PRICE_BUFFER.
        TIF=DAY: expires end of session if not filled.
        Returns order_id and the computed limit price.
        """
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        limit_px = self._limit_price(action, reference_price)
        order = LimitOrder(action, quantity, limit_px, tif="DAY")
        trade = self.ib.placeOrder(contract, order)
        logger.info(
            f"[ORDER] {action} {quantity} {symbol} "
            f"limit={limit_px} (ref={reference_price}) "
            f"order_id={trade.order.orderId}"
        )
        return {"order_id": trade.order.orderId, "limit_price": limit_px}

    async def wait_for_order_fill(
        self, order_id: int, timeout_seconds: int = ORDER_FILL_TIMEOUT_SECONDS
    ) -> dict | None:
        """
        Waits for a FULL fill on a specific order via IB's execDetailsEvent.

        - Filters by order_id: ignores fills for any other order on this connection.
        - Handles multi-tranche fills: IB can fill in parts (e.g. 60 then 40).
          We keep waiting until cumQty == totalQty (full fill confirmed).
        - future.done() guard: IB may send duplicate callbacks; we ignore any
          callback that arrives after the future is already resolved.
        - On timeout: returns None. Does NOT cancel the order — caller must do that.
        - Always unsubscribes the callback in `finally` to prevent stale listeners.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        trade = next(
            (t for t in self.ib.trades() if t.order.orderId == order_id), None
        )
        expected_qty = trade.order.totalQuantity if trade else None

        def on_exec_details(trade, fill):
            if fill.execution.orderId != order_id or future.done():
                return
            filled_qty = fill.execution.cumQty
            if expected_qty is not None and filled_qty < expected_qty:
                logger.info(
                    f"[FILL] order_id={order_id} partial: "
                    f"{filled_qty}/{expected_qty} {fill.contract.symbol} — waiting for rest"
                )
                return
            future.set_result({
                "order_id":   fill.execution.orderId,
                "symbol":     fill.contract.symbol,
                "action":     "BUY" if fill.execution.side == "BOT" else "SELL",
                "quantity":   int(fill.execution.cumQty),
                "fill_price": fill.execution.price,
                "filled_at":  fill.execution.time,
                "fill_id":    str(uuid.uuid4()),
            })

        self.ib.execDetailsEvent += on_exec_details
        try:
            return await asyncio.wait_for(future, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            return None
        finally:
            self.ib.execDetailsEvent -= on_exec_details

    async def cancel_order(self, order_id: int) -> str:
        """
        Sends a cancel request for order_id and waits for IB's confirmation.

        Possible return values:
          'Cancelled'       — cleanly cancelled, no exposure.
          'Filled'          — fill arrived before cancel landed; caller must log it.
          'PartiallyFilled' — partial fill before cancel; caller must flatten.
          'NotFound'        — order not in active trades (unexpected).
          'CancelTimeout'   — IB did not confirm cancel within 10s; check TWS manually.

        Always unsubscribes the status callback in `finally`.
        """
        trade = next(
            (t for t in self.ib.trades() if t.order.orderId == order_id), None
        )
        if trade is None:
            logger.error(f"[CANCEL] order_id={order_id} not found in active trades — check TWS.")
            return "NotFound"

        current_status = trade.orderStatus.status
        if current_status in ("Filled", "Cancelled", "Inactive"):
            logger.info(f"[CANCEL] order_id={order_id} already resolved as {current_status}, skipping cancel.")
            return current_status

        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def on_status_change(trade):
            status = trade.orderStatus.status
            if status in ("Cancelled", "Filled", "PartiallyFilled") and not future.done():
                future.set_result(status)

        trade.statusEvent += on_status_change
        try:
            self.ib.cancelOrder(trade.order)
            final_status = await asyncio.wait_for(future, timeout=10)
            logger.info(f"[CANCEL] order_id={order_id} confirmed as: {final_status}")
            return final_status
        except asyncio.TimeoutError:
            logger.error(
                f"[CANCEL] order_id={order_id} cancel confirmation timed out — "
                f"unknown order state. Check TWS manually."
            )
            return "CancelTimeout"
        finally:
            trade.statusEvent -= on_status_change

    async def flatten_position(
        self, symbol: str, quantity: int, action: str
    ) -> dict:
        """
        Closes a partial position immediately using a market order.
        `action` is the ORIGINAL order action — we reverse it to close.
        Returns the IB trade dict for the flatten order so the caller can log the fill.
        """
        close_action = "SELL" if action.upper() == "BUY" else "BUY"
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder(close_action, quantity)
        trade = self.ib.placeOrder(contract, order)
        logger.warning(
            f"[FLATTEN] Closing partial: {close_action} {quantity} {symbol} "
            f"market order — order_id={trade.order.orderId}"
        )
        return {"order_id": trade.order.orderId}

    # ── Open Orders ───────────────────────────────────────────────────────────

    async def get_open_orders(self) -> list[dict]:
        trades = await self.ib.reqOpenOrdersAsync()
        return [
            {
                "order_id": t.order.orderId,
                "symbol":   t.contract.symbol,
                "action":   t.order.action,
                "quantity": t.order.totalQuantity,
                "status":   t.orderStatus.status,
            }
            for t in trades
        ]
