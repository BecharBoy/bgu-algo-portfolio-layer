import asyncio
from ib_async import IB, Stock, MarketOrder
import uuid


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
        positions = await self.ib.reqPositionsAsync()
        return {
            pos.contract.symbol: {
                "quantity": pos.position,
                "avg_cost": pos.avgCost,
            }
            for pos in positions
            if pos.position != 0
        }

    async def place_market_order(self, action: str, symbol: str, quantity: int) -> dict:
        contract = Stock(symbol, "SMART", "USD")
        await self.ib.qualifyContractsAsync(contract)
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        return {"order_id": trade.order.orderId, "status": trade.orderStatus.status}

    async def wait_for_order_fill(
        self, order_id: int, timeout_seconds: int = 30
    ) -> dict | None:
        """
        Wait for a specific order to fill using IB's execDetailsEvent.
        Returns the fill dict once confirmed, or None on timeout.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        def on_exec_details(trade, fill):
            # Only resolve for the specific order we are waiting on
            if fill.execution.orderId == order_id and not future.done():
                future.set_result({
                    "order_id":   fill.execution.orderId,
                    "symbol":     fill.contract.symbol,
                    "action":     fill.execution.side,   # "BOT" or "SLD"
                    "quantity":   fill.execution.shares,
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
            # Always unsubscribe to avoid stale callbacks accumulating
            self.ib.execDetailsEvent -= on_exec_details

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
