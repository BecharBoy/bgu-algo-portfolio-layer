import asyncio
from ib_async import IB, Stock, MarketOrder

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

    async def wait_for_order_updates(self, timeout_seconds: int = 15) -> list[dict]:
        await asyncio.sleep(timeout_seconds)
        return [
            {
                "order_id":   f.execution.orderId,
                "symbol":     f.contract.symbol,
                "quantity":   f.execution.shares,
                "fill_price": f.execution.price,
                "filled_at":  f.execution.time,
            }
            for f in self.ib.fills()
        ]

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
