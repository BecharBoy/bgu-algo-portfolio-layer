


class IB_Connect:
    def __init__(self, host, port, client_id):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()

    async def connect(self):
        pass

    def disconnect(self):
        pass

    async def get_account_summary(self) -> dict:
        pass

    async def get_positions(self) -> dict:
        pass

    async def place_market_order(self, action: str, symbol: str, quantity: int):
        pass

    async def place_bracket_order(self, action: str, symbol: str, quantity: int, take_profit: float, stop_loss: float):
        pass
