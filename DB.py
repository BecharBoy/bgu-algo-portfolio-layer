from typing import List, Dict

class DB:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string


    async def save_signals(self, signals: List[Dict]):
        pass


    async def log_trade_execution(self, trade_data: Dict):
        pass

    async def update_account_snapshot(self, nlv: float, cash: float):
        pass

    async def get_open_positions_from_db(self):
        pass