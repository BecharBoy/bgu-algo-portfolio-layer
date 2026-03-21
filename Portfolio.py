import asyncio
from typing import List, Dict

class Portfolio:
    def __init__(self, ib_client, db_client, data_feed, total_capital: float):
        self.ib = ib_client
        self.db = db_client
        self.data_feed = data_feed
        self.total_capital = total_capital
        self.strategies = {}

    def add_strategy(self, name, strategy):
        self.strategies[name] = strategy

    async def gather_signals(self, market_data, current_positions) -> List[dict]:
        tasks = [strategy.generate_signals(market_data, current_positions) for strategy in self.strategies.values()]

        results = await asyncio.gather(*tasks)

        return [siganl for sublist in results for siganl in sublist]

    def _apply_risk_management(self, signals: List[Dict]) -> List[Dict]:
        pass

    async def run_cycle(self):
        pass


    def allocate_capital(self, approved_signals: List[Dict], free_cash: float) -> List[Dict]:
        pass

    async def execute_approved_signals(self, orders_to_place: List[Dict]):
        pass

