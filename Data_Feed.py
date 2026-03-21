import pandas as pd
from typing import List


class Data_Feed:

    def __init__(self):
        pass

    async def fetch_historical_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        pass

    async def fetch_current_prices(self, tickers: List[str]) -> dict[str, float]:
        pass


