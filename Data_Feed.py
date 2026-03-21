import pandas as pd
from typing import List


class Data_Feed:

    def __init__(self):
        # TODO: Inject yahoo finance client abstraction.
        # TODO: Add request throttling and retry policy.
        self.default_period = "2y"

    async def fetch_historical_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        # TODO: Return structure should match strategy input contract.
        # TODO: Prefer Dict[str, pd.DataFrame] when fetching multi-ticker history.
        # TODO: Ensure adjusted OHLCV handling is consistent.
        pass

    async def fetch_current_prices(self, tickers: List[str]) -> dict[str, float]:
        # TODO: Fetch latest tradable prices for sizing/execution checks.
        # TODO: Add fallback for missing/partial symbol responses.
        pass

    async def bootstrap_two_year_history(self, tickers: List[str]) -> dict[str, pd.DataFrame]:
        # TODO: Initial bulk download job for two years of daily data.
        # TODO: Persist to DB via idempotent upsert path.
        pass

    async def fetch_latest_daily_bars(self, tickers: List[str]) -> dict[str, pd.DataFrame]:
        # TODO: Daily incremental fetch for most recent completed session.
        # TODO: Keep output schema identical to historical loader.
        pass


