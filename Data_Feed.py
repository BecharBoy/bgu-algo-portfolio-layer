import logging

import pandas as pd
from typing import List
import yfinance as yf
from datetime import date

class Data_Feed:

    def __init__(self):
        self.tickers = pd.read_csv("tickers.csv")
        self.today = date.today().strftime("%Y-%m-%d")
        # TODO: Inject yahoo finance client abstraction.
        # TODO: Add request throttling and retry policy.

    """ didnt understand what this function suppose to do ?"""
    async def fetch_historical_data(self, tickers: List[str], period: str = "2y", interval='1d') -> bool:
        # TODO: Return structure should match strategy input contract.
        # TODO: Prefer Dict[str, pd.DataFrame] when fetching multi-ticker history.
        # TODO: Ensure adjusted OHLCV handling is consistent.
       pass

    async def fetch_current_prices(self, tickers: List[str], period: str = "1d") -> dict[str, pd.DataFrame]:

        # TODO: Fetch latest tradable prices for sizing/execution checks.
        # TODO: Add fallback for missing/partial symbol responses.
        todays_data = {}
        for tick in tickers:
            todays_data[tick] = yf.download(tick, period=period)
            if not todays_data:
                logging.log(f"failed to get {tick} data for {self.today}")


        return todays_data

    async def bootstrap_two_year_history(self, tickers: List[str],
                                         period: str = "2y", interval='1d') -> dict[str, pd.DataFrame]:
        # TODO: Initial bulk download job for two years of daily data.
        # TODO: Persist to DB via idempotent upsert path.
        historical_data = {}
        for tick in tickers:
            historical_data[tick] = yf.download(tick, period=period, interval=interval)

        return historical_data

    """ didnt understand what this function suppose to do ?"""
    async def fetch_latest_daily_bars(self, tickers: List[str]) -> dict[str, pd.DataFrame]:
        # TODO: Daily incremental fetch for most recent completed session.
        # TODO: Keep output schema identical to historical loader.
        pass


