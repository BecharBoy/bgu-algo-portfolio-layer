import logging
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
from typing import List
from config import Settings
from DB import DB

class Data_Feed:

    def __init__(self, setting: Settings, db: DB):
        self.universe = setting.universe
        self.db = db

    async def fetch_current_prices(self) -> dict:
        prices = {}
        for ticker in self.universe:
            df = yf.download(ticker, period="1d", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                logging.warning(f"No live price for {ticker}")
                continue
            prices[ticker] = float(df["Close"].iloc[-1])
        return prices


    async def bootstrap_two_year_history(self) -> None:
        for ticker in self.universe:
            df = yf.download(ticker, period="2y", interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                logging.warning(f"No data returned for {ticker} during bootstrap")
                continue
            bars = self._df_to_bars(df)
            await self.db.upsert_ohlcv_bars(ticker, bars)


    async def daily_incremental_update(self) -> None:
        yesterday = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
        today = date.today().strftime("%Y-%m-%d")
        for ticker in self.universe:
            df = yf.download(ticker, start=yesterday, end=today, interval="1d", auto_adjust=True, progress=False)
            if df.empty:
                logging.warning(f"No bar returned for {ticker} on {yesterday}")
                continue
            bars = self._df_to_bars(df)
            await self.db.upsert_ohlcv_bars(ticker, bars)

    def _df_to_bars(self, df: pd.DataFrame) -> List[dict]:
        bars = []
        for idx, row in df.iterrows():
            bars.append({
                "date":   str(idx.date()),
                "open":   float(row["Open"]),
                "high":   float(row["High"]),
                "low":    float(row["Low"]),
                "close":  float(row["Close"]),
                "volume": int(row["Volume"]),
            })
        return bars


