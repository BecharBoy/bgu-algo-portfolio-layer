from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Literal
from zoneinfo import ZoneInfo

import pandas as pd

from main_backtesting.models import PriceBar

Resolution = Literal["1h", "1d"]

SECTOR_ETFS = {
    "Basic Materials": "XLB",
    "Communication Services": "XLC",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Real Estate": "XLRE",
    "Technology": "XLK",
    "Utilities": "XLU",
}


def _download_prices(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: Resolution,
) -> list[PriceBar]:
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval=resolution,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
    )
    if frame.empty:
        return []
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)

    frame = frame.reset_index()
    timestamp_column = "Datetime" if "Datetime" in frame.columns else "Date"
    timestamps = pd.to_datetime(frame[timestamp_column], utc=True)
    bars: list[PriceBar] = []
    for index, timestamp in enumerate(timestamps):
        row = frame.iloc[index]
        values = [row.get(name) for name in ("Open", "High", "Low", "Close")]
        if any(pd.isna(value) for value in values):
            continue
        bar_timestamp = timestamp.to_pydatetime().astimezone(timezone.utc)
        if resolution == "1d":
            market_date = bar_timestamp.date()
            bar_timestamp = datetime(
                market_date.year,
                market_date.month,
                market_date.day,
                9,
                30,
                tzinfo=ZoneInfo("America/New_York"),
            ).astimezone(timezone.utc)
        bars.append(
            PriceBar(
                timestamp=bar_timestamp,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0) or 0.0),
            )
        )
    return [bar for bar in bars if start <= bar.timestamp < end]


def _download_metadata(symbol: str) -> dict[str, Any]:
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    info = ticker.get_info()
    sector = info.get("sector")
    return {
        "symbol": symbol.upper(),
        "asset_name": info.get("longName") or info.get("shortName"),
        "sector": sector,
        "sector_etf": SECTOR_ETFS.get(sector),
        "quote_type": info.get("quoteType"),
        "exchange": info.get("exchange"),
    }


class YFinanceClient:
    def __init__(self, *, concurrency: int = 4) -> None:
        self.semaphore = asyncio.Semaphore(concurrency)

    async def bars(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
        resolution: Resolution,
    ) -> list[PriceBar]:
        async with self.semaphore:
            return await asyncio.to_thread(_download_prices, symbol.upper(), start, end, resolution)

    async def metadata(self, symbol: str) -> dict[str, Any]:
        async with self.semaphore:
            return await asyncio.to_thread(_download_metadata, symbol.upper())


YFinanceHourlyClient = YFinanceClient


def next_bar_after(bars: list[PriceBar], timestamp: datetime) -> PriceBar | None:
    return next((bar for bar in bars if bar.timestamp > timestamp), None)


def bars_before(bars: list[PriceBar], timestamp: datetime) -> list[PriceBar]:
    return [bar for bar in bars if bar.timestamp < timestamp]


def bars_from(bars: list[PriceBar], timestamp: datetime) -> list[PriceBar]:
    return [bar for bar in bars if bar.timestamp >= timestamp]
