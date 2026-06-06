from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd

from main_backtesting.models import PriceBar


def _download_hourly(symbol: str, start: datetime, end: datetime) -> list[PriceBar]:
    import yfinance as yf

    frame = yf.download(
        symbol,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1h",
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
        bars.append(
            PriceBar(
                timestamp=timestamp.to_pydatetime().astimezone(timezone.utc),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row.get("Volume", 0.0) or 0.0),
            )
        )
    return [bar for bar in bars if start <= bar.timestamp <= end]


class YFinanceHourlyClient:
    def __init__(self) -> None:
        self._cache: dict[str, list[PriceBar]] = {}

    async def hourly_bars(
        self,
        symbol: str,
        *,
        start: datetime,
        end: datetime,
    ) -> list[PriceBar]:
        normalized = symbol.upper()
        if normalized not in self._cache:
            self._cache[normalized] = await asyncio.to_thread(
                _download_hourly,
                normalized,
                start,
                end,
            )
        return self._cache[normalized]


def next_bar_after(bars: list[PriceBar], timestamp: datetime) -> PriceBar | None:
    return next((bar for bar in bars if bar.timestamp >= timestamp), None)


def bars_before(bars: list[PriceBar], timestamp: datetime) -> list[PriceBar]:
    return [bar for bar in bars if bar.timestamp < timestamp]


def bars_from(bars: list[PriceBar], timestamp: datetime) -> list[PriceBar]:
    return [bar for bar in bars if bar.timestamp >= timestamp]

