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

ETF_BENCHMARKS = {
    "BNO": "USO",
    "DIA": "DIA",
    "EEM": "EEM",
    "EFA": "EFA",
    "EIS": "EIS",
    "EWJ": "EWJ",
    "EWZ": "EWZ",
    "FXI": "FXI",
    "GLD": "GLD",
    "HYG": "HYG",
    "ICLN": "ICLN",
    "IEF": "TLT",
    "ISRA": "EIS",
    "ITA": "XLI",
    "IWM": "IWM",
    "KRE": "KRE",
    "OIH": "XLE",
    "LQD": "LQD",
    "QQQ": "QQQ",
    "RWR": "VNQ",
    "SLV": "SLV",
    "SPY": "SPY",
    "TBT": "TLT",
    "TAN": "ICLN",
    "TLT": "TLT",
    "TNA": "IWM",
    "TQQQ": "QQQ",
    "USO": "USO",
    "VIXY": "VIXY",
    "VOO": "SPY",
    "VTI": "VTI",
    "VNQ": "XLRE",
    "VWO": "VWO",
    "XLB": "XLB",
    "XAR": "ITA",
    "XHB": "XHB",
    "XLC": "XLC",
    "XLE": "XLE",
    "XLF": "XLF",
    "XLI": "XLI",
    "XLK": "XLK",
    "XLP": "XLP",
    "XLRE": "XLRE",
    "XLU": "XLU",
    "XLV": "XLV",
    "XLY": "XLY",
}


def benchmark_symbol(
    symbol: str,
    *,
    quote_type: str | None,
    sector: str | None,
) -> str | None:
    if (quote_type or "").upper() == "ETF":
        return ETF_BENCHMARKS.get(symbol.upper())
    return SECTOR_ETFS.get(sector or "")


def yahoo_request_bounds(start: datetime, end: datetime) -> tuple[datetime, datetime]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Yahoo price request boundaries must be timezone-aware")
    if start >= end:
        raise ValueError("Yahoo price request start must be before end")

    request_start = start.astimezone(timezone.utc).replace(microsecond=0)
    request_end = end.astimezone(timezone.utc)
    if request_end.microsecond:
        request_end = (request_end + timedelta(seconds=1)).replace(microsecond=0)
    return request_start, request_end


def _download_prices(
    symbol: str,
    start: datetime,
    end: datetime,
    resolution: Resolution,
) -> list[PriceBar]:
    import yfinance as yf

    request_start, request_end = yahoo_request_bounds(start, end)
    frame = yf.download(
        symbol,
        start=request_start,
        end=request_end,
        interval=resolution,
        auto_adjust=False,
        prepost=False,
        progress=False,
        threads=False,
        ignore_tz=False,
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
    quote_type = info.get("quoteType")
    return {
        "symbol": symbol.upper(),
        "asset_name": info.get("longName") or info.get("shortName"),
        "sector": sector,
        "sector_etf": SECTOR_ETFS.get(sector),
        "benchmark_symbol": benchmark_symbol(
            symbol,
            quote_type=quote_type,
            sector=sector,
        ),
        "quote_type": quote_type,
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
