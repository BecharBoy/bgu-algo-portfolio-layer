from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

import pandas as pd
from pandas.tseries.offsets import BDay

if TYPE_CHECKING:
    import asyncpg


REQUIRED_SOURCE_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
INTERNAL_COLUMNS = ["date", "Open", "High", "Low", "Close", "Volume"]


def _normalize_ohlcv_frame(df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    lowered = {column.lower(): column for column in df.columns}
    missing = REQUIRED_SOURCE_COLUMNS - set(lowered)
    if missing:
        raise ValueError(f"{source_name}: missing required columns {sorted(missing)}")

    normalized = df.rename(
        columns={
            lowered["date"]: "date",
            lowered["open"]: "Open",
            lowered["high"]: "High",
            lowered["low"]: "Low",
            lowered["close"]: "Close",
            lowered["volume"]: "Volume",
        }
    )[INTERNAL_COLUMNS].copy()

    normalized["date"] = pd.to_datetime(normalized["date"]).dt.normalize()
    normalized["Open"] = normalized["Open"].astype(float)
    normalized["High"] = normalized["High"].astype(float)
    normalized["Low"] = normalized["Low"].astype(float)
    normalized["Close"] = normalized["Close"].astype(float)
    normalized["Volume"] = normalized["Volume"].astype(float)

    normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def _validate_date_range(start_date: str, end_date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()
    if end_ts < start_ts:
        raise ValueError(f"end_date {end_date} is earlier than start_date {start_date}")
    return start_ts, end_ts


def _bucket_rows_by_ticker(rows: Iterable[object]) -> dict[str, list[dict]]:
    bucketed: dict[str, list[dict]] = {}
    for row in rows:
        bucketed.setdefault(row["ticker"], []).append(
            {
                "date": row["date"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row["volume"],
            }
        )
    return bucketed


async def load_from_db(
    tickers: list[str],
    start_date: str,
    end_date: str,
    lookback_days: int = 60,
    connection_string: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load daily OHLCV bars from PostgreSQL using the same underlying table
    as the live system, then normalize to the strategy-facing schema:
    `date`, `Open`, `High`, `Low`, `Close`, `Volume`.
    """
    if not tickers:
        return {}

    start_ts, end_ts = _validate_date_range(start_date, end_date)
    warmup_start = (start_ts - BDay(lookback_days)).normalize()

    dsn = connection_string or os.environ.get("DB_CONNECTION_STRING")
    if not dsn:
        raise ValueError("DB connection string is required for load_from_db()")

    import asyncpg

    conn = await asyncpg.connect(dsn=dsn)
    try:
        rows = await conn.fetch(
            """
            SELECT ticker, date, open, high, low, close, volume
            FROM ohlcv_bars
            WHERE ticker = ANY($1::text[])
              AND date >= $2::date
              AND date <= $3::date
            ORDER BY ticker ASC, date ASC
            """,
            tickers,
            warmup_start.date(),
            end_ts.date(),
        )
    finally:
        await conn.close()

    grouped_rows = _bucket_rows_by_ticker(rows)
    data: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        ticker_rows = grouped_rows.get(ticker, [])
        if not ticker_rows:
            continue
        frame = pd.DataFrame(ticker_rows)
        normalized = _normalize_ohlcv_frame(frame, source_name=f"db:{ticker}")
        data[ticker] = normalized

    return data


def load_from_csv(path: str, ticker: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load OHLCV bars from either:
    - a single CSV file, optionally naming the ticker explicitly
    - a directory of CSV files, using each filename stem as the ticker
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{path} does not exist")

    if csv_path.is_file():
        inferred_ticker = ticker or csv_path.stem.upper()
        frame = pd.read_csv(csv_path)
        return {inferred_ticker: _normalize_ohlcv_frame(frame, source_name=str(csv_path))}

    data: dict[str, pd.DataFrame] = {}
    for file_path in sorted(csv_path.glob("*.csv")):
        inferred_ticker = file_path.stem.upper()
        frame = pd.read_csv(file_path)
        data[inferred_ticker] = _normalize_ohlcv_frame(frame, source_name=str(file_path))

    if not data:
        raise ValueError(f"{path} does not contain any CSV files")

    return data
