from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from main_backtesting.models import ProbabilityPoint, SourceMarket

CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"
DATA_TRADES_URL = "https://data-api.polymarket.com/trades"
DATA_TRADES_PAGE_SIZE = 10_000
DATA_TRADES_MAX_OFFSET = 10_000
VOLUME_RETRY_ATTEMPTS = 3
VOLUME_RETRYABLE_STATUSES = {408, 429, 500, 502, 503, 504}


def hourly_as_of_points(
    rows: list[tuple[datetime, float]],
    *,
    history_end: datetime,
    volume_rows: list[tuple[datetime, float]] | None = None,
) -> list[ProbabilityPoint]:
    if not rows:
        return []
    frame = pd.DataFrame(rows, columns=["source_ts", "probability"])
    frame = frame.drop_duplicates("source_ts").sort_values("source_ts")
    first_hour = frame["source_ts"].min().ceil("h")
    final_hour = pd.Timestamp(history_end - timedelta(microseconds=1)).floor("h")
    if first_hour > final_hour:
        return []
    hours = pd.DataFrame(
        {"hour_ts": pd.date_range(start=first_hour, end=final_hour, freq="1h")}
    )
    hourly = pd.merge_asof(
        hours,
        frame,
        left_on="hour_ts",
        right_on="source_ts",
        direction="backward",
    ).dropna(subset=["source_ts", "probability"])
    completed_hour_volumes: dict[datetime, float] | None = None
    if volume_rows is not None:
        completed_hour_volumes = {}
        for timestamp, notional in volume_rows:
            completed_at = timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(
                hours=1
            )
            completed_hour_volumes[completed_at] = (
                completed_hour_volumes.get(completed_at, 0.0) + notional
            )
    return [
        ProbabilityPoint(
            timestamp=row["hour_ts"].to_pydatetime(),
            probability=float(row["probability"]),
            source_timestamp=row["source_ts"].to_pydatetime(),
            available_at=row["source_ts"].to_pydatetime(),
            volume_usdc=(
                completed_hour_volumes.get(row["hour_ts"].to_pydatetime(), 0.0)
                if completed_hour_volumes is not None
                else None
            ),
        )
        for _, row in hourly.iterrows()
    ]


class PolymarketHistoryClient:
    def __init__(self, *, chunk_days: int = 10) -> None:
        self.chunk_days = chunk_days
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60),
            headers={"User-Agent": "my-traders-backtest/2.0"},
        )
        self.volume_status = "unknown"
        self.volume_error: str | None = None

    async def close(self) -> None:
        await self.client.aclose()

    async def hourly_probabilities(
        self,
        market: SourceMarket,
        *,
        start: datetime,
        end: datetime,
    ) -> list[ProbabilityPoint]:
        history_start = max(start, market.created_at)
        history_end = min(end, market.end_at)
        if history_start >= history_end:
            return []

        rows: list[tuple[datetime, float]] = []
        self.volume_status = "unknown"
        self.volume_error = None
        cursor = history_start
        while cursor < history_end:
            chunk_end = min(cursor + timedelta(days=self.chunk_days), history_end)
            response = await self.client.get(
                CLOB_PRICE_HISTORY_URL,
                params={
                    "market": market.yes_token_id,
                    "startTs": int(cursor.timestamp()),
                    "endTs": int(chunk_end.timestamp()),
                    "fidelity": 60,
                },
            )
            response.raise_for_status()
            for item in response.json().get("history") or []:
                source_ts = datetime.fromtimestamp(float(item["t"]), tz=timezone.utc)
                probability = min(max(float(item["p"]), 0.0), 1.0)
                rows.append((source_ts, probability))
            cursor = chunk_end

        volume_rows = (
            await self._trade_volumes(
                condition_id=market.condition_id,
                start=history_start,
                end=history_end,
            )
            if market.condition_id
            else None
        )
        if not market.condition_id:
            self.volume_status = "unavailable"
            self.volume_error = "market has no condition_id"
        return hourly_as_of_points(
            rows,
            history_end=min(history_end, end),
            volume_rows=volume_rows,
        )

    async def _trade_volumes(
        self,
        *,
        condition_id: str,
        start: datetime,
        end: datetime,
    ) -> list[tuple[datetime, float]] | None:
        rows: list[tuple[datetime, float]] = []
        offset = 0
        while True:
            response = None
            for attempt in range(VOLUME_RETRY_ATTEMPTS):
                try:
                    response = await self.client.get(
                        DATA_TRADES_URL,
                        params={
                            "market": condition_id,
                            "limit": DATA_TRADES_PAGE_SIZE,
                            "offset": offset,
                            "takerOnly": "true",
                        },
                    )
                    if response.status_code not in VOLUME_RETRYABLE_STATUSES:
                        response.raise_for_status()
                        break
                except httpx.HTTPError as error:
                    if attempt == VOLUME_RETRY_ATTEMPTS - 1:
                        self.volume_status = "retryable_failure"
                        self.volume_error = str(error)
                        return None
                if attempt < VOLUME_RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(2 ** attempt)
            if response is None or response.status_code in VOLUME_RETRYABLE_STATUSES:
                self.volume_status = "retryable_failure"
                self.volume_error = (
                    f"Polymarket volume request failed after {VOLUME_RETRY_ATTEMPTS} attempts"
                )
                return None
            batch = response.json()
            if not isinstance(batch, list):
                self.volume_status = "invalid_response"
                self.volume_error = "Polymarket Data API trades response must be a list"
                return None
            for item in batch:
                timestamp = datetime.fromtimestamp(float(item["timestamp"]), tz=timezone.utc)
                if not start <= timestamp < end:
                    continue
                size = float(item.get("size") or 0.0)
                price = float(item.get("price") or 0.0)
                rows.append((timestamp, max(size * price, 0.0)))
            if len(batch) < DATA_TRADES_PAGE_SIZE:
                self.volume_status = "complete"
                self.volume_error = None
                return rows
            if offset >= DATA_TRADES_MAX_OFFSET:
                self.volume_status = "incomplete"
                self.volume_error = (
                    "Polymarket Data API trade history exceeded the supported "
                    f"{DATA_TRADES_MAX_OFFSET + DATA_TRADES_PAGE_SIZE:,}-row window "
                    f"for condition {condition_id}"
                )
                return None
            offset += DATA_TRADES_PAGE_SIZE


def probability_as_of(
    probabilities: list[ProbabilityPoint],
    timestamp: datetime,
) -> float | None:
    value = None
    for point in probabilities:
        available_at = point.available_at or point.source_timestamp or point.timestamp
        if available_at > timestamp:
            break
        value = point.probability
    return value
