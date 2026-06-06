from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pandas as pd

from main_backtesting.models import ProbabilityPoint, SourceMarket

CLOB_PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"


class PolymarketHistoryClient:
    def __init__(self, *, chunk_days: int = 10) -> None:
        self.chunk_days = chunk_days
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60),
            headers={"User-Agent": "my-traders-backtest/1.0"},
        )

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
                timestamp = datetime.fromtimestamp(float(item["t"]), tz=timezone.utc)
                probability = min(max(float(item["p"]), 0.0), 1.0)
                rows.append((timestamp, probability))
            cursor = chunk_end

        if not rows:
            return []
        frame = pd.DataFrame(rows, columns=["timestamp", "probability"])
        frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
        frame = frame.set_index("timestamp").resample("1h").last().dropna().reset_index()
        return [
            ProbabilityPoint(row.timestamp.to_pydatetime(), float(row.probability))
            for row in frame.itertuples(index=False)
        ]


def probability_as_of(
    probabilities: list[ProbabilityPoint],
    timestamp: datetime,
) -> float | None:
    value = None
    for point in probabilities:
        if point.timestamp > timestamp:
            break
        value = point.probability
    return value

