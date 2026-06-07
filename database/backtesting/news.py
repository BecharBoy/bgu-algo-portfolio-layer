from __future__ import annotations

import asyncio
import html
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from database.db_connection import connect
from main_backtesting.models import Asset, NewsArticle, SourceMarket

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_SEARCH_SCHEDULE = "doc_search"
GDELT_ADVISORY_LOCK_KEY = 0x4744454C54534541
ConnectionFactory = Callable[[], Awaitable[Any]]


class GdeltRateLimitError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        response_body: str,
        response_headers: dict[str, str],
        request_timestamp: datetime,
        previous_request_timestamp: datetime | None,
        previous_completion_timestamp: datetime | None,
        minimum_interval_seconds: float,
    ) -> None:
        self.status_code = status_code
        self.response_body = response_body
        self.response_headers = response_headers
        self.request_timestamp = request_timestamp
        self.previous_request_timestamp = previous_request_timestamp
        self.previous_completion_timestamp = previous_completion_timestamp
        self.minimum_interval_seconds = minimum_interval_seconds
        super().__init__(
            "GDELT search failed with "
            f"HTTP status={status_code}; "
            f"response_body={response_body!r}; "
            f"request_timestamp={request_timestamp.isoformat()}; "
            "previous_known_gdelt_request_timestamp="
            f"{previous_request_timestamp.isoformat() if previous_request_timestamp else None}; "
            "previous_known_gdelt_completion_timestamp="
            f"{previous_completion_timestamp.isoformat() if previous_completion_timestamp else None}; "
            f"configured_minimum_interval_seconds={minimum_interval_seconds}; "
            f"response_headers={response_headers!r}"
        )


@dataclass
class GdeltRequestLease:
    connection: Any
    request_timestamp: datetime
    previous_request_timestamp: datetime | None
    previous_completion_timestamp: datetime | None
    minimum_interval_seconds: float

    async def complete(self, status_code: int) -> None:
        await self.connection.execute(
            """
            UPDATE checking_relevant_events.gdelt_request_schedule
            SET last_request_completed_at = clock_timestamp(),
                last_status = $2,
                updated_at = clock_timestamp()
            WHERE schedule_name = $1
            """,
            GDELT_SEARCH_SCHEDULE,
            status_code,
        )


class PostgresGdeltSearchLimiter:
    def __init__(
        self,
        minimum_interval_seconds: float,
        *,
        connection_factory: ConnectionFactory = connect,
    ) -> None:
        if minimum_interval_seconds < 0:
            raise ValueError("minimum_interval_seconds cannot be negative")
        self.minimum_interval_seconds = minimum_interval_seconds
        self.connection_factory = connection_factory

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[GdeltRequestLease]:
        connection = await self.connection_factory()
        lock_acquired = False
        try:
            await connection.execute(
                "SELECT pg_advisory_lock($1)",
                GDELT_ADVISORY_LOCK_KEY,
            )
            lock_acquired = True
            await connection.execute(
                """
                INSERT INTO checking_relevant_events.gdelt_request_schedule (schedule_name)
                VALUES ($1)
                ON CONFLICT (schedule_name) DO NOTHING
                """,
                GDELT_SEARCH_SCHEDULE,
            )
            schedule = await connection.fetchrow(
                """
                SELECT last_request_started_at,
                       last_request_completed_at,
                       clock_timestamp() AS database_now
                FROM checking_relevant_events.gdelt_request_schedule
                WHERE schedule_name = $1
                """,
                GDELT_SEARCH_SCHEDULE,
            )
            previous_request_timestamp = schedule["last_request_started_at"]
            previous_completion_timestamp = schedule["last_request_completed_at"]
            known_timestamps = [
                timestamp
                for timestamp in (
                    previous_request_timestamp,
                    previous_completion_timestamp,
                )
                if timestamp is not None
            ]
            anchor = max(known_timestamps) if known_timestamps else None
            delay = (
                (
                    anchor
                    + timedelta(seconds=self.minimum_interval_seconds)
                    - schedule["database_now"]
                ).total_seconds()
                if anchor is not None
                else 0.0
            )
            if delay > 0:
                await asyncio.sleep(delay)
            request_timestamp = await connection.fetchval(
                """
                UPDATE checking_relevant_events.gdelt_request_schedule
                SET last_request_started_at = clock_timestamp(),
                    last_status = NULL,
                    updated_at = clock_timestamp()
                WHERE schedule_name = $1
                RETURNING last_request_started_at
                """,
                GDELT_SEARCH_SCHEDULE,
            )
            yield GdeltRequestLease(
                connection=connection,
                request_timestamp=request_timestamp,
                previous_request_timestamp=previous_request_timestamp,
                previous_completion_timestamp=previous_completion_timestamp,
                minimum_interval_seconds=self.minimum_interval_seconds,
            )
        finally:
            if lock_acquired:
                await connection.execute(
                    "SELECT pg_advisory_unlock($1)",
                    GDELT_ADVISORY_LOCK_KEY,
                )
            await connection.close()


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"}:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth and data.strip():
            self.parts.append(data.strip())


def _parse_gdelt_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _article_text(document: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(document)
    text = html.unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()


class GdeltNewsClient:
    def __init__(
        self,
        max_articles: int = 9,
        *,
        search_concurrency: int = 8,
        article_concurrency: int = 12,
        minimum_search_interval_seconds: float = 5.5,
        search_limiter: PostgresGdeltSearchLimiter | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.max_articles = max_articles
        self.search_semaphore = asyncio.Semaphore(search_concurrency)
        self.article_semaphore = asyncio.Semaphore(article_concurrency)
        self.search_limiter = search_limiter or PostgresGdeltSearchLimiter(
            minimum_search_interval_seconds
        )
        self.client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(45),
            follow_redirects=True,
            headers={"User-Agent": "my-traders-backtest/2.0"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    @staticmethod
    def query(market: SourceMarket, asset: Asset) -> str:
        return f'"{asset.asset_name}" OR {asset.symbol} "{market.event_title}"'

    async def _search(
        self,
        *,
        query: str,
        start: datetime,
        end: datetime,
        max_records: int,
    ) -> list[dict[str, Any]]:
        async with self.search_semaphore:
            async with self.search_limiter.reserve() as request:
                response = await self.client.get(
                    GDELT_DOC_URL,
                    params={
                        "query": query,
                        "mode": "ArtList",
                        "format": "json",
                        "maxrecords": max_records,
                        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                        "sort": "DateDesc",
                    },
                )
                await request.complete(response.status_code)
                if response.status_code == 429:
                    raise GdeltRateLimitError(
                        status_code=response.status_code,
                        response_body=response.text,
                        response_headers=dict(response.headers),
                        request_timestamp=request.request_timestamp,
                        previous_request_timestamp=request.previous_request_timestamp,
                        previous_completion_timestamp=request.previous_completion_timestamp,
                        minimum_interval_seconds=request.minimum_interval_seconds,
                    )
                response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles") or []
        if not isinstance(articles, list):
            raise ValueError("GDELT response articles field is not a list")
        return articles

    async def healthcheck(self, *, as_of: datetime) -> int:
        articles = await self._search(
            query='"financial markets"',
            start=as_of - timedelta(days=1),
            end=as_of,
            max_records=1,
        )
        return len(articles)

    async def _download_article(
        self,
        raw: dict[str, Any],
        *,
        start: datetime,
        end: datetime,
    ) -> NewsArticle | None:
        url = str(raw.get("url") or "")
        seen_date = str(raw.get("seendate") or "")
        if not url or not seen_date:
            raise ValueError("GDELT article result is missing url or seendate")
        published_at = _parse_gdelt_timestamp(seen_date)
        if not start <= published_at <= end:
            return None
        async with self.article_semaphore:
            response = await self.client.get(url)
            response.raise_for_status()
        text = _article_text(response.text)
        if not text:
            raise ValueError(f"Downloaded GDELT article has no extractable text: {url}")
        return NewsArticle(
            url=url,
            title=str(raw.get("title") or ""),
            published_at=published_at,
            domain=raw.get("domain"),
            text=text,
        )

    async def articles(
        self,
        *,
        market: SourceMarket,
        asset: Asset,
        start: datetime,
        end: datetime,
    ) -> list[NewsArticle]:
        query = self.query(market, asset)
        raw_articles = await self._search(
            query=query,
            start=start,
            end=end,
            max_records=self.max_articles,
        )
        downloaded = await asyncio.gather(
            *[
                self._download_article(raw, start=start, end=end)
                for raw in raw_articles[: self.max_articles]
            ]
        )
        return [article for article in downloaded if article is not None]
