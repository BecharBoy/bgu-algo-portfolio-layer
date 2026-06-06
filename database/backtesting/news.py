from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any

import httpx

from main_backtesting.models import Asset, NewsArticle, SourceMarket

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


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
        if not self.hidden_depth:
            stripped = data.strip()
            if stripped:
                self.parts.append(stripped)


def _parse_gdelt_timestamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def _article_text(document: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(document)
    text = html.unescape(" ".join(parser.parts))
    return re.sub(r"\s+", " ", text).strip()[:20_000]


class GdeltNewsClient:
    def __init__(self, max_articles: int = 9) -> None:
        self.max_articles = max_articles
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(45),
            follow_redirects=True,
            headers={"User-Agent": "my-traders-backtest/1.0"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def articles(
        self,
        *,
        market: SourceMarket,
        asset: Asset,
        start: datetime,
        end: datetime,
    ) -> list[NewsArticle]:
        query = f'"{asset.asset_name}" OR {asset.symbol} "{market.event_title}"'
        response = await self.client.get(
            GDELT_DOC_URL,
            params={
                "query": query,
                "mode": "ArtList",
                "format": "json",
                "maxrecords": self.max_articles,
                "startdatetime": start.strftime("%Y%m%d%H%M%S"),
                "enddatetime": end.strftime("%Y%m%d%H%M%S"),
                "sort": "DateDesc",
            },
        )
        response.raise_for_status()
        raw_articles: list[dict[str, Any]] = response.json().get("articles") or []

        articles: list[NewsArticle] = []
        for raw in raw_articles[: self.max_articles]:
            url = str(raw.get("url") or "")
            seen_date = str(raw.get("seendate") or "")
            if not url or not seen_date:
                continue
            published_at = _parse_gdelt_timestamp(seen_date)
            if not start <= published_at <= end:
                continue
            try:
                article_response = await self.client.get(url)
                article_response.raise_for_status()
                text = _article_text(article_response.text)
            except httpx.HTTPError:
                text = ""
            if not text:
                text = str(raw.get("title") or "")
            articles.append(
                NewsArticle(
                    url=url,
                    title=str(raw.get("title") or ""),
                    published_at=published_at,
                    domain=raw.get("domain"),
                    text=text,
                )
            )
        return articles

