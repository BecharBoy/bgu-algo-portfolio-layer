from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
import asyncpg
from main_backtesting.models import NewsArticle, SentimentResult

from database.backtesting.repositories._shared import SCHEMA, json_text, json_value


async def save_article_set(
    conn: asyncpg.Connection,
    *,
    input_hash: str,
    market_id: str,
    pass_number: int,
    as_of: datetime,
    symbol: str,
    query: str,
    window_start: datetime,
    window_end: datetime,
    query_settings: dict[str, Any],
    articles: list[NewsArticle],
) -> UUID:
    existing = await conn.fetchrow(
        f"SELECT article_set_id FROM {SCHEMA}.historical_article_sets WHERE input_hash = $1",
        input_hash,
    )
    if existing:
        return existing["article_set_id"]
    article_set_id = uuid4()
    for article in articles:
        await conn.execute(
            f"""
            INSERT INTO {SCHEMA}.historical_articles
                (url, title, published_at, domain, article_text)
            VALUES ($1,$2,$3,$4,$5)
            ON CONFLICT (url) DO NOTHING
            """,
            article.url,
            article.title,
            article.published_at,
            article.domain,
            article.text,
        )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_article_sets (
            article_set_id, input_hash, market_id, pass_number, as_of, symbol,
            query, window_start, window_end, query_settings
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::JSONB)
        """,
        article_set_id,
        input_hash,
        market_id,
        pass_number,
        as_of,
        symbol,
        query,
        window_start,
        window_end,
        json_text(query_settings),
    )
    if articles:
        await conn.executemany(
            f"""
            INSERT INTO {SCHEMA}.historical_article_set_items (article_set_id, url)
            VALUES ($1,$2)
            """,
            [(article_set_id, article.url) for article in articles],
        )
    return article_set_id


async def reusable_article_set(
    conn: asyncpg.Connection,
    input_hash: str,
) -> tuple[UUID, list[NewsArticle]] | None:
    row = await conn.fetchrow(
        f"SELECT article_set_id FROM {SCHEMA}.historical_article_sets WHERE input_hash = $1",
        input_hash,
    )
    if row is None:
        return None
    rows = await conn.fetch(
        f"""
        SELECT a.url, a.title, a.published_at, a.domain, a.article_text
        FROM {SCHEMA}.historical_article_set_items i
        JOIN {SCHEMA}.historical_articles a ON a.url = i.url
        WHERE i.article_set_id = $1 ORDER BY a.published_at DESC
        """,
        row["article_set_id"],
    )
    return (
        row["article_set_id"],
        [
            NewsArticle(r["url"], r["title"], r["published_at"], r["domain"], r["article_text"])
            for r in rows
        ],
    )


async def save_sentiment(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    input_hash: str,
    article_set_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    provider: str,
    model_name: str,
    prompt_version: str,
    model_input: dict[str, Any],
    model_output: dict[str, Any],
    result: SentimentResult,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_sentiment_results (
            input_hash, article_set_id, market_id, pass_number, symbol, provider,
            model_name, prompt_version, model_input, model_output, label, score, details
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::JSONB,$10::JSONB,$11,$12,$13::JSONB)
        ON CONFLICT (input_hash) DO NOTHING
        """,
        input_hash,
        article_set_id,
        market_id,
        pass_number,
        symbol,
        provider,
        model_name,
        prompt_version,
        json_text(model_input),
        json_text(model_output),
        result.label,
        result.score,
        json_text(result.details),
    )
    await link_run_sentiment(
        conn,
        run_id=run_id,
        market_id=market_id,
        pass_number=pass_number,
        symbol=symbol,
        provider=provider,
        input_hash=input_hash,
    )


async def link_run_sentiment(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    provider: str,
    input_hash: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_sentiments (
            run_id, market_id, pass_number, symbol, provider, input_hash
        )
        VALUES ($1,$2,$3,$4,$5,$6)
        ON CONFLICT (run_id, market_id, pass_number, symbol, provider)
        DO UPDATE SET input_hash=EXCLUDED.input_hash
        """,
        run_id,
        market_id,
        pass_number,
        symbol.upper(),
        provider,
        input_hash,
    )


async def reusable_sentiment(
    conn: asyncpg.Connection,
    input_hash: str,
) -> SentimentResult | None:
    row = await conn.fetchrow(
        f"SELECT label, score, details FROM {SCHEMA}.historical_sentiment_results WHERE input_hash = $1",
        input_hash,
    )
    if row is None:
        return None
    details = json_value(row["details"])
    return SentimentResult(
        row["label"],
        row["score"],
        sum(item.get("label") == "positive" for item in details),
        sum(item.get("label") == "neutral" for item in details),
        sum(item.get("label") == "negative" for item in details),
        details,
    )


async def sentiment_for_job(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    provider: str,
) -> SentimentResult | None:
    row = await conn.fetchrow(
        f"""
        SELECT s.label, s.score, s.details
        FROM {SCHEMA}.historical_run_sentiments r
        JOIN {SCHEMA}.historical_sentiment_results s ON s.input_hash=r.input_hash
        WHERE r.run_id=$1 AND r.market_id=$2 AND r.pass_number=$3
          AND r.symbol=$4 AND r.provider=$5
        """,
        run_id,
        market_id,
        pass_number,
        symbol.upper(),
        provider,
    )
    if row is None:
        return None
    details = json_value(row["details"])
    return SentimentResult(
        row["label"],
        row["score"],
        sum(item.get("label") == "positive" for item in details),
        sum(item.get("label") == "neutral" for item in details),
        sum(item.get("label") == "negative" for item in details),
        details,
    )
