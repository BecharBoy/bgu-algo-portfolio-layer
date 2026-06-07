from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

import asyncpg

from main_backtesting.models import (
    NewsArticle,
    SentimentResult,
    SourceEvent,
    SourceMarket,
    ThresholdPass,
    Trade,
)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def _json_array(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return json.loads(value or "[]")


def _yes_token_and_outcome(raw_market: dict[str, Any]) -> tuple[str | None, str | None]:
    outcomes = [str(value) for value in _json_array(raw_market.get("outcomes"))]
    token_ids = [str(value) for value in _json_array(raw_market.get("clobTokenIds"))]
    prices = [float(value) for value in _json_array(raw_market.get("outcomePrices"))]
    by_outcome = {outcome.lower(): index for index, outcome in enumerate(outcomes)}

    yes_index = by_outcome.get("yes")
    yes_token = (
        token_ids[yes_index]
        if yes_index is not None and yes_index < len(token_ids)
        else None
    )
    final_outcome = None
    if raw_market.get("closed") and len(prices) == len(outcomes) and prices:
        winning_index = max(range(len(prices)), key=prices.__getitem__)
        if prices[winning_index] >= 0.95:
            final_outcome = outcomes[winning_index]
    return yes_token, final_outcome


async def candidate_events(
    conn: asyncpg.Connection,
    *,
    start: datetime,
    end: datetime,
    minimum_days_remaining: float,
    maximum_days_remaining: float,
    included_tags: list[str],
    excluded_tags: list[str],
    limit: int = 0,
) -> list[SourceEvent]:
    rows = await conn.fetch(
        """
        SELECT event_id, title, created_at, end_at, tags, matched_tags
        FROM checking_relevant_events.source_events
        WHERE created_at IS NOT NULL
          AND end_at IS NOT NULL
          AND created_at < $2
          AND end_at > $1
          AND tags && $5::TEXT[]
          AND NOT (tags && $6::TEXT[])
          AND end_at > created_at + ($3 * INTERVAL '1 day')
          AND end_at <= created_at + ($4 * INTERVAL '1 day')
        ORDER BY created_at, event_id
        """,
        start,
        end,
        minimum_days_remaining,
        maximum_days_remaining,
        included_tags,
        excluded_tags,
    )
    events = [
        SourceEvent(
            event_id=row["event_id"],
            title=row["title"],
            created_at=row["created_at"],
            end_at=row["end_at"],
            tags=list(row["tags"]),
            matched_tags=list(row["matched_tags"]),
        )
        for row in rows
    ]
    return events[:limit] if limit else events


async def event_markets(
    conn: asyncpg.Connection,
    event: SourceEvent,
) -> list[SourceMarket]:
    rows = await conn.fetch(
        """
        SELECT market_id, event_id, question, created_at, end_at, raw_market
        FROM checking_relevant_events.source_questions
        WHERE event_id = $1
          AND question IS NOT NULL
          AND created_at IS NOT NULL
          AND end_at IS NOT NULL
        ORDER BY created_at, market_id
        """,
        event.event_id,
    )
    markets: list[SourceMarket] = []
    for row in rows:
        raw = _json_object(row["raw_market"])
        yes_token, final_outcome = _yes_token_and_outcome(raw)
        if not yes_token:
            continue
        markets.append(
            SourceMarket(
                market_id=row["market_id"],
                event_id=event.event_id,
                event_title=event.title,
                question=row["question"],
                created_at=row["created_at"],
                end_at=row["end_at"],
                tags=event.tags,
                raw_market=raw,
                yes_token_id=yes_token,
                condition_id=raw.get("conditionId") or raw.get("condition_id"),
                final_outcome=final_outcome,
            )
        )
    return markets


async def create_run(conn: asyncpg.Connection, run_id: UUID, config: dict[str, Any]) -> None:
    await conn.execute(
        "INSERT INTO checking_relevant_events.backtest_runs "
        "(run_id, status, config) VALUES ($1, 'running', $2::JSONB)",
        run_id,
        json_text(config),
    )


async def finish_run(
    conn: asyncpg.Connection,
    run_id: UUID,
    *,
    status: str,
    error: str | None = None,
) -> None:
    await conn.execute(
        """
        UPDATE checking_relevant_events.backtest_runs
        SET status = $2, error = $3, finished_at = NOW()
        WHERE run_id = $1
        """,
        run_id,
        status,
        error,
    )


async def save_event_decision(
    conn: asyncpg.Connection,
    run_id: UUID,
    event: SourceEvent,
    relevant: bool,
    reason: str,
    raw_output: dict[str, Any],
) -> None:
    await conn.execute(
        """
        INSERT INTO checking_relevant_events.backtest_event_decisions
            (run_id, event_id, event_title, relevant, reason, raw_output)
        VALUES ($1, $2, $3, $4, $5, $6::JSONB)
        """,
        run_id,
        event.event_id,
        event.title,
        relevant,
        reason,
        json_text(raw_output),
    )


async def save_passes(
    conn: asyncpg.Connection,
    run_id: UUID,
    market: SourceMarket,
    passes: list[ThresholdPass],
) -> None:
    if not passes:
        return
    await conn.executemany(
        """
        INSERT INTO checking_relevant_events.backtest_market_passes (
            run_id, market_id, event_id, question, pass_number,
            above_at, above_probability, fell_below_at,
            fell_below_probability, final_outcome
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)
        """,
        [
            (
                run_id,
                market.market_id,
                market.event_id,
                market.question,
                item.pass_number,
                item.above_at,
                item.above_probability,
                item.fell_below_at,
                item.fell_below_probability,
                market.final_outcome,
            )
            for item in passes
        ],
    )


async def save_asset_world(
    conn: asyncpg.Connection,
    run_id: UUID,
    market_id: str,
    model_name: str,
    raw_output: dict[str, Any],
) -> None:
    await conn.execute(
        """
        INSERT INTO checking_relevant_events.backtest_asset_worlds
            (run_id, market_id, model_name, raw_output)
        VALUES ($1, $2, $3, $4::JSONB)
        """,
        run_id,
        market_id,
        model_name,
        json_text(raw_output),
    )


async def save_articles(
    conn: asyncpg.Connection,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    articles: list[NewsArticle],
) -> None:
    if not articles:
        return
    await conn.executemany(
        """
        INSERT INTO checking_relevant_events.backtest_news_articles (
            run_id, market_id, pass_number, symbol, url, title,
            published_at, domain, article_text
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT DO NOTHING
        """,
        [
            (
                run_id,
                market_id,
                pass_number,
                symbol,
                article.url,
                article.title,
                article.published_at,
                article.domain,
                article.text,
            )
            for article in articles
        ],
    )


async def save_sentiment(
    conn: asyncpg.Connection,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    provider: str,
    result: SentimentResult,
) -> None:
    await conn.execute(
        """
        INSERT INTO checking_relevant_events.backtest_sentiment_results (
            run_id, market_id, pass_number, symbol, provider,
            label, score, article_count, details
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::JSONB)
        """,
        run_id,
        market_id,
        pass_number,
        symbol,
        provider,
        result.label,
        result.score,
        len(result.details),
        json_text(result.details),
    )


async def save_skip(
    conn: asyncpg.Connection,
    run_id: UUID,
    *,
    stage: str,
    reason: str,
    event_id: str | None = None,
    market_id: str | None = None,
    pass_number: int | None = None,
    symbol: str | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO checking_relevant_events.backtest_skips (
            run_id, event_id, market_id, pass_number, symbol, stage, reason
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        """,
        run_id,
        event_id,
        market_id,
        pass_number,
        symbol,
        stage,
        reason[:2_000],
    )


async def save_trade(conn: asyncpg.Connection, trade: Trade) -> None:
    await conn.execute(
        """
        INSERT INTO checking_relevant_events.backtest_trades (
            trade_id, run_id, market_id, event_id, question, symbol, asset_name,
            pass_number, trigger_at, entry_at, entry_price, quantity,
            entry_commission, initial_stop, exit_at, exit_price, exit_commission,
            exit_reason, final_mark_price, maximum_price, minimum_price,
            final_outcome, gross_profit, net_profit, maximum_profit,
            maximum_loss, stop_history, graph_path
        )
        VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,
            $19,$20,$21,$22,$23,$24,$25,$26,$27::JSONB,$28
        )
        """,
        trade.trade_id,
        trade.run_id,
        trade.market_id,
        trade.event_id,
        trade.question,
        trade.symbol,
        trade.asset_name,
        trade.pass_number,
        trade.trigger_at,
        trade.entry_at,
        trade.entry_price,
        trade.quantity,
        trade.entry_commission,
        trade.initial_stop,
        trade.exit_at,
        trade.exit_price,
        trade.exit_commission,
        trade.exit_reason,
        trade.final_mark_price,
        trade.maximum_price,
        trade.minimum_price,
        trade.final_outcome,
        trade.gross_profit,
        trade.net_profit,
        trade.maximum_profit,
        trade.maximum_loss,
        json_text(trade.stop_history),
        trade.graph_path,
    )
