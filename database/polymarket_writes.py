from __future__ import annotations

from datetime import datetime
from typing import Sequence
from uuid import UUID

import asyncpg

from database.polymarket_common import (
    EventRecord,
    MarketRecord,
    event_tag_labels,
    infer_resolution,
    jsonb,
    market_group_id,
    normalize_question,
    optional_bool,
    optional_float,
    parse_dt,
)


async def create_ingestion_run(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    data_start: datetime,
    data_end: datetime,
    min_duration_days: float,
    max_duration_days: float,
    target_tag_slugs: Sequence[str],
) -> None:
    await conn.execute(
        """
        INSERT INTO polymarket.ingestion_runs (
            run_id, data_start, data_end, min_duration_days, max_duration_days, target_tag_slugs
        )
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        run_id,
        data_start,
        data_end,
        min_duration_days,
        max_duration_days,
        list(target_tag_slugs),
    )


async def finish_ingestion_run(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    status: str,
    events_discovered: int,
    markets_seen: int,
    markets_ingested: int,
    probability_rows: int,
) -> None:
    await conn.execute(
        """
        UPDATE polymarket.ingestion_runs
        SET finished_at = NOW(),
            status = $2,
            events_discovered = $3,
            markets_seen = $4,
            markets_ingested = $5,
            probability_rows = $6
        WHERE run_id = $1
        """,
        run_id,
        status,
        events_discovered,
        markets_seen,
        markets_ingested,
        probability_rows,
    )


async def upsert_event(conn: asyncpg.Connection, record: EventRecord) -> None:
    event = record.event
    event_id = str(event.get("id") or event.get("event_id"))
    matched_tags = sorted(record.matched_tags)
    tags = event_tag_labels(event)
    created_at = parse_dt(event.get("createdAt"))
    end_at = parse_dt(event.get("endDate"))
    duration_days = None if created_at is None or end_at is None else (end_at - created_at).total_seconds() / 86400.0

    await conn.execute(
        """
        INSERT INTO polymarket.events (
            event_id, slug, title, description, category, tags, matched_target_tags,
            created_at, start_at, end_at, closed_at, duration_days,
            active, closed, archived, competitive,
            volume, volume_24hr, volume_1wk, volume_1mo, volume_1yr,
            open_interest, liquidity_amm, liquidity_clob, raw_event, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7,
            $8, $9, $10, $11, $12,
            $13, $14, $15, $16,
            $17, $18, $19, $20, $21,
            $22, $23, $24, $25::jsonb, NOW()
        )
        ON CONFLICT (event_id) DO UPDATE SET
            slug = EXCLUDED.slug,
            title = EXCLUDED.title,
            description = EXCLUDED.description,
            category = EXCLUDED.category,
            tags = EXCLUDED.tags,
            matched_target_tags = EXCLUDED.matched_target_tags,
            created_at = EXCLUDED.created_at,
            start_at = EXCLUDED.start_at,
            end_at = EXCLUDED.end_at,
            closed_at = EXCLUDED.closed_at,
            duration_days = EXCLUDED.duration_days,
            active = EXCLUDED.active,
            closed = EXCLUDED.closed,
            archived = EXCLUDED.archived,
            competitive = EXCLUDED.competitive,
            volume = EXCLUDED.volume,
            volume_24hr = EXCLUDED.volume_24hr,
            volume_1wk = EXCLUDED.volume_1wk,
            volume_1mo = EXCLUDED.volume_1mo,
            volume_1yr = EXCLUDED.volume_1yr,
            open_interest = EXCLUDED.open_interest,
            liquidity_amm = EXCLUDED.liquidity_amm,
            liquidity_clob = EXCLUDED.liquidity_clob,
            raw_event = EXCLUDED.raw_event,
            updated_at = NOW()
        """,
        event_id,
        event.get("slug"),
        event.get("title"),
        event.get("description"),
        event.get("category"),
        tags,
        matched_tags,
        created_at,
        parse_dt(event.get("startDate")),
        end_at,
        parse_dt(event.get("closedTime")),
        duration_days,
        optional_bool(event.get("active")),
        optional_bool(event.get("closed")),
        optional_bool(event.get("archived")),
        optional_float(event.get("competitive")),
        optional_float(event.get("volume")),
        optional_float(event.get("volume24hr")),
        optional_float(event.get("volume1wk")),
        optional_float(event.get("volume1mo")),
        optional_float(event.get("volume1yr")),
        optional_float(event.get("openInterest")),
        optional_float(event.get("liquidityAmm")),
        optional_float(event.get("liquidityClob")),
        jsonb(event),
    )
    await conn.executemany(
        """
        INSERT INTO polymarket.event_target_tags (event_id, tag_slug)
        VALUES ($1, $2)
        ON CONFLICT (event_id, tag_slug) DO NOTHING
        """,
        [(event_id, tag_slug) for tag_slug in matched_tags],
    )


def current_yes_percentage(outcomes: Sequence[str], outcome_prices: Sequence[float]) -> float | None:
    if len(outcomes) != len(outcome_prices):
        return None
    for outcome, price in zip(outcomes, outcome_prices):
        if outcome.lower() == "yes":
            return float(price) * 100.0
    return None


async def upsert_market_group(
    conn: asyncpg.Connection,
    group_id: str,
    question: str,
    normalized_question: str,
) -> None:
    await conn.execute(
        """
        INSERT INTO polymarket.market_groups (group_id, example_question, updated_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (group_id) DO UPDATE SET
            updated_at = NOW()
        """,
        group_id,
        question,
    )
    await conn.execute(
        """
        INSERT INTO polymarket.market_group_members (group_id, question, normalized_question)
        VALUES ($1, $2, $3)
        ON CONFLICT (group_id, normalized_question) DO UPDATE SET
            question = EXCLUDED.question
        """,
        group_id,
        question,
        normalized_question,
    )


async def upsert_market(conn: asyncpg.Connection, market: MarketRecord) -> None:
    detail = market.market_detail
    raw = market.raw_market
    question = market.question or ""
    group_id = market_group_id(question)
    normalized_question = normalize_question(question)
    closed = optional_bool(detail.get("closed") if detail.get("closed") is not None else raw.get("closed")) or False
    resolved_outcome, did_happen, resolution_status = infer_resolution(
        market.outcomes,
        market.outcome_prices,
        closed,
    )
    yes_percentage = current_yes_percentage(market.outcomes, market.outcome_prices)

    await upsert_market_group(conn, group_id, question, normalized_question)

    await conn.execute(
        """
        INSERT INTO polymarket.markets (
            market_id, event_id, group_id, condition_id, slug, question,
            group_item_title, group_item_threshold, market_type, format_type,
            outcomes, outcome_prices, yes_token_id, no_token_id,
            resolved_outcome, did_happen, resolution_status,
            active, closed, accepting_orders,
            market_created_at, start_at, end_at, closed_at,
            volume, volume_24hr, volume_1wk, liquidity_amm, liquidity_clob,
            best_bid, best_ask, spread, last_trade_price,
            yes_percentage, raw_market, raw_market_detail, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6,
            $7, $8, $9, $10,
            $11, $12, $13, $14,
            $15, $16, $17,
            $18, $19, $20,
            $21, $22, $23, $24,
            $25, $26, $27, $28, $29,
            $30, $31, $32, $33,
            $34, $35::jsonb, $36::jsonb, NOW()
        )
        ON CONFLICT (market_id) DO UPDATE SET
            event_id = EXCLUDED.event_id,
            group_id = EXCLUDED.group_id,
            condition_id = EXCLUDED.condition_id,
            slug = EXCLUDED.slug,
            question = EXCLUDED.question,
            group_item_title = EXCLUDED.group_item_title,
            group_item_threshold = EXCLUDED.group_item_threshold,
            market_type = EXCLUDED.market_type,
            format_type = EXCLUDED.format_type,
            outcomes = EXCLUDED.outcomes,
            outcome_prices = EXCLUDED.outcome_prices,
            yes_token_id = EXCLUDED.yes_token_id,
            no_token_id = EXCLUDED.no_token_id,
            resolved_outcome = EXCLUDED.resolved_outcome,
            did_happen = EXCLUDED.did_happen,
            resolution_status = EXCLUDED.resolution_status,
            active = EXCLUDED.active,
            closed = EXCLUDED.closed,
            accepting_orders = EXCLUDED.accepting_orders,
            market_created_at = EXCLUDED.market_created_at,
            start_at = EXCLUDED.start_at,
            end_at = EXCLUDED.end_at,
            closed_at = EXCLUDED.closed_at,
            volume = EXCLUDED.volume,
            volume_24hr = EXCLUDED.volume_24hr,
            volume_1wk = EXCLUDED.volume_1wk,
            liquidity_amm = EXCLUDED.liquidity_amm,
            liquidity_clob = EXCLUDED.liquidity_clob,
            best_bid = EXCLUDED.best_bid,
            best_ask = EXCLUDED.best_ask,
            spread = EXCLUDED.spread,
            last_trade_price = EXCLUDED.last_trade_price,
            yes_percentage = EXCLUDED.yes_percentage,
            raw_market = EXCLUDED.raw_market,
            raw_market_detail = EXCLUDED.raw_market_detail,
            updated_at = NOW()
        """,
        market.market_id,
        market.event_id,
        group_id,
        market.condition_id,
        detail.get("slug") or raw.get("slug"),
        question,
        detail.get("groupItemTitle") or raw.get("groupItemTitle"),
        detail.get("groupItemThreshold") or raw.get("groupItemThreshold"),
        detail.get("marketType") or raw.get("marketType"),
        detail.get("formatType") or raw.get("formatType"),
        market.outcomes,
        market.outcome_prices,
        market.yes_token_id,
        market.no_token_id,
        resolved_outcome,
        did_happen,
        resolution_status,
        optional_bool(detail.get("active") if detail.get("active") is not None else raw.get("active")),
        closed,
        optional_bool(detail.get("acceptingOrders")),
        parse_dt(detail.get("createdAt") or raw.get("createdAt")),
        parse_dt(detail.get("startDateIso") or raw.get("startDate")),
        parse_dt(detail.get("endDateIso") or raw.get("endDate")),
        parse_dt(detail.get("closedTime") or raw.get("closedTime")),
        optional_float(detail.get("volumeNum") or raw.get("volumeNum") or raw.get("volume")),
        optional_float(detail.get("volume24hr") or raw.get("volume24hr")),
        optional_float(detail.get("volume1wk") or raw.get("volume1wk")),
        optional_float(detail.get("liquidityAmm") or raw.get("liquidityAmm")),
        optional_float(detail.get("liquidityClob") or raw.get("liquidityClob")),
        optional_float(detail.get("bestBid") or raw.get("bestBid")),
        optional_float(detail.get("bestAsk") or raw.get("bestAsk")),
        optional_float(detail.get("spread") or raw.get("spread")),
        optional_float(detail.get("lastTradePrice") or raw.get("lastTradePrice")),
        yes_percentage,
        jsonb(raw),
        jsonb(detail),
    )


async def mark_market_state(
    conn: asyncpg.Connection,
    *,
    market: MarketRecord,
    run_id: UUID,
    status: str,
    row_count: int,
    last_error: str | None = None,
) -> None:
    await conn.execute(
        """
        INSERT INTO polymarket.market_ingestion_state (
            market_id, event_id, last_run_id, status, history_start, history_end, row_count, last_error, updated_at
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
        ON CONFLICT (market_id) DO UPDATE SET
            last_run_id = EXCLUDED.last_run_id,
            status = EXCLUDED.status,
            history_start = EXCLUDED.history_start,
            history_end = EXCLUDED.history_end,
            row_count = EXCLUDED.row_count,
            last_error = EXCLUDED.last_error,
            updated_at = NOW()
        """,
        market.market_id,
        market.event_id,
        run_id,
        status,
        market.history_start,
        market.history_end,
        row_count,
        last_error,
    )


async def insert_probability_rows(
    conn: asyncpg.Connection,
    *,
    market: MarketRecord,
    rows: Sequence[tuple[datetime, float]],
    source_fidelity_minutes: int,
    batch_size: int,
) -> int:
    inserted = 0
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        await conn.executemany(
            """
            INSERT INTO polymarket.market_probability_1m (
                market_id, event_id, yes_token_id, ts, available_at, yes_probability, source_fidelity_minutes
            )
            VALUES ($1, $2, $3, $4, $4, $5, $6)
            ON CONFLICT (market_id, ts) DO UPDATE SET
                yes_token_id = EXCLUDED.yes_token_id,
                event_id = EXCLUDED.event_id,
                available_at = EXCLUDED.available_at,
                yes_probability = EXCLUDED.yes_probability,
                source_fidelity_minutes = EXCLUDED.source_fidelity_minutes,
                ingested_at = NOW()
            """,
            [
                (
                    market.market_id,
                    market.event_id,
                    market.yes_token_id,
                    ts,
                    probability,
                    source_fidelity_minutes,
                )
                for ts, probability in batch
            ],
        )
        inserted += len(batch)
    return inserted


async def set_market_yes_percentage(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    yes_probability: float,
) -> None:
    await conn.execute(
        """
        UPDATE polymarket.markets
        SET yes_percentage = $2,
            updated_at = NOW()
        WHERE market_id = $1
        """,
        market_id,
        yes_probability * 100.0,
    )


async def mark_market_triggered_70(conn: asyncpg.Connection, market_id: str) -> bool:
    row = await conn.fetchrow(
        """
        UPDATE polymarket.markets
        SET triggered_70 = TRUE,
            updated_at = NOW()
        WHERE market_id = $1
          AND triggered_70 = FALSE
        RETURNING market_id
        """,
        market_id,
    )
    return row is not None
