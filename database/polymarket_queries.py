from __future__ import annotations

from datetime import datetime
from uuid import UUID

import asyncpg


async def market_is_complete(conn: asyncpg.Connection, market_id: str) -> bool:
    status = await conn.fetchval(
        "SELECT status FROM polymarket.market_ingestion_state WHERE market_id = $1",
        market_id,
    )
    return status == "complete"


async def probability_bounds(conn: asyncpg.Connection, market_id: str) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT
            market_id,
            MIN(ts) AS first_ts,
            MAX(ts) AS last_ts,
            COUNT(*) AS row_count
        FROM polymarket.market_probability_1m
        WHERE market_id = $1
        GROUP BY market_id
        """,
        market_id,
    )


async def query_probability_window(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    start_at: datetime,
    end_at: datetime,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT market_id, event_id, yes_token_id, ts, available_at, yes_probability
        FROM polymarket.market_probability_1m
        WHERE market_id = $1
          AND ts >= $2
          AND ts <= $3
        ORDER BY ts ASC
        """,
        market_id,
        start_at,
        end_at,
    )


async def query_asof_probability(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    decision_time: datetime,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT market_id, event_id, yes_token_id, ts, available_at, yes_probability
        FROM polymarket.market_probability_1m
        WHERE market_id = $1
          AND available_at <= $2
        ORDER BY available_at DESC
        LIMIT 1
        """,
        market_id,
        decision_time,
    )


async def ingestion_run_summary(conn: asyncpg.Connection, run_id: UUID) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT *
        FROM polymarket.ingestion_runs
        WHERE run_id = $1
        """,
        run_id,
    )


async def unresolved_markets(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT market_id, event_id, question, resolution_status, did_happen
        FROM polymarket.markets
        WHERE did_happen IS NULL
        ORDER BY end_at ASC NULLS LAST
        """
    )


async def market_groups_by_count(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT
            g.group_id,
            g.example_question,
            COUNT(m.market_id) AS market_count,
            g.created_at,
            g.updated_at
        FROM polymarket.market_groups g
        LEFT JOIN polymarket.markets m ON m.group_id = g.group_id
        GROUP BY g.group_id, g.example_question, g.created_at, g.updated_at
        ORDER BY market_count DESC, g.group_id ASC
        """
    )


async def markets_in_group(conn: asyncpg.Connection, group_id: str) -> list[asyncpg.Record]:
    return await conn.fetch(
        """
        SELECT
            market_id,
            event_id,
            question,
            market_created_at,
            end_at,
            volume,
            yes_percentage,
            triggered_70
        FROM polymarket.markets
        WHERE group_id = $1
        ORDER BY market_created_at ASC NULLS LAST, market_id ASC
        """,
        group_id,
    )


async def latest_market_group_weights(
    conn: asyncpg.Connection,
    *,
    group_id: str,
    model_name: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        """
        SELECT group_id, model_name, training_sample_count, trained_at, created_at
        FROM polymarket.market_group_ml_weights
        WHERE group_id = $1
          AND model_name = $2
        ORDER BY trained_at DESC, created_at DESC
        LIMIT 1
        """,
        group_id,
        model_name,
    )
