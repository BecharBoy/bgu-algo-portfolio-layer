from __future__ import annotations

from datetime import datetime
from uuid import UUID
import asyncpg
from main_backtesting.models import ProbabilityPoint, SourceMarket, ThresholdPass

from database.backtesting.repositories._shared import SCHEMA


async def save_probability_history(
    conn: asyncpg.Connection,
    *,
    market: SourceMarket,
    requested_start: datetime,
    requested_end: datetime,
    points: list[ProbabilityPoint],
) -> None:
    if points:
        await conn.executemany(
            f"""
            INSERT INTO {SCHEMA}.historical_probability_points (
                market_id, yes_token_id, hour_ts, source_ts, available_at,
                probability, volume_usdc
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7)
            ON CONFLICT (market_id, hour_ts) DO UPDATE SET
                source_ts = EXCLUDED.source_ts,
                available_at = EXCLUDED.available_at,
                probability = EXCLUDED.probability,
                volume_usdc = EXCLUDED.volume_usdc,
                downloaded_at = NOW()
            """,
            [
                (
                    market.market_id,
                    market.yes_token_id,
                    point.timestamp,
                    point.source_timestamp or point.timestamp,
                    point.available_at or point.source_timestamp or point.timestamp,
                    point.probability,
                    point.volume_usdc,
                )
                for point in points
            ],
        )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_probability_coverage (
            market_id, yes_token_id, requested_start, requested_end,
            first_hour, last_hour, row_count
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (market_id) DO UPDATE SET
            requested_start = LEAST({SCHEMA}.historical_probability_coverage.requested_start, EXCLUDED.requested_start),
            requested_end = GREATEST({SCHEMA}.historical_probability_coverage.requested_end, EXCLUDED.requested_end),
            first_hour = LEAST({SCHEMA}.historical_probability_coverage.first_hour, EXCLUDED.first_hour),
            last_hour = GREATEST({SCHEMA}.historical_probability_coverage.last_hour, EXCLUDED.last_hour),
            row_count = EXCLUDED.row_count,
            completed_at = NOW()
        """,
        market.market_id,
        market.yes_token_id,
        requested_start,
        requested_end,
        points[0].timestamp if points else None,
        points[-1].timestamp if points else None,
        len(points),
    )


async def probability_history(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    start: datetime,
    end: datetime,
) -> list[ProbabilityPoint]:
    rows = await conn.fetch(
        f"""
        SELECT hour_ts, probability, source_ts, available_at, volume_usdc
        FROM {SCHEMA}.historical_probability_points
        WHERE market_id = $1 AND hour_ts >= $2 AND hour_ts < $3
        ORDER BY hour_ts
        """,
        market_id,
        start,
        end,
    )
    return [
        ProbabilityPoint(
            timestamp=row["hour_ts"],
            probability=row["probability"],
            source_timestamp=row["source_ts"],
            available_at=row["available_at"],
            volume_usdc=row["volume_usdc"],
        )
        for row in rows
    ]


async def probability_is_covered(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    start: datetime,
    end: datetime,
) -> bool:
    return bool(
        await conn.fetchval(
            f"""
            SELECT requested_start <= $2 AND requested_end >= $3
            FROM {SCHEMA}.historical_probability_coverage
            WHERE market_id = $1
            """,
            market_id,
            start,
            end,
        )
    )


async def save_run_passes(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market: SourceMarket,
    passes: list[ThresholdPass],
) -> None:
    await conn.execute(
        f"DELETE FROM {SCHEMA}.historical_run_market_passes WHERE run_id = $1 AND market_id = $2",
        run_id,
        market.market_id,
    )
    if passes:
        await conn.executemany(
            f"""
            INSERT INTO {SCHEMA}.historical_run_market_passes (
                run_id, market_id, event_id, question, pass_number, above_at,
                above_probability, fell_below_at, fell_below_probability, final_outcome
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


async def save_run_market(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market: SourceMarket,
    probability_hour_count: int,
    probability_graph_path: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_markets (
            run_id, market_id, event_id, question, created_at, end_at,
            final_outcome, probability_hour_count, probability_graph_path
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
        ON CONFLICT (run_id, market_id) DO UPDATE SET
            final_outcome=EXCLUDED.final_outcome,
            probability_hour_count=EXCLUDED.probability_hour_count,
            probability_graph_path=EXCLUDED.probability_graph_path
        """,
        run_id,
        market.market_id,
        market.event_id,
        market.question,
        market.created_at,
        market.end_at,
        market.final_outcome,
        probability_hour_count,
        probability_graph_path,
    )


async def run_passes(conn: asyncpg.Connection, run_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT * FROM {SCHEMA}.historical_run_market_passes
        WHERE run_id = $1 ORDER BY above_at, market_id, pass_number
        """,
        run_id,
    )
