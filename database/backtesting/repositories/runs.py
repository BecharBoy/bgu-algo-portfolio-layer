from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID
import asyncpg

from database.backtesting.repositories._shared import SCHEMA, json_text


async def create_historical_run(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    config: dict[str, Any],
    hourly_boundary: datetime,
    output_dir: Path,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_backtest_runs
            (run_id, status, current_stage, config, hourly_boundary, output_dir)
        VALUES ($1, 'running', 'event_filter', $2::JSONB, $3, $4)
        """,
        run_id,
        json_text(config),
        hourly_boundary,
        str(output_dir),
    )


async def historical_run(conn: asyncpg.Connection, run_id: UUID) -> asyncpg.Record:
    row = await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_backtest_runs WHERE run_id = $1",
        run_id,
    )
    if row is None:
        raise ValueError(f"Historical backtest run does not exist: {run_id}")
    return row


async def update_run(
    conn: asyncpg.Connection,
    run_id: UUID,
    *,
    status: str,
    stage: str | None = None,
    error: str | None = None,
) -> None:
    await conn.execute(
        f"""
        UPDATE {SCHEMA}.historical_backtest_runs
        SET status = $2,
            current_stage = COALESCE($3, current_stage),
            error = $4,
            finished_at = CASE WHEN $2 IN ('complete', 'failed') THEN NOW() ELSE NULL END
        WHERE run_id = $1
        """,
        run_id,
        status,
        stage,
        error,
    )


async def start_work(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    stage: str,
    work_key: str,
    payload: dict[str, Any],
) -> bool:
    row = await conn.fetchrow(
        f"""
        INSERT INTO {SCHEMA}.historical_backtest_stage_work
            (run_id, stage, work_key, status, attempts, payload, started_at)
        VALUES ($1, $2, $3, 'running', 1, $4::JSONB, NOW())
        ON CONFLICT (run_id, stage, work_key) DO UPDATE SET
            status = 'running',
            attempts = {SCHEMA}.historical_backtest_stage_work.attempts + 1,
            payload = EXCLUDED.payload,
            error = NULL,
            started_at = NOW(),
            finished_at = NULL
        WHERE {SCHEMA}.historical_backtest_stage_work.status <> 'complete'
        RETURNING work_key
        """,
        run_id,
        stage,
        work_key,
        json_text(payload),
    )
    return row is not None


async def finish_work(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    stage: str,
    work_key: str,
    result: dict[str, Any] | None = None,
) -> None:
    await conn.execute(
        f"""
        UPDATE {SCHEMA}.historical_backtest_stage_work
        SET status = 'complete', result = $4::JSONB, error = NULL, finished_at = NOW()
        WHERE run_id = $1 AND stage = $2 AND work_key = $3
        """,
        run_id,
        stage,
        work_key,
        json_text(result or {}),
    )


async def record_stage_failure(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    stage: str,
    work_key: str | None,
    error: BaseException,
) -> None:
    if work_key is not None:
        await conn.execute(
            f"""
            UPDATE {SCHEMA}.historical_backtest_stage_work
            SET status = 'failed', error = $4, finished_at = NOW()
            WHERE run_id = $1 AND stage = $2 AND work_key = $3
            """,
            run_id,
            stage,
            work_key,
            str(error)[:10_000],
        )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_failures
            (run_id, stage, work_key, error_type, error)
        VALUES ($1, $2, $3, $4, $5)
        """,
        run_id,
        stage,
        work_key,
        type(error).__name__,
        str(error)[:10_000],
    )
    await update_run(conn, run_id, status="failed", stage=stage, error=str(error))


async def purge_run(conn: asyncpg.Connection, run_id: UUID) -> str | None:
    output_dir = await conn.fetchval(
        f"SELECT output_dir FROM {SCHEMA}.historical_backtest_runs WHERE run_id = $1",
        run_id,
    )
    if output_dir is None:
        return None
    run_specific_tables = [
        "historical_ml_predictions",
        "historical_trades",
        "historical_run_market_passes",
        "historical_run_markets",
        "historical_run_worlds",
        "historical_run_sentiments",
        "historical_run_market_decisions",
        "historical_run_event_decisions",
        "historical_run_failures",
        "historical_backtest_stage_work",
    ]
    async with conn.transaction():
        for table in run_specific_tables:
            await conn.execute(
                f"DELETE FROM {SCHEMA}.{table} WHERE run_id = $1",
                run_id,
            )
        await conn.execute(
            f"""
            UPDATE {SCHEMA}.historical_backtest_runs
            SET status='purged', current_stage='purged', error=NULL, finished_at=NOW()
            WHERE run_id=$1
            """,
            run_id,
        )
    return output_dir
