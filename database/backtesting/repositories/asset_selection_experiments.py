from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from database.backtesting.repositories._shared import SCHEMA


async def asset_selection_experiment(
    conn: asyncpg.Connection,
    experiment_id: UUID,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_asset_selection_experiments WHERE experiment_id = $1",
        experiment_id,
    )


async def completed_experiment_arms(
    conn: asyncpg.Connection,
    experiment_id: UUID,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT *
        FROM {SCHEMA}.historical_asset_selection_experiment_results
        WHERE experiment_id = $1 AND status = 'complete'
        ORDER BY query_index, arm
        """,
        experiment_id,
    )


async def create_asset_selection_experiment(
    conn: asyncpg.Connection,
    *,
    experiment_id: UUID,
    source_run_id: UUID,
    query_limit: int,
    sample_seed: int,
    model_name: str,
    catalog_hash: str,
    catalog_asset_count: int,
    output_dir: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_asset_selection_experiments
            (experiment_id, source_run_id, status, query_limit, sample_seed, model_name,
             catalog_hash, catalog_asset_count, output_dir)
        VALUES ($1, $2, 'running', $3, $4, $5, $6, $7, $8)
        """,
        experiment_id,
        source_run_id,
        query_limit,
        sample_seed,
        model_name,
        catalog_hash,
        catalog_asset_count,
        output_dir,
    )


async def experiment_queries(
    conn: asyncpg.Connection,
    experiment_id: UUID,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT *
        FROM {SCHEMA}.historical_asset_selection_experiment_queries
        WHERE experiment_id = $1
        ORDER BY query_index
        """,
        experiment_id,
    )


async def save_experiment_queries(
    conn: asyncpg.Connection,
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    await conn.executemany(
        f"""
        INSERT INTO {SCHEMA}.historical_asset_selection_experiment_queries
            (experiment_id, query_index, market_id, event_id, pass_number, as_of,
             event_title, question, tags, market_created_at, market_end_at, final_outcome)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::JSONB, $10, $11, $12)
        ON CONFLICT (experiment_id, query_index) DO NOTHING
        """,
        rows,
    )


async def save_experiment_result(
    conn: asyncpg.Connection,
    *,
    experiment_id: UUID,
    query_index: int,
    arm: str,
    status: str,
    duration_seconds: float,
    candidate_count: int | None = None,
    universe_name: str | None = None,
    universe_reason: str | None = None,
    method_input: dict | None = None,
    method_output: dict | None = None,
    error: str | None = None,
) -> None:
    from database.backtesting.repositories._shared import json_text

    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_asset_selection_experiment_results
            (experiment_id, query_index, arm, status, duration_seconds, candidate_count,
             universe_name, universe_reason, method_input, method_output, error)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9::JSONB, $10::JSONB, $11)
        ON CONFLICT (experiment_id, query_index, arm) DO UPDATE SET
            status = EXCLUDED.status,
            duration_seconds = EXCLUDED.duration_seconds,
            candidate_count = EXCLUDED.candidate_count,
            universe_name = EXCLUDED.universe_name,
            universe_reason = EXCLUDED.universe_reason,
            method_input = EXCLUDED.method_input,
            method_output = EXCLUDED.method_output,
            error = EXCLUDED.error
        """,
        experiment_id,
        query_index,
        arm,
        status,
        duration_seconds,
        candidate_count,
        universe_name,
        universe_reason,
        json_text(method_input or {}),
        json_text(method_output or {}),
        error,
    )


async def source_run_experiment_queries(
    conn: asyncpg.Connection,
    source_run_id: UUID,
    *,
    limit: int,
    seed: int,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT market_id, event_id, pass_number, above_at AS as_of
        FROM {SCHEMA}.historical_run_market_passes
        WHERE run_id = $1
        ORDER BY above_at
        LIMIT $2
        """,
        source_run_id,
        limit,
    )


async def update_asset_selection_experiment(
    conn: asyncpg.Connection,
    experiment_id: UUID,
    *,
    status: str,
    error: str | None = None,
) -> None:
    await conn.execute(
        f"""
        UPDATE {SCHEMA}.historical_asset_selection_experiments
        SET status = $2,
            error = $3,
            finished_at = CASE WHEN $2 IN ('complete', 'failed') THEN NOW() ELSE finished_at END
        WHERE experiment_id = $1
        """,
        experiment_id,
        status,
        error,
    )
