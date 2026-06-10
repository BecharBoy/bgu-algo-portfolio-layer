from __future__ import annotations

from typing import Any
from uuid import UUID

import asyncpg

from database.backtesting.repositories._shared import SCHEMA, json_text


async def save_market_decision(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    input_hash: str,
    market_id: str,
    event_id: str,
    event_title: str,
    market_question: str,
    model_name: str,
    prompt_version: str,
    llm_input: dict[str, Any],
    llm_output: dict[str, Any],
    relevant: bool,
    reason: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_market_decisions (
            input_hash, market_id, event_id, event_title, market_question,
            model_name, prompt_version, llm_input, llm_output, relevant, reason
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8::JSONB,$9::JSONB,$10,$11)
        ON CONFLICT (input_hash) DO NOTHING
        """,
        input_hash,
        market_id,
        event_id,
        event_title,
        market_question,
        model_name,
        prompt_version,
        json_text(llm_input),
        json_text(llm_output),
        relevant,
        reason,
    )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_market_decisions (run_id, market_id, input_hash)
        VALUES ($1, $2, $3)
        ON CONFLICT (run_id, market_id) DO UPDATE SET input_hash = EXCLUDED.input_hash
        """,
        run_id,
        market_id,
        input_hash,
    )


async def reusable_market_decision(
    conn: asyncpg.Connection,
    input_hash: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_market_decisions WHERE input_hash = $1",
        input_hash,
    )


async def reusable_market_decision_for_market(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    model_name: str,
    prompt_version: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"""
        SELECT * FROM {SCHEMA}.historical_market_decisions
        WHERE market_id=$1 AND model_name=$2 AND prompt_version=$3
        ORDER BY processed_at DESC
        LIMIT 1
        """,
        market_id,
        model_name,
        prompt_version,
    )


async def link_run_market_decision(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market_id: str,
    input_hash: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_market_decisions (run_id, market_id, input_hash)
        VALUES ($1,$2,$3)
        ON CONFLICT (run_id, market_id) DO UPDATE SET input_hash=EXCLUDED.input_hash
        """,
        run_id,
        market_id,
        input_hash,
    )


async def accepted_market_ids(conn: asyncpg.Connection, run_id: UUID) -> list[str]:
    rows = await conn.fetch(
        f"""
        SELECT d.market_id
        FROM {SCHEMA}.historical_run_market_decisions r
        JOIN {SCHEMA}.historical_market_decisions d ON d.input_hash = r.input_hash
        WHERE r.run_id = $1 AND d.relevant
        ORDER BY d.market_id
        """,
        run_id,
    )
    return [row["market_id"] for row in rows]
