from __future__ import annotations

from typing import Any
from uuid import UUID
import asyncpg

from database.backtesting.repositories._shared import SCHEMA, json_text


async def save_event_decision(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    input_hash: str,
    event_id: str,
    event_title: str,
    model_name: str,
    prompt_version: str,
    llm_input: dict[str, Any],
    llm_output: dict[str, Any],
    relevant: bool,
    reason: str,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_event_decisions (
            input_hash, event_id, event_title, model_name, prompt_version,
            llm_input, llm_output, relevant, reason
        )
        VALUES ($1,$2,$3,$4,$5,$6::JSONB,$7::JSONB,$8,$9)
        ON CONFLICT (input_hash) DO NOTHING
        """,
        input_hash,
        event_id,
        event_title,
        model_name,
        prompt_version,
        json_text(llm_input),
        json_text(llm_output),
        relevant,
        reason,
    )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_event_decisions (run_id, event_id, input_hash)
        VALUES ($1, $2, $3)
        ON CONFLICT (run_id, event_id) DO UPDATE SET input_hash = EXCLUDED.input_hash
        """,
        run_id,
        event_id,
        input_hash,
    )


async def reusable_event_decision(
    conn: asyncpg.Connection,
    input_hash: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_event_decisions WHERE input_hash = $1",
        input_hash,
    )


async def accepted_event_ids(conn: asyncpg.Connection, run_id: UUID) -> list[str]:
    rows = await conn.fetch(
        f"""
        SELECT d.event_id
        FROM {SCHEMA}.historical_run_event_decisions r
        JOIN {SCHEMA}.historical_event_decisions d ON d.input_hash = r.input_hash
        WHERE r.run_id = $1 AND d.relevant
        ORDER BY d.event_id
        """,
        run_id,
    )
    return [row["event_id"] for row in rows]
