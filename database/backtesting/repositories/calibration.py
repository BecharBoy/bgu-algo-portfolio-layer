from __future__ import annotations

from typing import Any
from uuid import UUID
import asyncpg

from database.backtesting.repositories._shared import SCHEMA, json_text


async def save_batch_calibration(
    conn: asyncpg.Connection,
    *,
    calibration_id: UUID,
    task: str,
    model_name: str,
    tested_sizes: list[dict[str, Any]],
    selected_batch_size: int,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_batch_calibrations
            (calibration_id, task, model_name, tested_sizes, selected_batch_size)
        VALUES ($1,$2,$3,$4::JSONB,$5)
        """,
        calibration_id,
        task,
        model_name,
        json_text(tested_sizes),
        selected_batch_size,
    )


async def latest_batch_sizes(conn: asyncpg.Connection, model_name: str) -> dict[str, int]:
    rows = await conn.fetch(
        f"""
        SELECT DISTINCT ON (task) task, selected_batch_size
        FROM {SCHEMA}.historical_batch_calibrations
        WHERE model_name = $1
        ORDER BY task, created_at DESC
        """,
        model_name,
    )
    return {row["task"]: row["selected_batch_size"] for row in rows}
