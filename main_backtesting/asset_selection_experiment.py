from __future__ import annotations

from uuid import UUID, uuid4

from database.db_connection import connect
from database.backtesting.schema import initialize_historical_schema


async def run_asset_selection_experiment(
    *,
    source_run_id: UUID | None,
    experiment_id: UUID | None,
    query_limit: int,
    sample_seed: int,
) -> UUID:
    if experiment_id is None and source_run_id is None:
        raise ValueError("Either source_run_id or experiment_id is required")
    resolved_id = experiment_id or uuid4()
    conn = await connect()
    try:
        await initialize_historical_schema(conn)
        if source_run_id is not None:
            raise NotImplementedError(
                "Asset selection A/B experiments require a fully configured experiment runner; "
                "repository stubs are available but the experiment pipeline is not implemented."
            )
        if experiment_id is not None:
            raise NotImplementedError(
                "Resuming asset selection experiments is not implemented in this codebase snapshot."
            )
    finally:
        await conn.close()
    return resolved_id
