from __future__ import annotations

from typing import Any
from database.backtesting.repositories.runs import finish_work, start_work
from main_backtesting.reporting import generate_run_reports


def validate_pipeline_integrity(
    *,
    pass_count: int,
    asset_candidate_count: int,
    completed_simulation_count: int,
) -> None:
    if pass_count and not asset_candidate_count:
        raise RuntimeError(
            "Pipeline integrity failure: probability passes exist but no asset candidates exist"
        )
    if completed_simulation_count != asset_candidate_count:
        raise RuntimeError(
            "Pipeline integrity failure: not every asset candidate reached simulation"
        )


async def run(self, conn: Any) -> None:
    funnel = await conn.fetchrow(
        """
        SELECT
            (SELECT COUNT(*) FROM checking_relevant_events.historical_run_market_passes
             WHERE run_id=$1) AS pass_count,
            (SELECT COUNT(*) FROM (
             SELECT DISTINCT rw.market_id, rw.pass_number, r.resolved_symbol
             FROM checking_relevant_events.historical_run_worlds rw
             JOIN checking_relevant_events.historical_asset_world_assets a
               ON a.world_id=rw.world_id
             JOIN checking_relevant_events.historical_run_asset_resolutions r
               ON r.run_id=rw.run_id AND r.original_symbol=a.symbol
             WHERE rw.run_id=$1 AND r.resolved_symbol IS NOT NULL
            ) resolved_assets) AS asset_candidate_count,
            (SELECT COUNT(*) FROM checking_relevant_events.historical_backtest_stage_work
             WHERE run_id=$1 AND stage='simulation' AND status='complete')
                AS completed_simulation_count
        """,
        self.run_id,
    )
    validate_pipeline_integrity(**dict(funnel))
    self.current_work_key = "run-reports"
    if not await start_work(
        conn,
        run_id=self.run_id,
        stage="reports",
        work_key=self.current_work_key,
        payload={},
    ):
        return
    await generate_run_reports(conn, run_id=self.run_id, run_dir=self.run_dir)
    await finish_work(
        conn,
        run_id=self.run_id,
        stage="reports",
        work_key=self.current_work_key,
        result={},
    )
