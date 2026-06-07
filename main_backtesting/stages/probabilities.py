from __future__ import annotations

from typing import Any
from database.backtesting.repositories.probabilities import (
    probability_history,
    probability_is_covered,
    save_probability_history,
    save_run_market,
    save_run_passes,
)
from database.backtesting.repositories.runs import finish_work, start_work
from main_backtesting.models import ProbabilityPoint, ThresholdPass
from main_backtesting.reporting import create_probability_graph
from strategies.event_driven_long import ThresholdPassTracker

from main_backtesting.stages.event_filter import accepted_markets


def detect_passes(
    market_id: str,
    probabilities: list[ProbabilityPoint],
    threshold: float,
) -> list[ThresholdPass]:
    tracker = ThresholdPassTracker(market_id, threshold)
    for point in probabilities:
        tracker.observe(point.timestamp, point.probability)
    return tracker.passes


async def run(self, conn: Any) -> None:
    for market in await accepted_markets(self, conn):
        self.current_work_key = market.market_id
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="probabilities",
            work_key=market.market_id,
            payload={"market_id": market.market_id},
        ):
            continue
        start = market.created_at
        end = min(self.config.end, market.end_at)
        if await probability_is_covered(conn, market_id=market.market_id, start=start, end=end):
            points = await probability_history(
                conn, market_id=market.market_id, start=start, end=end
            )
        else:
            points = await self.polymarket.hourly_probabilities(market, start=start, end=end)
            await save_probability_history(
                conn,
                market=market,
                requested_start=start,
                requested_end=end,
                points=points,
            )
        passes = detect_passes(market.market_id, points, self.config.threshold)
        await save_run_passes(conn, run_id=self.run_id, market=market, passes=passes)
        graph_path = create_probability_graph(
            market_id=market.market_id,
            question=market.question,
            probabilities=points,
            passes=passes,
            event_end=market.end_at,
            final_outcome=market.final_outcome,
            graph_dir=self.run_dir / "graphs" / "markets",
        )
        await save_run_market(
            conn,
            run_id=self.run_id,
            market=market,
            probability_hour_count=len(points),
            probability_graph_path=str(graph_path),
        )
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="probabilities",
            work_key=market.market_id,
            result={
                "hour_count": len(points),
                "pass_count": len(passes),
                "graph_path": str(graph_path),
            },
        )
        print(f"[probability] market={market.market_id} hours={len(points)} passes={len(passes)}")
