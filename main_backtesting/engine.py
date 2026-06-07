from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID, uuid4

from database.backtesting.polymarket import PolymarketHistoryClient
from database.backtesting.market_data import YFinanceClient
from database.backtesting.repositories import json_value
from database.backtesting.repositories.calibration import latest_batch_sizes
from database.backtesting.repositories.runs import (
    create_historical_run,
    historical_run,
    purge_run,
    record_stage_failure,
    update_run,
)
from database.backtesting.schema import initialize_historical_schema
from database.db_connection import connect
from LLM.ollama_client import OllamaClient
from main_backtesting.config import BacktestConfig, hourly_availability_boundary
from main_backtesting.stages import STAGES, STAGE_FUNCTIONS
from main_backtesting.stages.probabilities import detect_passes
from main_backtesting.stages.reports import validate_pipeline_integrity
from strategies.event_driven_long import EventDrivenStrategy

class HistoricalBacktestEngine:
    def __init__(
        self,
        config: BacktestConfig,
        *,
        run_id: UUID | None = None,
        stop_after_stage: str | None = None,
    ) -> None:
        self.config = config
        self.run_id = run_id or uuid4()
        if stop_after_stage is not None and stop_after_stage not in STAGES:
            raise ValueError(f"Unknown stop stage: {stop_after_stage}")
        self.stop_after_stage = stop_after_stage
        self.hourly_boundary = hourly_availability_boundary()
        self.run_dir = self.config.run_dir(self.run_id)
        self.current_work_key: str | None = None
        self.ollama = OllamaClient()
        self.polymarket = PolymarketHistoryClient(chunk_days=config.probability_chunk_days)
        self.prices = YFinanceClient(concurrency=config.price_download_concurrency)
        self.strategy = EventDrivenStrategy(
            trade_notional=config.trade_notional,
            range_period=config.trailing_range_bars,
            range_multiplier=config.trailing_range_multiplier,
        )


    async def close(self) -> None:
        await asyncio.gather(self.ollama.close(), self.polymarket.close())


    async def _run_stage(
        self,
        conn: Any,
        stage: str,
        function: Callable[[Any, Any], Awaitable[None]],
    ) -> None:
        self.current_work_key = None
        await update_run(conn, self.run_id, status="running", stage=stage, error=None)
        try:
            await function(self, conn)
        except Exception as error:
            await record_stage_failure(
                conn,
                run_id=self.run_id,
                stage=stage,
                work_key=self.current_work_key,
                error=error,
            )
            print(
                f"[failed] run_id={self.run_id} stage={stage} "
                f"work_key={self.current_work_key}"
            )
            raise


    async def run(self, *, resume: bool = False) -> UUID:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "graphs" / "markets").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "graphs" / "trades").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        conn = await connect()
        try:
            await initialize_historical_schema(conn)
            if resume:
                row = await historical_run(conn, self.run_id)
                self.config = BacktestConfig.from_json(json_value(row["config"]))
                self.hourly_boundary = row["hourly_boundary"]
                self.run_dir = Path(row["output_dir"])
            else:
                calibrated = await latest_batch_sizes(conn, self.ollama.model_name)
                self.config = replace(
                    self.config,
                    event_filter_batch_size=calibrated.get(
                        "event_filter", self.config.event_filter_batch_size
                    ),
                    asset_world_batch_size=calibrated.get(
                        "asset_world", self.config.asset_world_batch_size
                    ),
                )
                await create_historical_run(
                    conn,
                    run_id=self.run_id,
                    config=self.config.to_json(),
                    hourly_boundary=self.hourly_boundary,
                    output_dir=self.run_dir,
                )
            for stage, function in zip(STAGES, STAGE_FUNCTIONS):
                await self._run_stage(conn, stage, function)
                if stage == self.stop_after_stage:
                    await update_run(conn, self.run_id, status="paused", stage=stage)
                    return self.run_id
            await update_run(conn, self.run_id, status="complete", stage="complete")
            return self.run_id
        finally:
            await conn.close()
            await self.close()


BacktestEngine = HistoricalBacktestEngine


async def purge_historical_run(run_id: UUID) -> None:
    conn = await connect()
    try:
        await initialize_historical_schema(conn)
        output_dir = await purge_run(conn, run_id)
    finally:
        await conn.close()
    if output_dir:
        path = Path(output_dir).resolve()
        root = (Path(__file__).resolve().parent / "output" / "runs").resolve()
        if path.parent != root:
            raise RuntimeError(f"Refusing to purge unexpected run directory: {path}")
        if path.exists():
            shutil.rmtree(path)
