from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from database.backtesting.repositories.machine_learning import (
    prior_ml_observations,
    run_ml_observations,
    save_ml_observation,
    save_model_snapshot,
)
from database.backtesting.repositories.prices import asset_metadata, price_bars
from database.backtesting.repositories.probabilities import probability_history
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import run_resolved_world_assets
from database.backtesting.polymarket import probability_as_of
from main_backtesting.utils import event_archetype
from strategies.event_driven_ml import DAILY_SESSION_LENGTH, build_observation, return_between, train_snapshot

from main_backtesting.stages.event_filter import accepted_markets, run_events


async def run(self, conn: Any) -> None:
    events = {event.event_id: event for event in await run_events(self, conn)}
    markets = {market.market_id: market for market in await accepted_markets(self, conn)}
    world_rows = list(await run_resolved_world_assets(conn, self.run_id))
    first_by_event_asset: dict[tuple[str, str], Any] = {}
    for row in world_rows:
        key = (row["event_id"], row["symbol"])
        if key not in first_by_event_asset or row["as_of"] < first_by_event_asset[key]["as_of"]:
            first_by_event_asset[key] = row
    for (event_id, symbol), row in sorted(
        first_by_event_asset.items(), key=lambda item: item[1]["as_of"]
    ):
        self.current_work_key = f"{event_id}:{symbol}"
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="ml_observations",
            work_key=self.current_work_key,
            payload={"event_id": event_id, "symbol": symbol},
        ):
            continue
        market = markets[row["market_id"]]
        event = events[event_id]
        archetype = event_archetype(
            market.tags,
            question=market.question,
            symbol=symbol,
        )
        if archetype is None:
            await finish_work(
                conn,
                run_id=self.run_id,
                stage="ml_observations",
                work_key=self.current_work_key,
                result={
                    "valid_for_training": False,
                    "exclusion_reason": "no_defined_semantic_ml_archetype",
                },
            )
            continue
        metadata = await asset_metadata(conn, symbol)
        asset_benchmark = metadata["benchmark_symbol"] if metadata else None
        year_start = datetime(row["as_of"].year, 1, 1, tzinfo=timezone.utc)
        daily_start = min(
            year_start - timedelta(days=40),
            row["as_of"] - timedelta(days=40),
            event.created_at,
        )
        daily_end = min(
            event.end_at + timedelta(days=1),
            self.config.historical_data_cutoff + timedelta(days=1),
        )
        asset_daily = await price_bars(
            conn, symbol=symbol, resolution="1d", start=daily_start, end=daily_end
        )
        spy_daily = await price_bars(
            conn, symbol="SPY", resolution="1d", start=daily_start, end=daily_end
        )
        sector_daily = (
            await price_bars(
                conn,
                symbol=asset_benchmark,
                resolution="1d",
                start=daily_start,
                end=daily_end,
            )
            if asset_benchmark
            else []
        )
        probabilities = await probability_history(
            conn,
            market_id=market.market_id,
            start=market.created_at,
            end=min(self.config.end, market.end_at),
        )
        benchmark_returns: dict[str, float | None] = {}
        for benchmark_symbol in ["SPY", "QQQ", "IWM", "TLT", asset_benchmark]:
            if not benchmark_symbol or benchmark_symbol in benchmark_returns:
                continue
            benchmark_bars = await price_bars(
                conn,
                symbol=benchmark_symbol,
                resolution="1d",
                start=daily_start,
                end=row["as_of"] + timedelta(days=1),
            )
            benchmark_returns[benchmark_symbol] = return_between(
                benchmark_bars,
                row["as_of"] - timedelta(days=14),
                row["as_of"],
            )
        completed_asset_bars = [
            bar
            for bar in asset_daily
            if bar.timestamp + DAILY_SESSION_LENGTH <= row["as_of"]
        ]
        current_probability = probability_as_of(probabilities, row["as_of"])
        probability_one_hour_ago = probability_as_of(
            probabilities, row["as_of"] - timedelta(hours=1)
        )
        probability_twenty_four_hours_ago = probability_as_of(
            probabilities, row["as_of"] - timedelta(hours=24)
        )
        observation = build_observation(
            run_id=self.run_id,
            event_id=event_id,
            market_id=market.market_id,
            first_pass_number=row["pass_number"],
            first_pass_at=row["as_of"],
            event_created_at=event.created_at,
            event_end_at=event.end_at,
            label_data_cutoff=self.config.historical_data_cutoff,
            symbol=symbol,
            event_archetype=archetype,
            resolution="1h" if row["as_of"] >= self.hourly_boundary else "1d",
            asset_daily=asset_daily,
            sector_daily=sector_daily,
            spy_daily=spy_daily,
            research_data={
                "current_probability": current_probability,
                "recent_probability_changes": {
                    "one_hour": (
                        current_probability - probability_one_hour_ago
                        if current_probability is not None
                        and probability_one_hour_ago is not None
                        else None
                    ),
                    "twenty_four_hours": (
                        current_probability - probability_twenty_four_hours_ago
                        if current_probability is not None
                        and probability_twenty_four_hours_ago is not None
                        else None
                    ),
                },
                "previous_pass_count": row["pass_number"] - 1,
                "latest_completed_daily_volume": (
                    completed_asset_bars[-1].volume if completed_asset_bars else None
                ),
                "price_path_reference": {"symbol": symbol, "resolution": "1d"},
                "benchmark_references": ["SPY", "QQQ", "IWM", "TLT", asset_benchmark],
                "benchmark_two_week_returns": benchmark_returns,
            },
        )
        await save_ml_observation(conn, observation)
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="ml_observations",
            work_key=self.current_work_key,
            result={
                "valid_for_training": observation.valid_for_training,
                "exclusion_reason": observation.exclusion_reason,
            },
        )
    for observation in await run_ml_observations(conn, self.run_id):
        if observation.label_available_at > self.config.historical_data_cutoff:
            continue
        snapshot_cutoff = observation.label_available_at + timedelta(microseconds=1)
        self.current_work_key = f"completed-model-state:{observation.observation_id}"
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="ml_observations",
            work_key=self.current_work_key,
            payload={
                "observation_id": str(observation.observation_id),
                "label_available_at": observation.label_available_at,
            },
        ):
            continue
        completed = await prior_ml_observations(
            conn,
            run_id=self.run_id,
            symbol=observation.symbol,
            event_archetype=observation.event_archetype,
            before=snapshot_cutoff,
        )
        snapshot = train_snapshot(
            run_id=self.run_id,
            symbol=observation.symbol,
            event_archetype=observation.event_archetype,
            training_cutoff=snapshot_cutoff,
            observations=completed,
            minimum_prior_observations=self.config.minimum_ml_prior_observations,
        )
        snapshot.snapshot_id = await save_model_snapshot(conn, snapshot)
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="ml_observations",
            work_key=self.current_work_key,
            result={
                "model_status": snapshot.status,
                "training_sample_count": snapshot.training_sample_count,
            },
        )
