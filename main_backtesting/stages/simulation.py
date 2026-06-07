from __future__ import annotations

from datetime import datetime, timedelta
from statistics import median
from typing import Any
from database.backtesting.repositories import json_value
from database.backtesting.repositories.machine_learning import (
    prior_ml_observations,
    save_ml_prediction,
    save_model_snapshot,
)
from database.backtesting.repositories.prices import price_bars
from database.backtesting.repositories.probabilities import probability_history
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.trades import save_trade
from database.backtesting.repositories.worlds import run_world_assets
from database.backtesting.market_data import bars_before, bars_from, next_bar_after
from database.backtesting.polymarket import probability_as_of
from main_backtesting.models import Asset, MLObservation, PriceBar, ProbabilityPoint, SourceMarket, Trade
from main_backtesting.reporting import create_trade_graph
from strategies.event_driven_long import rate_of_change
from strategies.event_driven_ml import DAILY_SESSION_LENGTH, close_as_of, evaluate_prediction, predict, train_snapshot

from main_backtesting.stages.event_filter import accepted_markets, run_events


def polymarket_volume_ratio(
    probabilities: list[ProbabilityPoint],
    *,
    trigger_at: datetime,
    lookback_hours: int,
    minimum_history_hours: int,
) -> float | None:
    visible = [
        point
        for point in probabilities
        if point.timestamp <= trigger_at and point.volume_usdc is not None
    ]
    if not visible:
        return None
    current = float(visible[-1].volume_usdc or 0.0)
    history = [
        float(point.volume_usdc or 0.0)
        for point in visible[-lookback_hours - 1 : -1]
    ]
    if len(history) < minimum_history_hours:
        return None
    baseline = median(history)
    if baseline <= 0:
        return float("inf") if current > 0 else 0.0
    return current / baseline


async def simulate_one_trade(
    self,
    *,
    market: SourceMarket,
    asset: Asset,
    pass_number: int,
    trigger_at: datetime,
    portfolio: str,
    strategy_branch: str,
    direction: str,
    resolution: str,
    bars: list[PriceBar],
    probabilities: list[ProbabilityPoint],
    predicted_target_price: float | None = None,
    predicted_target_reached: bool | None = None,
    evaluation_event_end: datetime | None = None,
) -> tuple[Trade | None, dict[str, Any]]:
    decision: dict[str, Any] = {
        "portfolio": portfolio,
        "strategy_branch": strategy_branch,
        "direction": direction,
        "opened": False,
    }
    entry_bar = next_bar_after(bars, trigger_at)
    if (
        entry_bar is None
        or entry_bar.timestamp < self.config.start
        or entry_bar.timestamp >= self.config.end
    ):
        return None, {**decision, "reason": "no_next_price_bar_before_simulation_end"}
    latest_probability = probability_as_of(probabilities, entry_bar.timestamp)
    decision["entry_at"] = entry_bar.timestamp
    decision["probability_at_entry"] = latest_probability
    if latest_probability is None or latest_probability <= self.config.threshold:
        return None, {**decision, "reason": "probability_not_above_threshold_at_entry"}
    exit_deadline = (
        evaluation_event_end - timedelta(days=1)
        if strategy_branch == "machine_learning" and evaluation_event_end
        else None
    )
    if exit_deadline is not None and entry_bar.timestamp >= exit_deadline:
        return None, {**decision, "reason": "ml_entry_after_one_day_before_market_end"}
    trade = self.strategy.open_trade(
        run_id=self.run_id,
        market_id=market.market_id,
        event_id=market.event_id,
        question=market.question,
        symbol=asset.symbol,
        asset_name=asset.asset_name,
        pass_number=pass_number,
        trigger_at=trigger_at,
        entry_bar=entry_bar,
        previous_bars=bars_before(bars, entry_bar.timestamp),
        final_outcome=market.final_outcome,
        direction=direction,  # type: ignore[arg-type]
        portfolio=portfolio,
        strategy_branch=strategy_branch,
        resolution=resolution,  # type: ignore[arg-type]
        predicted_target_price=predicted_target_price,
    )
    if trade is None:
        return None, {
            **decision,
            "reason": "insufficient_trailing_stop_history_or_invalid_entry_price",
        }
    trade.predicted_target_reached = predicted_target_reached
    pending_momentum_exit = False
    for bar in bars_from(bars, entry_bar.timestamp):
        if bar.timestamp > entry_bar.timestamp and pending_momentum_exit:
            self.strategy.close_trade(
                trade,
                timestamp=bar.timestamp,
                price=bar.open,
                reason="momentum_reversal",
            )
            break
        if (
            exit_deadline is not None
            and bar.timestamp > entry_bar.timestamp
            and bar.timestamp >= exit_deadline
        ):
            self.strategy.close_trade(
                trade,
                timestamp=bar.timestamp,
                price=bar.open,
                reason="one_day_before_market_end",
            )
            break
        if self.strategy.update_trade(trade, bar, bars_before(bars, bar.timestamp)):
            break
        if predicted_target_price is not None:
            target_touched = (
                bar.high >= predicted_target_price
                if direction == "long"
                else bar.low <= predicted_target_price
            )
            if target_touched:
                self.strategy.close_trade(
                    trade,
                    timestamp=bar.timestamp,
                    price=predicted_target_price,
                    reason="ml_predicted_target",
                )
                break
        if strategy_branch == "momentum" and bar.timestamp >= entry_bar.timestamp:
            completed_bars = [
                item for item in bars if item.timestamp <= bar.timestamp
            ]
            momentum = rate_of_change(
                completed_bars,
                self.config.momentum_lookback_bars,
            )
            pending_momentum_exit = momentum is not None and momentum <= 0
    if trade.exit_at is None:
        mark_end = min(self.config.end, exit_deadline) if exit_deadline else self.config.end
        final_bars = [bar for bar in bars if bar.timestamp < mark_end]
        if final_bars:
            self.strategy.close_trade(
                trade,
                timestamp=final_bars[-1].timestamp,
                price=final_bars[-1].close,
                reason=(
                    "one_day_before_market_end"
                    if exit_deadline is not None
                    else "market_end"
                ),
            )
    trade.graph_path = str(
        create_trade_graph(
            trade,
            bars=bars,
            probabilities=probabilities,
            simulation_end=self.config.end,
            event_end=evaluation_event_end if strategy_branch == "machine_learning" else None,
            graph_dir=self.run_dir / "graphs" / "trades",
        )
    )
    return trade, {
        **decision,
        "opened": True,
        "reason": "opened",
        "trade_id": str(trade.trade_id),
    }


async def run(self, conn: Any) -> None:
    events = {event.event_id: event for event in await run_events(self, conn)}
    markets = {market.market_id: market for market in await accepted_markets(self, conn)}
    observations_rows = await conn.fetch(
        """
        SELECT * FROM checking_relevant_events.historical_ml_observations
        WHERE run_id = $1
        """,
        self.run_id,
    )
    observations = {
        (row["event_id"], row["symbol"]): MLObservation(
            row["observation_id"], row["run_id"], row["event_id"], row["market_id"],
            row["first_pass_number"], row["first_pass_at"], row["label_available_at"], row["symbol"],
            row["event_archetype"], row["resolution"], json_value(row["features"]), json_value(row["research_data"]),
            row["classification_target"], row["regression_target"],
            row["valid_for_training"], row["exclusion_reason"],
        )
        for row in observations_rows
    }
    for row in await run_world_assets(conn, self.run_id):
        work_key = f"{row['market_id']}:{row['pass_number']}:{row['symbol']}"
        self.current_work_key = work_key
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="simulation",
            work_key=work_key,
            payload={"market_id": row["market_id"], "pass_number": row["pass_number"], "symbol": row["symbol"]},
        ):
            continue
        market = markets[row["market_id"]]
        event = events[market.event_id]
        asset = Asset(row["symbol"], row["asset_name"], row["asset_class"], row["reason"])
        resolution = "1h" if row["as_of"] >= self.hourly_boundary else "1d"
        bar_start = (
            max(market.created_at, self.hourly_boundary)
            if resolution == "1h"
            else market.created_at
        )
        bars = await price_bars(
            conn,
            symbol=asset.symbol,
            resolution=resolution,
            start=bar_start,
            end=min(self.config.end, market.end_at),
        )
        probabilities = await probability_history(
            conn,
            market_id=market.market_id,
            start=market.created_at,
            end=min(self.config.end, market.end_at),
        )
        observation = observations[(market.event_id, asset.symbol)]
        prior = await prior_ml_observations(
            conn,
            run_id=self.run_id,
            symbol=asset.symbol,
            event_archetype=observation.event_archetype,
            before=row["as_of"],
        )
        snapshot = train_snapshot(
            run_id=self.run_id,
            symbol=asset.symbol,
            event_archetype=observation.event_archetype,
            training_cutoff=row["as_of"],
            observations=prior,
            minimum_prior_observations=self.config.minimum_ml_prior_observations,
            prediction_features=observation.features,
        )
        await save_model_snapshot(conn, snapshot)
        opened = 0
        entry_decisions: list[dict[str, Any]] = []
        if snapshot.status == "trained":
            entry_bar = next_bar_after(bars, row["as_of"])
            event_open_price = observation.research_data.get("event_open_price")
            pass_price = (
                close_as_of(bars, row["as_of"])
                if resolution == "1d"
                else next(
                    (
                        bar.close
                        for bar in reversed(bars)
                        if bar.timestamp < row["as_of"]
                    ),
                    None,
                )
            )
            prediction = (
                predict(
                    snapshot,
                    run_id=self.run_id,
                    market_id=market.market_id,
                    event_id=market.event_id,
                    pass_number=row["pass_number"],
                    symbol=asset.symbol,
                    features=observation.features,
                    event_open_price=float(event_open_price),
                    realized_price_at_pass=float(pass_price),
                )
                if entry_bar and event_open_price and pass_price
                else None
            )
            if prediction:
                if event.end_at <= self.config.historical_data_cutoff:
                    event_bars = await price_bars(
                        conn,
                        symbol=asset.symbol,
                        resolution="1d",
                        start=event.created_at,
                        end=event.end_at + timedelta(days=1),
                    )
                    event_bars = [
                        bar
                        for bar in event_bars
                        if event.created_at <= bar.timestamp
                        and bar.timestamp + DAILY_SESSION_LENGTH <= event.end_at
                    ]
                    evaluate_prediction(
                        prediction,
                        event_open_price=float(event_open_price),
                        bars_until_event_end=event_bars,
                    )
                await save_ml_prediction(conn, prediction)
                if prediction.directions_agree and prediction.remaining_gap > 0:
                    trade, decision = await simulate_one_trade(self, 
                        market=market,
                        asset=asset,
                        pass_number=row["pass_number"],
                        trigger_at=row["as_of"],
                        portfolio="machine_learning",
                        strategy_branch="machine_learning",
                        direction=prediction.direction,
                        resolution=resolution,
                        bars=bars,
                        probabilities=probabilities,
                        predicted_target_price=prediction.predicted_target_price,
                        predicted_target_reached=prediction.target_reached,
                        evaluation_event_end=market.end_at,
                    )
                    entry_decisions.append(decision)
                    if trade:
                        await save_trade(conn, trade)
                        opened += 1
                else:
                    entry_decisions.append(
                        {
                            "portfolio": "machine_learning",
                            "strategy_branch": "machine_learning",
                            "opened": False,
                            "reason": (
                                "classifier_and_ridge_directions_disagree"
                                if not prediction.directions_agree
                                else "remaining_predicted_gap_not_positive"
                            ),
                            "directions_agree": prediction.directions_agree,
                            "remaining_gap": prediction.remaining_gap,
                        }
                    )
            else:
                entry_decisions.append(
                    {
                        "portfolio": "machine_learning",
                        "strategy_branch": "machine_learning",
                        "opened": False,
                        "reason": "missing_prediction_input",
                        "has_entry_bar": entry_bar is not None,
                        "has_event_open_price": event_open_price is not None,
                        "has_pass_price": pass_price is not None,
                    }
                )
        elif snapshot.status in {"insufficient_history", "insufficient_class_diversity"}:
            entry_bar = next_bar_after(bars, row["as_of"])
            completed_before_entry = (
                bars_before(bars, entry_bar.timestamp) if entry_bar else []
            )
            momentum = rate_of_change(
                completed_before_entry,
                self.config.momentum_lookback_bars,
            )
            volume_ratio = polymarket_volume_ratio(
                probabilities,
                trigger_at=row["as_of"],
                lookback_hours=self.config.polymarket_volume_lookback_hours,
                minimum_history_hours=self.config.polymarket_volume_minimum_history_hours,
            )
            blocked_reason = None
            if momentum is None:
                blocked_reason = "insufficient_price_momentum_history"
            elif momentum <= 0:
                blocked_reason = "price_momentum_not_positive"
            elif (
                volume_ratio is not None
                and volume_ratio < self.config.polymarket_volume_confirmation_ratio
            ):
                blocked_reason = "polymarket_volume_not_confirmed"
            if blocked_reason:
                entry_decisions.append(
                    {
                        "portfolio": "polymarket_momentum",
                        "strategy_branch": "momentum",
                        "opened": False,
                        "reason": blocked_reason,
                        "price_momentum": momentum,
                        "polymarket_volume_ratio": volume_ratio,
                    }
                )
            else:
                trade, decision = await simulate_one_trade(
                    self,
                    market=market,
                    asset=asset,
                    pass_number=row["pass_number"],
                    trigger_at=row["as_of"],
                    portfolio="polymarket_momentum",
                    strategy_branch="momentum",
                    direction="long",
                    resolution=resolution,
                    bars=bars,
                    probabilities=probabilities,
                )
                decision.update(
                    {
                        "price_momentum": momentum,
                        "polymarket_volume_ratio": volume_ratio,
                        "volume_confirmation": (
                            "unavailable"
                            if volume_ratio is None
                            else "confirmed"
                        ),
                    }
                )
                entry_decisions.append(decision)
                if trade:
                    await save_trade(conn, trade)
                    opened += 1
        else:
            entry_decisions.append(
                {
                    "strategy_branch": "none",
                    "opened": False,
                    "reason": f"model_state_{snapshot.status}",
                }
            )
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="simulation",
            work_key=work_key,
            result={
                "model_status": snapshot.status,
                "trades_opened": opened,
                "entry_decisions": entry_decisions,
            },
        )
