from __future__ import annotations

from datetime import datetime, timedelta
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
from database.backtesting.repositories.trades import (
    save_momentum_parameter_results,
    save_trade,
    select_walk_forward_momentum_parameters,
)
from database.backtesting.repositories.worlds import run_resolved_world_assets
from database.backtesting.market_data import bars_before, bars_from, next_bar_after
from database.backtesting.polymarket import probability_as_of
from main_backtesting.models import Asset, MLObservation, PriceBar, ProbabilityPoint, SourceMarket, Trade
from main_backtesting.reporting import create_trade_graph
from strategies.event_driven_long import EventDrivenStrategy, rate_of_change
from strategies.event_driven_ml import DAILY_SESSION_LENGTH, evaluate_prediction, predict, train_snapshot

from main_backtesting.stages.event_filter import accepted_markets, run_events
from main_backtesting.stages.simulation_portfolio import (
    assemble_candidate,
    load_pass1_candidates_from_work,
    run_portfolio_pass2,
)
from portfolio.serialization import candidate_to_dict


def polymarket_volume_quality(
    probabilities: list[ProbabilityPoint],
    *,
    trigger_at: datetime,
    minimum_pre_entry_usdc: float,
    concentration_minimum_usdc: float,
    max_single_hour_share: float,
    gate_applied: bool = False,
) -> dict[str, Any]:
    completed_volumes = [
        max(float(point.volume_usdc or 0.0), 0.0)
        for point in probabilities
        if point.timestamp <= trigger_at and point.volume_usdc is not None
    ]
    if not completed_volumes:
        return {
            "gate_applied": False,
            "allowed": False,
            "reason": "polymarket_volume_unavailable",
            "pre_entry_volume_usdc": None,
            "active_volume_hours": 0,
            "largest_hour_volume_usdc": None,
            "largest_hour_share": None,
            "median_active_hour_volume_usdc": None,
            "latest_completed_hour_volume_usdc": None,
        }

    total = sum(completed_volumes)
    active_volumes = sorted(volume for volume in completed_volumes if volume > 0)
    active_hours = sum(volume > 0 for volume in completed_volumes)
    largest_hour = max(completed_volumes)
    largest_hour_share = largest_hour / total if total > 0 else None
    median_active = None
    if active_volumes:
        midpoint = len(active_volumes) // 2
        median_active = (
            active_volumes[midpoint]
            if len(active_volumes) % 2
            else (active_volumes[midpoint - 1] + active_volumes[midpoint]) / 2
        )
    if total < minimum_pre_entry_usdc:
        reason = "polymarket_volume_not_significant"
        allowed = False
    elif (
        largest_hour >= concentration_minimum_usdc
        and largest_hour_share is not None
        and largest_hour_share > max_single_hour_share
    ):
        reason = "polymarket_volume_single_hour_concentration"
        allowed = False
    else:
        reason = "polymarket_volume_quality_confirmed"
        allowed = True
    return {
        "gate_applied": gate_applied,
        "allowed": allowed,
        "reason": reason,
        "pre_entry_volume_usdc": total,
        "active_volume_hours": active_hours,
        "largest_hour_volume_usdc": largest_hour,
        "largest_hour_share": largest_hour_share,
        "median_active_hour_volume_usdc": median_active,
        "latest_completed_hour_volume_usdc": completed_volumes[-1],
    }


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
    strategy: EventDrivenStrategy | None = None,
    momentum_lookback: int | None = None,
    graph_bars: list[PriceBar] | None = None,
    predicted_target_price: float | None = None,
    predicted_target_reached: bool | None = None,
    evaluation_event_end: datetime | None = None,
    create_graph: bool = True,
) -> tuple[Trade | None, dict[str, Any]]:
    active_strategy = strategy or self.strategy
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
    trade = active_strategy.open_trade(
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
    predicted_target_locked = False
    for bar in bars_from(bars, entry_bar.timestamp):
        if bar.timestamp > entry_bar.timestamp and pending_momentum_exit:
            active_strategy.close_trade(
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
            active_strategy.close_trade(
                trade,
                timestamp=bar.timestamp,
                price=bar.open,
                reason="one_day_before_market_end",
            )
            break
        if active_strategy.update_trade(trade, bar, bars_before(bars, bar.timestamp)):
            if predicted_target_locked and trade.exit_reason == "trailing_stop":
                trade.exit_reason = "ml_predicted_target_lock"
            break
        if predicted_target_price is not None and not predicted_target_locked:
            target_touched = (
                bar.high >= predicted_target_price
                if direction == "long"
                else bar.low <= predicted_target_price
            )
            if target_touched:
                predicted_target_locked = True
                trade.current_stop = (
                    max(trade.current_stop, predicted_target_price)
                    if direction == "long"
                    else min(trade.current_stop, predicted_target_price)
                )
                trade.stop_history.append(
                    {
                        "timestamp": bar.timestamp,
                        "stop": trade.current_stop,
                        "reason": "ml_predicted_target_locked",
                    }
                )
        if strategy_branch == "momentum" and bar.timestamp >= entry_bar.timestamp:
            completed_bars = [
                item for item in bars if item.timestamp <= bar.timestamp
            ]
            momentum = rate_of_change(
                completed_bars,
                momentum_lookback or self.config.momentum_lookback_bars,
            )
            pending_momentum_exit = momentum is not None and momentum <= 0
    if trade.exit_at is None:
        mark_end = min(self.config.end, exit_deadline) if exit_deadline else self.config.end
        final_bars = [bar for bar in bars if bar.timestamp < mark_end]
        if final_bars:
            active_strategy.close_trade(
                trade,
                timestamp=final_bars[-1].timestamp,
                price=final_bars[-1].close,
                reason=(
                    "one_day_before_market_end"
                    if exit_deadline is not None
                    else "market_end"
                ),
            )
    if create_graph:
        trade.graph_path = str(
            create_trade_graph(
                trade,
                bars=graph_bars or bars,
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
    for row in await run_resolved_world_assets(conn, self.run_id):
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
        warmup = timedelta(days=14 if resolution == "1h" else 45)
        bar_start = row["as_of"] - warmup
        if resolution == "1h":
            bar_start = max(bar_start, self.hourly_boundary)
        graph_end = market.end_at + timedelta(days=4)
        available_bars = await price_bars(
            conn,
            symbol=asset.symbol,
            resolution=resolution,
            start=bar_start,
            end=graph_end,
        )
        trade_end = min(self.config.end, market.end_at)
        bars = [bar for bar in available_bars if bar.timestamp < trade_end]
        entry_bar = next_bar_after(bars, row["as_of"])
        minimum_pre_entry_bars = max(self.config.momentum_lookback_grid) + 1
        if (
            resolution == "1h"
            and (
                entry_bar is None
                or len(bars_before(bars, entry_bar.timestamp)) < minimum_pre_entry_bars
            )
        ):
            daily_bars = await price_bars(
                conn,
                symbol=asset.symbol,
                resolution="1d",
                start=row["as_of"] - timedelta(days=45),
                end=graph_end,
            )
            daily_entry = next_bar_after(daily_bars, row["as_of"])
            if (
                daily_entry is not None
                and len(bars_before(daily_bars, daily_entry.timestamp))
                >= minimum_pre_entry_bars
            ):
                resolution = "1d"
                available_bars = daily_bars
                bars = [bar for bar in available_bars if bar.timestamp < trade_end]
        probabilities = await probability_history(
            conn,
            market_id=market.market_id,
            start=market.created_at,
            end=min(self.config.end, market.end_at),
        )
        volume_quality = polymarket_volume_quality(
            probabilities,
            trigger_at=row["as_of"],
            minimum_pre_entry_usdc=self.config.polymarket_volume_minimum_pre_entry_usdc,
            concentration_minimum_usdc=self.config.polymarket_volume_concentration_minimum_usdc,
            max_single_hour_share=self.config.polymarket_volume_max_single_hour_share,
            # Legacy parity (spec §6): pre-portfolio behavior is gate_applied=False
            # (diagnostic only). Only the portfolio path marks the gate applied,
            # keeping legacy-mode entry-decision diagnostics byte-identical.
            gate_applied=self.config.portfolio_enabled,
        )
        observation = observations.get((market.event_id, asset.symbol))
        snapshot = None
        if observation is not None:
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
            snapshot.snapshot_id = await save_model_snapshot(conn, snapshot)
        opened = 0
        entry_decisions: list[dict[str, Any]] = []
        pass1_candidates: list[dict[str, Any]] = []
        portfolio_mode = self.config.portfolio_enabled
        daily_bars_for_adv = await price_bars(
            conn,
            symbol=asset.symbol,
            resolution="1d",
            start=row["as_of"] - timedelta(days=60),
            end=graph_end,
        )
        if snapshot is not None and snapshot.status == "trained":
            entry_bar = next_bar_after(bars, row["as_of"])
            event_open_price = observation.research_data.get("event_open_price")
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
                    realized_price_at_entry=float(entry_bar.open),
                )
                if entry_bar and event_open_price
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
                        if row["as_of"] <= bar.timestamp
                        and bar.timestamp + DAILY_SESSION_LENGTH <= event.end_at
                    ]
                    evaluate_prediction(
                        prediction,
                        event_open_price=float(event_open_price),
                        bars_until_event_end=event_bars,
                    )
                await save_ml_prediction(conn, prediction)
                prediction_log = {
                    "direction": prediction.direction,
                    "classification_probability": prediction.classification_probability,
                    "predicted_peak_percent": prediction.predicted_peak_percent,
                    "predicted_target_price": prediction.predicted_target_price,
                    "realized_move_at_entry": prediction.realized_move_at_entry,
                    "remaining_gap": prediction.remaining_gap,
                    "directions_agree": prediction.directions_agree,
                    "target_reached": prediction.target_reached,
                }
                if prediction.directions_agree and prediction.remaining_gap > 0:
                    trade, decision = await simulate_one_trade(
                        self,
                        market=market,
                        asset=asset,
                        pass_number=row["pass_number"],
                        trigger_at=row["as_of"],
                        portfolio="machine_learning",
                        strategy_branch="machine_learning",
                        direction=prediction.direction,
                        resolution=resolution,
                        bars=bars,
                        graph_bars=available_bars,
                        probabilities=probabilities,
                        predicted_target_price=prediction.predicted_target_price,
                        predicted_target_reached=prediction.target_reached,
                        evaluation_event_end=market.end_at,
                    )
                    decision.update(prediction_log)
                    decision["polymarket_volume_statistics"] = volume_quality
                    entry_decisions.append(decision)
                    if trade:
                        if portfolio_mode:
                            candidate = await assemble_candidate(
                                self,
                                conn,
                                trade=trade,
                                market=market,
                                event=event,
                                bars=bars,
                                daily_bars=daily_bars_for_adv,
                                bar_window_start=bar_start,
                                bar_window_end=graph_end,
                                polymarket_volume_quality=volume_quality,
                                consumes_capital=True,
                                observation_archetype=observation.event_archetype,
                                classification_probability=prediction.classification_probability,
                                predicted_peak_percent=prediction.predicted_peak_percent,
                                remaining_gap=prediction.remaining_gap,
                                directions_agree=prediction.directions_agree,
                                probability_at_entry=decision.get("probability_at_entry"),
                            )
                            if candidate:
                                pass1_candidates.append(candidate_to_dict(candidate))
                                decision["candidate_id"] = candidate.candidate_id
                                decision["opened"] = False
                        else:
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
                            **prediction_log,
                            "polymarket_volume_statistics": volume_quality,
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
                        "polymarket_volume_statistics": volume_quality,
                    }
                )
        elif snapshot is None or snapshot.status in {
            "insufficient_history",
            "insufficient_class_diversity",
        }:
            entry_bar = next_bar_after(bars, row["as_of"])
            completed_before_entry = (
                bars_before(bars, entry_bar.timestamp) if entry_bar else []
            )
            selection = await select_walk_forward_momentum_parameters(
                conn,
                run_id=self.run_id,
                before=row["as_of"],
                resolution=resolution,
                minimum_samples=self.config.momentum_walk_forward_minimum_samples,
                fallback_period=self.config.momentum_lookback_bars,
                fallback_multiplier=self.config.trailing_range_multiplier,
            )
            selected_period = int(selection["range_period"])
            selected_multiplier = float(selection["range_multiplier"])
            selected_trade: Trade | None = None
            selected_decision: dict[str, Any] | None = None
            shadow_results: list[dict[str, object]] = []
            for momentum_lookback in self.config.momentum_lookback_grid:
                momentum = rate_of_change(completed_before_entry, momentum_lookback)
                for range_multiplier in self.config.trailing_range_multiplier_grid:
                    variant = f"n{momentum_lookback}_k{range_multiplier:g}"
                    blocked_reason = None
                    if momentum is None:
                        blocked_reason = "insufficient_price_momentum_history"
                    elif momentum <= 0:
                        blocked_reason = "price_momentum_not_positive"
                    if blocked_reason:
                        shadow_results.append(
                            {
                                "run_id": self.run_id,
                                "market_id": market.market_id,
                                "event_id": market.event_id,
                                "pass_number": row["pass_number"],
                                "symbol": asset.symbol,
                                "trigger_at": row["as_of"],
                                "resolution": resolution,
                                "range_period": momentum_lookback,
                                "range_multiplier": range_multiplier,
                                "opened": False,
                                "reason": blocked_reason,
                                "net_profit": None,
                            }
                        )
                        if (
                            momentum_lookback == selected_period
                            and range_multiplier == selected_multiplier
                        ):
                            selected_decision = {
                                "portfolio": "polymarket_momentum",
                                "strategy_branch": "momentum",
                                "opened": False,
                                "reason": blocked_reason,
                                "price_momentum": momentum,
                            }
                        continue
                    strategy = EventDrivenStrategy(
                        trade_notional=self.config.trade_notional,
                        range_period=momentum_lookback,
                        range_multiplier=range_multiplier,
                    )
                    trade, decision = await simulate_one_trade(
                        self,
                        market=market,
                        asset=asset,
                        pass_number=row["pass_number"],
                        trigger_at=row["as_of"],
                        portfolio=f"momentum_shadow_{variant}",
                        strategy_branch="momentum",
                        direction="long",
                        resolution=resolution,
                        bars=bars,
                        graph_bars=available_bars,
                        probabilities=probabilities,
                        strategy=strategy,
                        momentum_lookback=momentum_lookback,
                        create_graph=False,
                    )
                    shadow_results.append(
                        {
                            "run_id": self.run_id,
                            "market_id": market.market_id,
                            "event_id": market.event_id,
                            "pass_number": row["pass_number"],
                            "symbol": asset.symbol,
                            "trigger_at": row["as_of"],
                            "resolution": resolution,
                            "range_period": momentum_lookback,
                            "range_multiplier": range_multiplier,
                            "opened": trade is not None,
                            "reason": decision["reason"],
                            "net_profit": trade.net_profit if trade else None,
                        }
                    )
                    if (
                        momentum_lookback == selected_period
                        and range_multiplier == selected_multiplier
                    ):
                        selected_trade = trade
                        selected_decision = decision
                        if selected_trade:
                            selected_trade.portfolio = "polymarket_momentum"
                            selected_trade.parameter_selection = dict(selection)
            await save_momentum_parameter_results(conn, shadow_results)
            if selected_decision is None:
                selected_decision = {
                    "portfolio": "polymarket_momentum",
                    "strategy_branch": "momentum",
                    "opened": False,
                    "reason": "selected_parameters_not_in_configured_grid",
                }
            selected_decision.update(
                {
                    "portfolio": "polymarket_momentum",
                    "price_momentum": rate_of_change(completed_before_entry, selected_period),
                    "momentum_lookback": selected_period,
                    "range_multiplier": selected_multiplier,
                    "parameter_selection": selection,
                    "polymarket_volume_statistics": volume_quality,
                }
            )
            entry_decisions.append(selected_decision)
            if selected_trade:
                if portfolio_mode:
                    candidate = await assemble_candidate(
                        self,
                        conn,
                        trade=selected_trade,
                        market=market,
                        event=event,
                        bars=bars,
                        daily_bars=daily_bars_for_adv,
                        bar_window_start=bar_start,
                        bar_window_end=graph_end,
                        polymarket_volume_quality=volume_quality,
                        consumes_capital=True,
                        observation_archetype=(
                            observation.event_archetype if observation else None
                        ),
                        price_momentum=selected_decision.get("price_momentum"),
                        probability_at_entry=selected_decision.get("probability_at_entry"),
                    )
                    if candidate:
                        pass1_candidates.append(candidate_to_dict(candidate))
                        selected_decision["candidate_id"] = candidate.candidate_id
                        selected_decision["opened"] = False
                else:
                    selected_trade.graph_path = str(
                        create_trade_graph(
                            selected_trade,
                            bars=available_bars,
                            probabilities=probabilities,
                            simulation_end=self.config.end,
                            graph_dir=self.run_dir / "graphs" / "trades",
                        )
                    )
                    await save_trade(conn, selected_trade)
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
                "schema_version": 1,
                "model_status": snapshot.status if snapshot else "not_ml_eligible",
                "trades_opened": opened,
                "pass1_candidates": pass1_candidates,
                "entry_decisions": entry_decisions,
            },
        )
    if self.config.portfolio_enabled:
        candidates = await load_pass1_candidates_from_work(conn, run_id=self.run_id)
        await run_portfolio_pass2(self, conn, candidates)
