from __future__ import annotations

from datetime import timedelta
from typing import Any
import numpy as np
from database.backtesting.repositories.prices import asset_metadata, price_bars
from database.backtesting.repositories.probabilities import probability_history
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import run_resolved_world_assets, save_world_feedback
from database.backtesting.polymarket import probability_as_of
from main_backtesting.models import PriceBar
from strategies.event_driven_ml import DAILY_SESSION_LENGTH

from main_backtesting.stages.event_filter import accepted_markets, run_events


async def run(self, conn: Any) -> None:
    events = {event.event_id: event for event in await run_events(self, conn)}
    markets = {market.market_id: market for market in await accepted_markets(self, conn)}
    for row in await run_resolved_world_assets(conn, self.run_id):
        work_key = f"{row['world_id']}:{row['symbol']}"
        self.current_work_key = work_key
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="world_feedback",
            work_key=work_key,
            payload={"world_id": str(row["world_id"]), "symbol": row["symbol"]},
        ):
            continue
        market = markets[row["market_id"]]
        event = events[market.event_id]
        evaluation_complete = event.end_at <= self.config.historical_data_cutoff
        bars = await price_bars(
            conn,
            symbol=row["symbol"],
            resolution="1d",
            start=event.created_at - timedelta(days=30),
            end=event.end_at + timedelta(days=1),
        )
        baseline = [
            bar
            for bar in bars
            if bar.timestamp + DAILY_SESSION_LENGTH <= event.created_at
        ]
        event_bars = [
            bar
            for bar in bars
            if event.created_at <= bar.timestamp
            and bar.timestamp + DAILY_SESSION_LENGTH <= event.end_at
        ]
        baseline_returns = np.diff(np.log([bar.close for bar in baseline])) if len(baseline) > 1 else np.array([])
        event_returns = np.diff(np.log([bar.close for bar in event_bars])) if len(event_bars) > 1 else np.array([])
        opening = event_bars[0].open if event_bars else None
        changes = [bar.close / opening - 1 for bar in event_bars] if opening else []
        metadata = await asset_metadata(conn, row["symbol"])
        sector_etf = metadata["sector_etf"] if metadata else None
        spy_bars = await price_bars(
            conn,
            symbol="SPY",
            resolution="1d",
            start=event.created_at,
            end=event.end_at + timedelta(days=1),
        )
        sector_bars = (
            await price_bars(
                conn,
                symbol=sector_etf,
                resolution="1d",
                start=event.created_at,
                end=event.end_at + timedelta(days=1),
            )
            if sector_etf
            else []
        )
        spy_bars = [
            bar
            for bar in spy_bars
            if event.created_at <= bar.timestamp
            and bar.timestamp + DAILY_SESSION_LENGTH <= event.end_at
        ]
        sector_bars = [
            bar
            for bar in sector_bars
            if event.created_at <= bar.timestamp
            and bar.timestamp + DAILY_SESSION_LENGTH <= event.end_at
        ]
        probabilities = await probability_history(
            conn,
            market_id=market.market_id,
            start=max(self.config.start, market.created_at),
            end=min(self.config.end, market.end_at),
        )

        def total_return(series: list[PriceBar]) -> float | None:
            if not series or series[0].open <= 0:
                return None
            return series[-1].close / series[0].open - 1.0

        aligned_asset_returns: list[float] = []
        aligned_probability_changes: list[float] = []
        previous_close: float | None = None
        previous_probability: float | None = None
        for bar in event_bars:
            current_probability = probability_as_of(
                probabilities, bar.timestamp + DAILY_SESSION_LENGTH
            )
            if (
                previous_close is not None
                and previous_probability is not None
                and current_probability is not None
                and previous_close > 0
            ):
                aligned_asset_returns.append(bar.close / previous_close - 1.0)
                aligned_probability_changes.append(current_probability - previous_probability)
            previous_close = bar.close
            if current_probability is not None:
                previous_probability = current_probability

        probability_correlation = None
        if (
            len(aligned_asset_returns) > 1
            and np.std(aligned_asset_returns) > 0
            and np.std(aligned_probability_changes) > 0
        ):
            probability_correlation = float(
                np.corrcoef(aligned_asset_returns, aligned_probability_changes)[0, 1]
            )
        asset_return = total_return(event_bars)
        spy_return = total_return(spy_bars)
        sector_return = total_return(sector_bars)
        ml_goal_reached = await conn.fetchval(
            """
            SELECT BOOL_OR(target_reached)
            FROM checking_relevant_events.historical_ml_predictions
            WHERE run_id=$1 AND market_id=$2 AND pass_number=$3 AND symbol=$4
            """,
            self.run_id,
            market.market_id,
            row["pass_number"],
            row["symbol"],
        )
        trade_net_profit = await conn.fetchval(
            """
            SELECT SUM(net_profit)
            FROM checking_relevant_events.historical_trades
            WHERE run_id=$1 AND market_id=$2 AND pass_number=$3 AND symbol=$4
            """,
            self.run_id,
            market.market_id,
            row["pass_number"],
            row["symbol"],
        )
        realized_vol = float(np.std(event_returns)) if len(event_returns) else None
        baseline_vol = float(np.std(baseline_returns)) if len(baseline_returns) else None
        metrics = {
            "realized_volatility": realized_vol,
            "baseline_volatility": baseline_vol,
            "volatility_increase": (
                realized_vol - baseline_vol
                if realized_vol is not None and baseline_vol is not None
                else None
            ),
            "probability_correlation": probability_correlation,
            "maximum_favorable_move": max(changes) if changes else None,
            "maximum_adverse_move": min(changes) if changes else None,
            "return_vs_spy": (
                asset_return - spy_return
                if asset_return is not None and spy_return is not None
                else None
            ),
            "return_vs_sector": (
                asset_return - sector_return
                if asset_return is not None and sector_return is not None
                else None
            ),
            "ml_goal_reached": ml_goal_reached,
            "trade_net_profit": float(trade_net_profit) if trade_net_profit is not None else None,
            "evaluation_complete": evaluation_complete,
            "sector_etf": sector_etf,
        }
        await save_world_feedback(
            conn,
            run_id=self.run_id,
            world_id=row["world_id"],
            symbol=row["symbol"],
            metrics=metrics,
        )
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="world_feedback",
            work_key=work_key,
            result=metrics,
        )
