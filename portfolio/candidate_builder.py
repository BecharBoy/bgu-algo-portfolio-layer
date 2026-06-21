from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from main_backtesting.models import PriceBar, Trade
from main_backtesting.utils import event_archetype
from portfolio.models import TradeCandidate
from strategies.event_driven_long import average_true_range


def candidate_id_for(
    *,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    symbol: str,
    strategy_branch: str,
) -> str:
    return f"{run_id}:{market_id}:{pass_number}:{symbol}:{strategy_branch}"


def compute_adv(daily_bars: list[PriceBar], entry_at: datetime, *, lookback: int, min_bars: int) -> float | None:
    eligible = [bar for bar in daily_bars if bar.timestamp <= entry_at][-lookback:]
    if len(eligible) < min_bars:
        return None
    return sum(bar.volume for bar in eligible) / len(eligible)


def partition_tags(
    event_tags: list[str],
    market_tags: list[str],
    *,
    theme_tags: set[str],
    geo_country_tags: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    combined = {tag.lower().strip() for tag in event_tags + market_tags}
    themes = tuple(sorted(combined & {tag.lower() for tag in theme_tags}))
    countries = tuple(sorted(combined & {tag.lower() for tag in geo_country_tags}))
    return themes, countries


def build_trade_candidate(
    *,
    trade: Trade,
    atr: float,
    consumes_capital: bool,
    sector_key: str,
    theme_tags: tuple[str, ...],
    country_tags: tuple[str, ...],
    adv: float | None,
    polymarket_volume_quality: dict[str, Any],
    bar_window_start: datetime,
    bar_window_end: datetime,
    event_archetype_value: str | None,
    classification_probability: float | None = None,
    predicted_peak_percent: float | None = None,
    remaining_gap: float | None = None,
    directions_agree: bool | None = None,
    price_momentum: float | None = None,
    probability_at_entry: float | None = None,
) -> TradeCandidate | None:
    if trade.exit_at is None or trade.exit_price is None or trade.exit_reason is None:
        return None
    stop_distance = abs(trade.entry_price - trade.initial_stop)
    if stop_distance <= 0:
        return None
    return TradeCandidate(
        candidate_id=candidate_id_for(
            run_id=trade.run_id,
            market_id=trade.market_id,
            pass_number=trade.pass_number,
            symbol=trade.symbol,
            strategy_branch=trade.strategy_branch,
        ),
        run_id=trade.run_id,
        market_id=trade.market_id,
        event_id=trade.event_id,
        symbol=trade.symbol,
        asset_name=trade.asset_name,
        pass_number=trade.pass_number,
        event_archetype=event_archetype_value,
        strategy_branch=trade.strategy_branch,
        portfolio_label=trade.portfolio,
        direction=trade.direction,
        consumes_capital=consumes_capital,
        trigger_at=trade.trigger_at,
        entry_at=trade.entry_at,
        resolution=trade.resolution,
        entry_price=trade.entry_price,
        exit_at=trade.exit_at,
        exit_price=trade.exit_price,
        exit_reason=trade.exit_reason,
        path_reference_quantity=trade.quantity,
        initial_stop=trade.initial_stop,
        atr=atr,
        stop_distance=stop_distance,
        range_period=trade.range_period or 14,
        range_multiplier=trade.range_multiplier or 3.0,
        classification_probability=classification_probability,
        predicted_peak_percent=predicted_peak_percent,
        predicted_target_price=trade.predicted_target_price,
        remaining_gap=remaining_gap,
        directions_agree=directions_agree,
        price_momentum=price_momentum,
        probability_at_entry=probability_at_entry,
        sector_key=sector_key or "UNKNOWN",
        theme_tags=theme_tags,
        country_tags=country_tags,
        adv=adv,
        polymarket_volume_quality=polymarket_volume_quality,
        question=trade.question,
        final_outcome=trade.final_outcome,
        bar_lookup_key=f"{trade.symbol}:{trade.resolution}",
        bar_window_start=bar_window_start,
        bar_window_end=bar_window_end,
        momentum_parameter_selection=trade.parameter_selection or None,
    )


def infer_archetype(
    *,
    observation_archetype: str | None,
    event_tags: list[str],
    question: str,
    symbol: str,
) -> str | None:
    if observation_archetype:
        return observation_archetype
    return event_archetype(event_tags, question=question, symbol=symbol)


def atr_from_trade(trade: Trade, previous_bars: list[PriceBar], period: int) -> float | None:
    return average_true_range(previous_bars, period)
