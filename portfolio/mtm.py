from __future__ import annotations

import math
from datetime import datetime

from main_backtesting.models import PriceBar
from portfolio.models import PortfolioState, Position


def build_bar_index(
    bars_by_key: dict[str, list[PriceBar]],
) -> dict[str, list[PriceBar]]:
    indexed: dict[str, list[PriceBar]] = {}
    for key, bars in bars_by_key.items():
        indexed[key] = sorted(bars, key=lambda bar: bar.timestamp)
    return indexed


def last_close_at_or_before(bars: list[PriceBar], timestamp: datetime) -> float | None:
    if not bars:
        return None
    selected: PriceBar | None = None
    for bar in bars:
        if bar.timestamp <= timestamp:
            selected = bar
        else:
            break
    return selected.close if selected is not None else None


def mark_to_market(
    state: PortfolioState,
    *,
    timestamp: datetime,
    bars_by_key: dict[str, list[PriceBar]],
) -> float:
    position_value = 0.0
    for position in state.open_positions.values():
        bars = bars_by_key.get(f"{position.symbol}:{position.resolution}", [])
        mark = last_close_at_or_before(bars, timestamp)
        if mark is None:
            mark = position.entry_price
        if position.direction == "long":
            position_value += position.quantity * mark
        else:
            position_value += position.quantity * (2 * position.entry_price - mark)
    equity = state.cash + position_value
    if math.isnan(equity):
        raise RuntimeError("PortfolioState invariant violation: NaN equity")
    return equity


def refresh_state_marks(
    state: PortfolioState,
    *,
    timestamp: datetime,
    bars_by_key: dict[str, list[PriceBar]],
    kill_switch_drawdown_pct: float,
) -> None:
    state.equity = mark_to_market(state, timestamp=timestamp, bars_by_key=bars_by_key)
    if state.equity > state.peak_equity:
        state.peak_equity = state.equity
    state.drawdown = (
        (state.peak_equity - state.equity) / state.peak_equity if state.peak_equity > 0 else 0.0
    )
    state.kill_switch_active = (
        state.equity <= 0 or state.drawdown >= kill_switch_drawdown_pct
    )
