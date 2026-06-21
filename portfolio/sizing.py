from __future__ import annotations

import math

from portfolio.config import PortfolioConfig


def floor_to_lot(quantity: float) -> float:
    return float(math.floor(quantity))


def effective_risk_pct(config: PortfolioConfig, drawdown: float) -> float:
    if not config.drawdown_derisk_schedule:
        return config.risk_per_trade_pct
    factor = 1.0
    for threshold in sorted(config.drawdown_derisk_schedule):
        if drawdown >= threshold:
            factor = config.drawdown_derisk_schedule[threshold]
    return config.risk_per_trade_pct * factor


def risk_target_quantity(
    *,
    config: PortfolioConfig,
    equity: float,
    entry_price: float,
    stop_distance: float,
    drawdown: float,
) -> tuple[float, float, float, float]:
    effective_pct = effective_risk_pct(config, drawdown)
    risk_dollars = effective_pct * equity
    if config.sizing_mode == "fixed_fraction":
        notional_risk = config.fixed_fraction_pct * equity
        qty_risk = notional_risk / entry_price if entry_price > 0 else 0.0
    else:
        qty_risk = risk_dollars / stop_distance if stop_distance > 0 else 0.0
        notional_risk = qty_risk * entry_price
    return qty_risk, notional_risk, risk_dollars, effective_pct
