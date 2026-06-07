"""Historical backtest stage implementations."""

from . import (
    event_filter,
    probabilities,
    asset_worlds,
    prices,
    ml_observations,
    simulation,
    reports,
)

STAGES = [
    "event_filter",
    "probabilities",
    "asset_worlds",
    "prices",
    "ml_observations",
    "simulation",
    "reports",
]

STAGE_FUNCTIONS = [
    event_filter.run,
    probabilities.run,
    asset_worlds.run,
    prices.run,
    ml_observations.run,
    simulation.run,
    reports.run,
]

__all__ = ["STAGES", "STAGE_FUNCTIONS"]
