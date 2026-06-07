"""PostgreSQL repositories for historical backtesting data."""

from database.backtesting.repositories._shared import SCHEMA, json_text, json_value


from database.backtesting.repositories.runs import (
    create_historical_run,
    historical_run,
    update_run,
    start_work,
    finish_work,
    record_stage_failure,
    purge_run,
)

from database.backtesting.repositories.event_decisions import (
    save_event_decision,
    reusable_event_decision,
    accepted_event_ids,
)

from database.backtesting.repositories.market_decisions import (
    save_market_decision,
    reusable_market_decision,
    accepted_market_ids,
)

from database.backtesting.repositories.probabilities import (
    save_probability_history,
    probability_history,
    probability_is_covered,
    save_run_passes,
    save_run_market,
    run_passes,
)

from database.backtesting.repositories.worlds import (
    reusable_world,
    save_world,
    run_world_assets,
    save_world_feedback,
)

from database.backtesting.repositories.research import (
    save_article_set,
    reusable_article_set,
    save_sentiment,
    link_run_sentiment,
    reusable_sentiment,
    sentiment_for_job,
)

from database.backtesting.repositories.prices import (
    save_price_bars,
    price_is_covered,
    price_bars,
    save_asset_metadata,
    asset_metadata,
)

from database.backtesting.repositories.machine_learning import (
    save_ml_observation,
    prior_ml_observations,
    run_ml_observations,
    save_model_snapshot,
    save_ml_prediction,
)

from database.backtesting.repositories.trades import (
    save_trade,
    run_trade_rows,
)

from database.backtesting.repositories.calibration import (
    save_batch_calibration,
    latest_batch_sizes,
)


__all__ = [
    "SCHEMA",
    "json_text",
    "json_value",
    "create_historical_run",
    "historical_run",
    "update_run",
    "start_work",
    "finish_work",
    "record_stage_failure",
    "purge_run",
    "save_event_decision",
    "reusable_event_decision",
    "accepted_event_ids",
    "save_market_decision",
    "reusable_market_decision",
    "accepted_market_ids",
    "save_probability_history",
    "probability_history",
    "probability_is_covered",
    "save_run_passes",
    "save_run_market",
    "run_passes",
    "reusable_world",
    "save_world",
    "run_world_assets",
    "save_world_feedback",
    "save_article_set",
    "reusable_article_set",
    "save_sentiment",
    "link_run_sentiment",
    "reusable_sentiment",
    "sentiment_for_job",
    "save_price_bars",
    "price_is_covered",
    "price_bars",
    "save_asset_metadata",
    "asset_metadata",
    "save_ml_observation",
    "prior_ml_observations",
    "run_ml_observations",
    "save_model_snapshot",
    "save_ml_prediction",
    "save_trade",
    "run_trade_rows",
    "save_batch_calibration",
    "latest_batch_sizes",
]
