from __future__ import annotations

import asyncpg

SCHEMA = "checking_relevant_events"

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_backtest_runs (
    run_id                  UUID PRIMARY KEY,
    started_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at             TIMESTAMPTZ,
    status                  TEXT NOT NULL,
    current_stage           TEXT,
    config                  JSONB NOT NULL,
    hourly_boundary         TIMESTAMPTZ NOT NULL,
    output_dir              TEXT NOT NULL,
    error                   TEXT
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_backtest_stage_work (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    stage                   TEXT NOT NULL,
    work_key                TEXT NOT NULL,
    status                  TEXT NOT NULL DEFAULT 'pending',
    attempts                INTEGER NOT NULL DEFAULT 0,
    payload                 JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    result                  JSONB,
    error                   TEXT,
    started_at              TIMESTAMPTZ,
    finished_at             TIMESTAMPTZ,
    PRIMARY KEY (run_id, stage, work_key)
);

CREATE INDEX IF NOT EXISTS idx_historical_stage_work_status
    ON {SCHEMA}.historical_backtest_stage_work(run_id, stage, status);

CREATE TABLE IF NOT EXISTS {SCHEMA}.gdelt_request_schedule (
    schedule_name              TEXT PRIMARY KEY,
    last_request_started_at    TIMESTAMPTZ,
    last_request_completed_at  TIMESTAMPTZ,
    last_status                INTEGER,
    updated_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Reusable historical model and download data. These rows are never removed by a run.
CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_event_decisions (
    input_hash              TEXT PRIMARY KEY,
    event_id                TEXT NOT NULL,
    event_title             TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    llm_input               JSONB NOT NULL,
    llm_output              JSONB NOT NULL,
    relevant                BOOLEAN NOT NULL,
    reason                  TEXT NOT NULL,
    processed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_market_decisions (
    input_hash              TEXT PRIMARY KEY,
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    event_title             TEXT NOT NULL,
    market_question         TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    llm_input               JSONB NOT NULL,
    llm_output              JSONB NOT NULL,
    relevant                BOOLEAN NOT NULL,
    reason                  TEXT NOT NULL,
    processed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_probability_points (
    market_id               TEXT NOT NULL,
    yes_token_id            TEXT NOT NULL,
    hour_ts                 TIMESTAMPTZ NOT NULL,
    source_ts               TIMESTAMPTZ NOT NULL,
    available_at            TIMESTAMPTZ NOT NULL,
    probability             DOUBLE PRECISION NOT NULL CHECK (probability >= 0 AND probability <= 1),
    volume_usdc             DOUBLE PRECISION,
    downloaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market_id, hour_ts)
);

ALTER TABLE {SCHEMA}.historical_probability_points
    ADD COLUMN IF NOT EXISTS volume_usdc DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_probability_coverage (
    market_id               TEXT PRIMARY KEY,
    yes_token_id            TEXT NOT NULL,
    requested_start         TIMESTAMPTZ NOT NULL,
    requested_end           TIMESTAMPTZ NOT NULL,
    first_hour              TIMESTAMPTZ,
    last_hour               TIMESTAMPTZ,
    row_count               BIGINT NOT NULL,
    completed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_asset_worlds (
    world_id                UUID PRIMARY KEY,
    input_hash              TEXT NOT NULL UNIQUE,
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    as_of                   TIMESTAMPTZ NOT NULL,
    model_name              TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    llm_input               JSONB NOT NULL,
    llm_output              JSONB NOT NULL,
    universe_name           TEXT NOT NULL,
    universe_reason         TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_asset_world_assets (
    world_id                UUID NOT NULL REFERENCES {SCHEMA}.historical_asset_worlds(world_id) ON DELETE CASCADE,
    symbol                  TEXT NOT NULL,
    asset_name              TEXT NOT NULL,
    asset_class             TEXT NOT NULL,
    reason                  TEXT NOT NULL,
    PRIMARY KEY (world_id, symbol)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_articles (
    url                     TEXT PRIMARY KEY,
    title                   TEXT NOT NULL,
    published_at            TIMESTAMPTZ NOT NULL,
    domain                  TEXT,
    article_text            TEXT NOT NULL,
    downloaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_article_sets (
    article_set_id          UUID PRIMARY KEY,
    input_hash              TEXT NOT NULL UNIQUE,
    market_id               TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    as_of                   TIMESTAMPTZ NOT NULL,
    symbol                  TEXT NOT NULL,
    query                   TEXT NOT NULL,
    window_start            TIMESTAMPTZ NOT NULL,
    window_end              TIMESTAMPTZ NOT NULL,
    query_settings          JSONB NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_article_set_items (
    article_set_id          UUID NOT NULL REFERENCES {SCHEMA}.historical_article_sets(article_set_id) ON DELETE CASCADE,
    url                     TEXT NOT NULL REFERENCES {SCHEMA}.historical_articles(url),
    PRIMARY KEY (article_set_id, url)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_sentiment_results (
    input_hash              TEXT PRIMARY KEY,
    article_set_id          UUID NOT NULL REFERENCES {SCHEMA}.historical_article_sets(article_set_id),
    market_id               TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    provider                TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    prompt_version          TEXT NOT NULL,
    model_input             JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    model_output            JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    label                   TEXT NOT NULL,
    score                   DOUBLE PRECISION NOT NULL,
    details                 JSONB NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE {SCHEMA}.historical_sentiment_results
    ADD COLUMN IF NOT EXISTS model_input JSONB NOT NULL DEFAULT '{{}}'::JSONB;
ALTER TABLE {SCHEMA}.historical_sentiment_results
    ADD COLUMN IF NOT EXISTS model_output JSONB NOT NULL DEFAULT '{{}}'::JSONB;

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_price_bars (
    symbol                  TEXT NOT NULL,
    resolution              TEXT NOT NULL CHECK (resolution IN ('1h', '1d')),
    ts                      TIMESTAMPTZ NOT NULL,
    open                    DOUBLE PRECISION NOT NULL,
    high                    DOUBLE PRECISION NOT NULL,
    low                     DOUBLE PRECISION NOT NULL,
    close                   DOUBLE PRECISION NOT NULL,
    volume                  DOUBLE PRECISION NOT NULL,
    downloaded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, resolution, ts)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_price_coverage (
    symbol                  TEXT NOT NULL,
    resolution              TEXT NOT NULL CHECK (resolution IN ('1h', '1d')),
    requested_start         TIMESTAMPTZ NOT NULL,
    requested_end           TIMESTAMPTZ NOT NULL,
    first_ts                TIMESTAMPTZ,
    last_ts                 TIMESTAMPTZ,
    row_count               BIGINT NOT NULL,
    completed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, resolution)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_price_download_windows (
    symbol                  TEXT NOT NULL,
    resolution              TEXT NOT NULL CHECK (resolution IN ('1h', '1d')),
    requested_start         TIMESTAMPTZ NOT NULL,
    requested_end           TIMESTAMPTZ NOT NULL,
    first_ts                TIMESTAMPTZ,
    last_ts                 TIMESTAMPTZ,
    row_count               BIGINT NOT NULL,
    completed_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, resolution, requested_start, requested_end)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_asset_metadata (
    symbol                  TEXT PRIMARY KEY,
    asset_name              TEXT,
    sector                  TEXT,
    sector_etf              TEXT,
    metadata                JSONB NOT NULL,
    missing_reason          TEXT,
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_ml_observations (
    observation_id          UUID PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    event_id                TEXT NOT NULL,
    market_id               TEXT NOT NULL,
    first_pass_number       INTEGER NOT NULL,
    first_pass_at           TIMESTAMPTZ NOT NULL,
    label_available_at      TIMESTAMPTZ NOT NULL,
    symbol                  TEXT NOT NULL,
    event_archetype         TEXT NOT NULL,
    resolution              TEXT NOT NULL CHECK (resolution IN ('1h', '1d')),
    features                JSONB NOT NULL,
    research_data           JSONB NOT NULL,
    classification_target   INTEGER CHECK (classification_target IN (-1, 1)),
    regression_target       DOUBLE PRECISION,
    valid_for_training      BOOLEAN NOT NULL,
    exclusion_reason        TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, event_id, symbol)
);

ALTER TABLE {SCHEMA}.historical_ml_observations
    ADD COLUMN IF NOT EXISTS label_available_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_historical_ml_prior_observations
    ON {SCHEMA}.historical_ml_observations(
        run_id, symbol, event_archetype, label_available_at
    );

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_ml_model_snapshots (
    snapshot_id             UUID PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    symbol                  TEXT NOT NULL,
    event_archetype         TEXT NOT NULL,
    training_cutoff         TIMESTAMPTZ NOT NULL,
    training_event_ids      TEXT[] NOT NULL,
    training_sample_count   INTEGER NOT NULL,
    status                  TEXT NOT NULL,
    feature_names           TEXT[] NOT NULL,
    feature_means           JSONB NOT NULL,
    feature_scales          JSONB NOT NULL,
    classifier_coefficients JSONB,
    classifier_intercept    DOUBLE PRECISION,
    ridge_coefficients      JSONB,
    ridge_intercept         DOUBLE PRECISION,
    hyperparameters         JSONB NOT NULL,
    validation_metrics      JSONB NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, symbol, event_archetype, training_cutoff)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_world_feedback (
    run_id                      UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    world_id                    UUID NOT NULL REFERENCES {SCHEMA}.historical_asset_worlds(world_id),
    symbol                      TEXT NOT NULL,
    realized_volatility         DOUBLE PRECISION,
    baseline_volatility         DOUBLE PRECISION,
    volatility_increase         DOUBLE PRECISION,
    probability_correlation     DOUBLE PRECISION,
    maximum_favorable_move      DOUBLE PRECISION,
    maximum_adverse_move        DOUBLE PRECISION,
    return_vs_spy               DOUBLE PRECISION,
    return_vs_sector            DOUBLE PRECISION,
    ml_goal_reached             BOOLEAN,
    trade_net_profit            DOUBLE PRECISION,
    human_valid                 BOOLEAN,
    human_notes                 TEXT,
    metrics                     JSONB NOT NULL DEFAULT '{{}}'::JSONB,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, world_id, symbol)
);

-- Run-specific derived state.
CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_event_decisions (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    event_id                TEXT NOT NULL,
    input_hash              TEXT NOT NULL REFERENCES {SCHEMA}.historical_event_decisions(input_hash),
    PRIMARY KEY (run_id, event_id)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_market_decisions (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    market_id               TEXT NOT NULL,
    input_hash              TEXT NOT NULL REFERENCES {SCHEMA}.historical_market_decisions(input_hash),
    PRIMARY KEY (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_market_passes (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    question                TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    above_at                TIMESTAMPTZ NOT NULL,
    above_probability       DOUBLE PRECISION NOT NULL,
    fell_below_at           TIMESTAMPTZ,
    fell_below_probability  DOUBLE PRECISION,
    final_outcome           TEXT,
    PRIMARY KEY (run_id, market_id, pass_number)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_markets (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    question                TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL,
    end_at                  TIMESTAMPTZ NOT NULL,
    final_outcome           TEXT,
    probability_hour_count  BIGINT NOT NULL,
    probability_graph_path  TEXT NOT NULL,
    PRIMARY KEY (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_worlds (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    market_id               TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    world_id                UUID NOT NULL REFERENCES {SCHEMA}.historical_asset_worlds(world_id),
    PRIMARY KEY (run_id, market_id, pass_number)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_sentiments (
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    market_id               TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    provider                TEXT NOT NULL,
    input_hash              TEXT NOT NULL REFERENCES {SCHEMA}.historical_sentiment_results(input_hash),
    PRIMARY KEY (run_id, market_id, pass_number, symbol, provider)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_ml_predictions (
    prediction_id           UUID PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    snapshot_id             UUID REFERENCES {SCHEMA}.historical_ml_model_snapshots(snapshot_id),
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    direction               TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    classification_probability DOUBLE PRECISION,
    predicted_peak_percent  DOUBLE PRECISION NOT NULL,
    predicted_target_price  DOUBLE PRECISION NOT NULL,
    realized_move_at_entry  DOUBLE PRECISION NOT NULL,
    remaining_gap           DOUBLE PRECISION NOT NULL,
    directions_agree        BOOLEAN NOT NULL DEFAULT FALSE,
    target_reached          BOOLEAN,
    target_reached_at       TIMESTAMPTZ,
    actual_max_favorable    DOUBLE PRECISION,
    actual_max_adverse      DOUBLE PRECISION,
    actual_direction        TEXT,
    classification_correct  BOOLEAN,
    regression_error        DOUBLE PRECISION,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE {SCHEMA}.historical_ml_predictions
    ADD COLUMN IF NOT EXISTS directions_agree BOOLEAN NOT NULL DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_trades (
    trade_id                UUID PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    portfolio               TEXT NOT NULL,
    strategy_branch         TEXT NOT NULL,
    resolution              TEXT NOT NULL CHECK (resolution IN ('1h', '1d')),
    direction               TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    market_id               TEXT NOT NULL,
    event_id                TEXT NOT NULL,
    question                TEXT NOT NULL,
    symbol                  TEXT NOT NULL,
    asset_name              TEXT NOT NULL,
    pass_number             INTEGER NOT NULL,
    trigger_at              TIMESTAMPTZ NOT NULL,
    entry_at                TIMESTAMPTZ NOT NULL,
    entry_price             DOUBLE PRECISION NOT NULL,
    quantity                DOUBLE PRECISION NOT NULL,
    entry_commission        DOUBLE PRECISION NOT NULL,
    initial_stop            DOUBLE PRECISION NOT NULL,
    exit_at                 TIMESTAMPTZ,
    exit_price              DOUBLE PRECISION,
    exit_commission         DOUBLE PRECISION NOT NULL,
    exit_reason             TEXT,
    final_mark_price        DOUBLE PRECISION,
    maximum_price           DOUBLE PRECISION,
    minimum_price           DOUBLE PRECISION,
    final_outcome           TEXT,
    predicted_target_price  DOUBLE PRECISION,
    gross_profit            DOUBLE PRECISION,
    net_profit              DOUBLE PRECISION,
    maximum_profit          DOUBLE PRECISION,
    maximum_loss            DOUBLE PRECISION,
    stop_history            JSONB NOT NULL,
    graph_path              TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, portfolio, market_id, pass_number, symbol)
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_run_failures (
    failure_id              BIGSERIAL PRIMARY KEY,
    run_id                  UUID NOT NULL REFERENCES {SCHEMA}.historical_backtest_runs(run_id) ON DELETE CASCADE,
    stage                   TEXT NOT NULL,
    work_key                TEXT,
    error_type              TEXT NOT NULL,
    error                   TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {SCHEMA}.historical_batch_calibrations (
    calibration_id          UUID PRIMARY KEY,
    task                    TEXT NOT NULL,
    model_name              TEXT NOT NULL,
    tested_sizes            JSONB NOT NULL,
    selected_batch_size     INTEGER NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


async def initialize_historical_schema(conn: asyncpg.Connection) -> None:
    await conn.execute(SCHEMA_SQL)


async def reset_backtesting_schema(conn: asyncpg.Connection) -> None:
    """Compatibility entry point. Historical runs are never truncated."""
    await initialize_historical_schema(conn)
