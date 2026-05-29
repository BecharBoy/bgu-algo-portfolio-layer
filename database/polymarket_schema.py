from __future__ import annotations

import asyncpg

SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS polymarket;

CREATE TABLE IF NOT EXISTS polymarket.events (
    event_id            TEXT PRIMARY KEY,
    slug                TEXT,
    title               TEXT,
    description         TEXT,
    category            TEXT,
    tags                TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    matched_target_tags TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    created_at          TIMESTAMPTZ,
    start_at            TIMESTAMPTZ,
    end_at              TIMESTAMPTZ,
    closed_at           TIMESTAMPTZ,
    duration_days       DOUBLE PRECISION,
    active              BOOLEAN,
    closed              BOOLEAN,
    archived            BOOLEAN,
    competitive         DOUBLE PRECISION,
    volume              DOUBLE PRECISION,
    volume_24hr         DOUBLE PRECISION,
    volume_1wk          DOUBLE PRECISION,
    volume_1mo          DOUBLE PRECISION,
    volume_1yr          DOUBLE PRECISION,
    open_interest       DOUBLE PRECISION,
    liquidity_amm       DOUBLE PRECISION,
    liquidity_clob      DOUBLE PRECISION,
    raw_event           JSONB NOT NULL,
    first_ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS polymarket.event_target_tags (
    event_id   TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    tag_slug   TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_id, tag_slug)
);

CREATE TABLE IF NOT EXISTS polymarket.market_groups (
    group_id         TEXT PRIMARY KEY,
    example_question TEXT NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS polymarket.market_group_members (
    group_id            TEXT NOT NULL REFERENCES polymarket.market_groups(group_id) ON DELETE CASCADE,
    question            TEXT NOT NULL,
    normalized_question TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (group_id, normalized_question)
);

CREATE TABLE IF NOT EXISTS polymarket.market_group_ml_weights (
    group_id              TEXT NOT NULL REFERENCES polymarket.market_groups(group_id) ON DELETE CASCADE,
    model_name            TEXT NOT NULL CHECK (model_name IN ('classification', 'regression')),
    training_sample_count INTEGER NOT NULL CHECK (training_sample_count >= 0),
    trained_at            TIMESTAMPTZ NOT NULL,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS polymarket.markets (
    market_id            TEXT PRIMARY KEY,
    event_id             TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    group_id             TEXT REFERENCES polymarket.market_groups(group_id),
    condition_id         TEXT,
    slug                 TEXT,
    question             TEXT,
    group_item_title     TEXT,
    group_item_threshold TEXT,
    market_type          TEXT,
    format_type          TEXT,
    outcomes             TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    outcome_prices       DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
    yes_token_id         TEXT,
    no_token_id          TEXT,
    resolved_outcome     TEXT,
    did_happen           BOOLEAN,
    resolution_status    TEXT NOT NULL DEFAULT 'unknown',
    active               BOOLEAN,
    closed               BOOLEAN,
    accepting_orders     BOOLEAN,
    market_created_at    TIMESTAMPTZ,
    start_at             TIMESTAMPTZ,
    end_at               TIMESTAMPTZ,
    closed_at            TIMESTAMPTZ,
    volume               DOUBLE PRECISION,
    volume_24hr          DOUBLE PRECISION,
    volume_1wk           DOUBLE PRECISION,
    liquidity_amm        DOUBLE PRECISION,
    liquidity_clob       DOUBLE PRECISION,
    best_bid             DOUBLE PRECISION,
    best_ask             DOUBLE PRECISION,
    spread               DOUBLE PRECISION,
    last_trade_price     DOUBLE PRECISION,
    yes_percentage       DOUBLE PRECISION CHECK (yes_percentage IS NULL OR (yes_percentage >= 0 AND yes_percentage <= 100)),
    triggered_70         BOOLEAN NOT NULL DEFAULT FALSE,
    raw_market           JSONB NOT NULL,
    raw_market_detail    JSONB NOT NULL,
    first_ingested_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE polymarket.markets
    ADD COLUMN IF NOT EXISTS group_id TEXT REFERENCES polymarket.market_groups(group_id),
    ADD COLUMN IF NOT EXISTS market_created_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS yes_percentage DOUBLE PRECISION CHECK (yes_percentage IS NULL OR (yes_percentage >= 0 AND yes_percentage <= 100)),
    ADD COLUMN IF NOT EXISTS triggered_70 BOOLEAN NOT NULL DEFAULT FALSE;

CREATE INDEX IF NOT EXISTS idx_polymarket_markets_event_id
    ON polymarket.markets(event_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_group_id
    ON polymarket.markets(group_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_yes_token
    ON polymarket.markets(yes_token_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_resolution
    ON polymarket.markets(did_happen, resolution_status);

CREATE TABLE IF NOT EXISTS polymarket.market_probability_1m (
    market_id               TEXT NOT NULL REFERENCES polymarket.markets(market_id) ON DELETE CASCADE,
    event_id                TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    yes_token_id            TEXT NOT NULL,
    ts                      TIMESTAMPTZ NOT NULL,
    available_at            TIMESTAMPTZ NOT NULL,
    yes_probability         DOUBLE PRECISION NOT NULL CHECK (yes_probability >= 0 AND yes_probability <= 1),
    source_fidelity_minutes INTEGER NOT NULL DEFAULT 1,
    ingested_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (market_id, ts)
);

CREATE INDEX IF NOT EXISTS idx_polymarket_probability_event_ts
    ON polymarket.market_probability_1m(event_id, ts);
CREATE INDEX IF NOT EXISTS idx_polymarket_probability_available_at
    ON polymarket.market_probability_1m(available_at);
CREATE INDEX IF NOT EXISTS idx_polymarket_probability_token_ts
    ON polymarket.market_probability_1m(yes_token_id, ts);

CREATE TABLE IF NOT EXISTS polymarket.ingestion_runs (
    run_id              UUID PRIMARY KEY,
    data_start          TIMESTAMPTZ NOT NULL,
    data_end            TIMESTAMPTZ NOT NULL,
    min_duration_days   DOUBLE PRECISION NOT NULL,
    max_duration_days   DOUBLE PRECISION NOT NULL,
    target_tag_slugs    TEXT[] NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at         TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'running',
    events_discovered   INTEGER NOT NULL DEFAULT 0,
    markets_seen        INTEGER NOT NULL DEFAULT 0,
    markets_ingested    INTEGER NOT NULL DEFAULT 0,
    probability_rows    BIGINT NOT NULL DEFAULT 0,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS polymarket.market_ingestion_state (
    market_id     TEXT PRIMARY KEY REFERENCES polymarket.markets(market_id) ON DELETE CASCADE,
    event_id      TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    last_run_id   UUID REFERENCES polymarket.ingestion_runs(run_id),
    status        TEXT NOT NULL,
    history_start TIMESTAMPTZ,
    history_end   TIMESTAMPTZ,
    row_count     BIGINT NOT NULL DEFAULT 0,
    last_error    TEXT,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'polymarket'
          AND table_name = 'events'
          AND column_name = 'competitive'
          AND data_type = 'boolean'
    ) THEN
        ALTER TABLE polymarket.events
            ALTER COLUMN competitive TYPE DOUBLE PRECISION
            USING CASE
                WHEN competitive IS NULL THEN NULL
                WHEN competitive THEN 1.0
                ELSE 0.0
            END;
    END IF;
END $$;
"""


async def init_polymarket_schema(conn: asyncpg.Connection) -> None:
    await conn.execute(SCHEMA_SQL)
