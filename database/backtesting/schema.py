from __future__ import annotations

import asyncpg

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_runs (
    run_id       UUID PRIMARY KEY,
    started_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at  TIMESTAMPTZ,
    status       TEXT NOT NULL,
    config       JSONB NOT NULL,
    error        TEXT
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_event_decisions (
    run_id       UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    event_id     TEXT NOT NULL,
    event_title  TEXT NOT NULL,
    relevant     BOOLEAN NOT NULL,
    reason       TEXT NOT NULL,
    raw_output   JSONB NOT NULL,
    PRIMARY KEY (run_id, event_id)
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_market_passes (
    run_id                  UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
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

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_asset_worlds (
    run_id        UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    market_id     TEXT NOT NULL,
    model_name    TEXT NOT NULL,
    raw_output    JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, market_id)
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_sentiment_results (
    run_id          UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    market_id       TEXT NOT NULL,
    pass_number     INTEGER NOT NULL,
    symbol          TEXT NOT NULL,
    provider        TEXT NOT NULL,
    label           TEXT NOT NULL,
    score           DOUBLE PRECISION NOT NULL,
    article_count   INTEGER NOT NULL,
    details         JSONB NOT NULL,
    PRIMARY KEY (run_id, market_id, pass_number, symbol, provider)
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_news_articles (
    run_id        UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    market_id     TEXT NOT NULL,
    pass_number   INTEGER NOT NULL,
    symbol        TEXT NOT NULL,
    url           TEXT NOT NULL,
    title         TEXT NOT NULL,
    published_at  TIMESTAMPTZ NOT NULL,
    domain        TEXT,
    article_text  TEXT NOT NULL,
    PRIMARY KEY (run_id, market_id, pass_number, symbol, url)
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_skips (
    skip_id       BIGSERIAL PRIMARY KEY,
    run_id        UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    event_id      TEXT,
    market_id     TEXT,
    pass_number   INTEGER,
    symbol        TEXT,
    stage         TEXT NOT NULL,
    reason        TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS checking_relevant_events.backtest_trades (
    trade_id            UUID PRIMARY KEY,
    run_id              UUID NOT NULL REFERENCES checking_relevant_events.backtest_runs(run_id) ON DELETE CASCADE,
    market_id           TEXT NOT NULL,
    event_id            TEXT NOT NULL,
    question            TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    asset_name          TEXT NOT NULL,
    pass_number         INTEGER NOT NULL,
    trigger_at          TIMESTAMPTZ NOT NULL,
    entry_at            TIMESTAMPTZ NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL,
    quantity            DOUBLE PRECISION NOT NULL,
    entry_commission    DOUBLE PRECISION NOT NULL,
    initial_stop        DOUBLE PRECISION NOT NULL,
    exit_at             TIMESTAMPTZ,
    exit_price          DOUBLE PRECISION,
    exit_commission     DOUBLE PRECISION NOT NULL,
    exit_reason         TEXT,
    final_mark_price    DOUBLE PRECISION,
    maximum_price       DOUBLE PRECISION,
    minimum_price       DOUBLE PRECISION,
    final_outcome       TEXT,
    gross_profit        DOUBLE PRECISION,
    net_profit          DOUBLE PRECISION,
    maximum_profit      DOUBLE PRECISION,
    maximum_loss        DOUBLE PRECISION,
    stop_history        JSONB NOT NULL,
    graph_path          TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_market
    ON checking_relevant_events.backtest_trades(run_id, market_id);
"""


async def reset_backtesting_schema(conn: asyncpg.Connection) -> None:
    await conn.execute(SCHEMA_SQL)
    await conn.execute(
        "TRUNCATE checking_relevant_events.backtest_runs CASCADE"
    )
