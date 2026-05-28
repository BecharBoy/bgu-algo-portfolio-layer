"""
DB-only Polymarket 1-minute probability ingestion.

Pulls events from 2022-01-01 through 2026-05-01, filtered by:
  - target tag slugs from getting_all_markets_2026_poly.py
  - event duration between 5 and 60 days

For each eligible binary market, the script stores:
  - event metadata
  - market metadata
  - matched target tags
  - the YES token's 1-minute probability history
  - whether the event happened, when Polymarket resolution can be inferred

No CSV or JSON files are written. The database stores raw API payloads in JSONB
columns for auditability, while the time-series table stays narrow and backtest-ready.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from dotenv import load_dotenv

sys.stdout.reconfigure(errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
GAMMA_EVENTS_KEYSET_API = "https://gamma-api.polymarket.com/events/keyset"
GAMMA_MARKET_API = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_HISTORY_API = "https://clob.polymarket.com/prices-history"

DEFAULT_DATA_START = "2022-01-01T00:00:00Z"
DEFAULT_DATA_END = "2026-05-01T23:59:59Z"
MIN_DURATION_DAYS = 5.0
MAX_DURATION_DAYS = 60.0
PRICE_FIDELITY_MINUTES = 1
PRICE_CHUNK_SECONDS = 10 * 86400
INSERT_BATCH_SIZE = 5_000

TARGET_TAG_SLUGS = [
    "equities",
    "earnings",
    "kpis",
    "economy",
    "macro-indicators",
    "business",
    "monthly",
    "hit-price",
    "finance-updown",
    "pyth-finance",
    "stocks",
    "geopolitics",
    "oil",
    "iran",
    "us-x-iran",
    "strait-of-hormuz",
    "ai",
    "big-tech",
    "tech",
    "privates",
]

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
    competitive         BOOLEAN,
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

CREATE TABLE IF NOT EXISTS polymarket.markets (
    market_id           TEXT PRIMARY KEY,
    event_id            TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    condition_id        TEXT,
    slug                TEXT,
    question            TEXT,
    group_item_title    TEXT,
    group_item_threshold TEXT,
    market_type         TEXT,
    format_type         TEXT,
    outcomes            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    outcome_prices      DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[]::DOUBLE PRECISION[],
    yes_token_id        TEXT,
    no_token_id         TEXT,
    resolved_outcome    TEXT,
    did_happen          BOOLEAN,
    resolution_status   TEXT NOT NULL DEFAULT 'unknown',
    active              BOOLEAN,
    closed              BOOLEAN,
    accepting_orders    BOOLEAN,
    start_at            TIMESTAMPTZ,
    end_at              TIMESTAMPTZ,
    closed_at           TIMESTAMPTZ,
    volume              DOUBLE PRECISION,
    volume_24hr         DOUBLE PRECISION,
    volume_1wk          DOUBLE PRECISION,
    liquidity_amm       DOUBLE PRECISION,
    liquidity_clob      DOUBLE PRECISION,
    best_bid            DOUBLE PRECISION,
    best_ask            DOUBLE PRECISION,
    spread              DOUBLE PRECISION,
    last_trade_price    DOUBLE PRECISION,
    raw_market          JSONB NOT NULL,
    raw_market_detail   JSONB NOT NULL,
    first_ingested_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_polymarket_markets_event_id
    ON polymarket.markets(event_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_yes_token
    ON polymarket.markets(yes_token_id);
CREATE INDEX IF NOT EXISTS idx_polymarket_markets_resolution
    ON polymarket.markets(did_happen, resolution_status);

CREATE TABLE IF NOT EXISTS polymarket.market_probability_1m (
    market_id                 TEXT NOT NULL REFERENCES polymarket.markets(market_id) ON DELETE CASCADE,
    event_id                  TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    yes_token_id              TEXT NOT NULL,
    ts                        TIMESTAMPTZ NOT NULL,
    available_at              TIMESTAMPTZ NOT NULL,
    yes_probability           DOUBLE PRECISION NOT NULL CHECK (yes_probability >= 0 AND yes_probability <= 1),
    source_fidelity_minutes   INTEGER NOT NULL DEFAULT 1,
    ingested_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
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
    market_id           TEXT PRIMARY KEY REFERENCES polymarket.markets(market_id) ON DELETE CASCADE,
    event_id            TEXT NOT NULL REFERENCES polymarket.events(event_id) ON DELETE CASCADE,
    last_run_id         UUID REFERENCES polymarket.ingestion_runs(run_id),
    status              TEXT NOT NULL,
    history_start       TIMESTAMPTZ,
    history_end         TIMESTAMPTZ,
    row_count           BIGINT NOT NULL DEFAULT 0,
    last_error          TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


@dataclass
class EventRecord:
    event: dict[str, Any]
    matched_tags: set[str]


@dataclass
class MarketRecord:
    event_id: str
    market_id: str
    condition_id: str | None
    yes_token_id: str | None
    no_token_id: str | None
    question: str | None
    history_start: datetime | None
    history_end: datetime | None
    raw_market: dict[str, Any]
    market_detail: dict[str, Any]


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_text_list(value: Any) -> list[str]:
    parsed = parse_jsonish(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed if item is not None]
    if parsed is None:
        return []
    return [str(parsed)]


def as_float_list(value: Any) -> list[float]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, list):
        return []
    out: list[float] = []
    for item in parsed:
        number = as_float(item)
        if number is not None:
            out.append(number)
    return out


def duration_days(event: dict[str, Any]) -> float | None:
    created = parse_dt(event.get("createdAt"))
    end = parse_dt(event.get("endDate"))
    if not created or not end:
        return None
    return (end - created).total_seconds() / 86400.0


def extract_event_tags(event: dict[str, Any]) -> list[str]:
    tags = event.get("tags") or []
    if not isinstance(tags, list):
        return []
    out: list[str] = []
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        slug = tag.get("slug") or tag.get("label") or tag.get("name")
        if slug:
            out.append(str(slug))
    return out


def infer_resolution(outcomes: list[str], outcome_prices: list[float], closed: bool) -> tuple[str | None, bool | None, str]:
    if not closed:
        return None, None, "open"
    if not outcomes or not outcome_prices or len(outcomes) != len(outcome_prices):
        return None, None, "closed_unpriced"

    best_idx = max(range(len(outcome_prices)), key=outcome_prices.__getitem__)
    best_outcome = outcomes[best_idx]
    best_price = outcome_prices[best_idx]
    if best_price < 0.95:
        return best_outcome, None, "closed_ambiguous"

    lower_map = {outcome.lower(): i for i, outcome in enumerate(outcomes)}
    if "yes" in lower_map:
        yes_price = outcome_prices[lower_map["yes"]]
        no_price = outcome_prices[lower_map["no"]] if "no" in lower_map else None
        if yes_price >= 0.95:
            return "Yes", True, "resolved"
        if no_price is not None and no_price >= 0.95:
            return "No", False, "resolved"
    return best_outcome, None, "resolved_non_binary"


def extract_yes_no_tokens(outcomes: list[str], token_ids: list[str]) -> tuple[str | None, str | None]:
    if len(outcomes) != len(token_ids):
        return None, None
    pairs = {outcome.lower(): token_id for outcome, token_id in zip(outcomes, token_ids)}
    return pairs.get("yes"), pairs.get("no")


def build_db_kwargs() -> dict[str, Any]:
    load_dotenv(REPO_ROOT / ".env")
    if os.environ.get("DB_CONNECTION_STRING"):
        return {"dsn": os.environ["DB_CONNECTION_STRING"]}
    required = ["DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing DB environment variables: {', '.join(missing)}")
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ.get("DB_PORT", "5432")),
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


def jsonb(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


class PolymarketIngestor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data_start = parse_dt(args.data_start)
        self.data_end = parse_dt(args.data_end)
        if self.data_start is None or self.data_end is None:
            raise ValueError("data_start and data_end must be ISO timestamps")
        self.run_id = uuid.uuid4()
        self.client = httpx.Client(
            timeout=30,
            headers={"User-Agent": "my_traders-polymarket-db-ingest/1.0"},
        )
        self.events_discovered = 0
        self.markets_seen = 0
        self.markets_ingested = 0
        self.probability_rows = 0

    def close(self) -> None:
        self.client.close()

    async def connect(self) -> asyncpg.Connection:
        return await asyncpg.connect(**build_db_kwargs())

    async def init_schema(self, conn: asyncpg.Connection) -> None:
        await conn.execute(SCHEMA_SQL)

    async def create_run(self, conn: asyncpg.Connection) -> None:
        await conn.execute(
            """
            INSERT INTO polymarket.ingestion_runs (
                run_id, data_start, data_end, min_duration_days, max_duration_days, target_tag_slugs
            )
            VALUES ($1, $2, $3, $4, $5, $6)
            """,
            self.run_id,
            self.data_start,
            self.data_end,
            MIN_DURATION_DAYS,
            MAX_DURATION_DAYS,
            TARGET_TAG_SLUGS,
        )

    async def finish_run(self, conn: asyncpg.Connection, status: str, error: str | None = None) -> None:
        await conn.execute(
            """
            UPDATE polymarket.ingestion_runs
            SET finished_at = NOW(),
                status = $2,
                events_discovered = $3,
                markets_seen = $4,
                markets_ingested = $5,
                probability_rows = $6,
                error = $7
            WHERE run_id = $1
            """,
            self.run_id,
            status,
            self.events_discovered,
            self.markets_seen,
            self.markets_ingested,
            self.probability_rows,
            error,
        )

    def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        for attempt in range(1, self.args.retries + 1):
            try:
                response = self.client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                print(f"    HTTP {response.status_code}: {response.text[:180]}")
            except Exception as exc:
                print(f"    request failed attempt={attempt}: {exc}")
            time.sleep(min(2.0 * attempt, 8.0))
        return None

    def eligible_event(self, event: dict[str, Any]) -> bool:
        created = parse_dt(event.get("createdAt"))
        end = parse_dt(event.get("endDate"))
        dur = duration_days(event)
        if created is None or end is None or dur is None:
            return False
        if created < self.data_start or end > self.data_end:
            return False
        return MIN_DURATION_DAYS <= dur <= MAX_DURATION_DAYS

    def discover_events(self) -> list[EventRecord]:
        records: dict[str, EventRecord] = {}
        for tag_slug in TARGET_TAG_SLUGS:
            print(f"\n[discover] tag={tag_slug}")
            cursor = None
            page = 0
            while True:
                params = {
                    "limit": 500,
                    "tag_slug": tag_slug,
                    "start_date_min": self.args.data_start,
                    "order": "createdAt",
                    "ascending": "false",
                }
                if cursor:
                    params["after_cursor"] = cursor
                body = self.get_json(GAMMA_EVENTS_KEYSET_API, params)
                if not isinstance(body, dict):
                    break
                batch = body.get("events") or []
                cursor = body.get("next_cursor")
                page += 1
                kept = 0
                for event in batch:
                    event_id = str(event.get("id") or event.get("event_id") or "")
                    if not event_id or not self.eligible_event(event):
                        continue
                    if event_id not in records:
                        records[event_id] = EventRecord(event=event, matched_tags=set())
                    records[event_id].matched_tags.add(tag_slug)
                    kept += 1
                print(f"  page={page} fetched={len(batch)} kept={kept} total_unique={len(records)}")
                if self.args.max_pages and page >= self.args.max_pages:
                    break
                if not cursor or not batch:
                    break
                time.sleep(self.args.sleep_seconds)

        ordered = sorted(
            records.values(),
            key=lambda record: parse_dt(record.event.get("createdAt")) or self.data_start,
        )
        if self.args.max_events:
            ordered = ordered[: self.args.max_events]
        self.events_discovered = len(ordered)
        return ordered

    async def upsert_event(self, conn: asyncpg.Connection, record: EventRecord) -> None:
        event = record.event
        event_id = str(event.get("id") or event.get("event_id"))
        dur = duration_days(event)
        tags = extract_event_tags(event)
        matched_tags = sorted(record.matched_tags)
        await conn.execute(
            """
            INSERT INTO polymarket.events (
                event_id, slug, title, description, category, tags, matched_target_tags,
                created_at, start_at, end_at, closed_at, duration_days,
                active, closed, archived, competitive,
                volume, volume_24hr, volume_1wk, volume_1mo, volume_1yr,
                open_interest, liquidity_amm, liquidity_clob, raw_event, updated_at
            )
            VALUES (
                $1, $2, $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12,
                $13, $14, $15, $16,
                $17, $18, $19, $20, $21,
                $22, $23, $24, $25::jsonb, NOW()
            )
            ON CONFLICT (event_id) DO UPDATE SET
                slug = EXCLUDED.slug,
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                category = EXCLUDED.category,
                tags = EXCLUDED.tags,
                matched_target_tags = EXCLUDED.matched_target_tags,
                created_at = EXCLUDED.created_at,
                start_at = EXCLUDED.start_at,
                end_at = EXCLUDED.end_at,
                closed_at = EXCLUDED.closed_at,
                duration_days = EXCLUDED.duration_days,
                active = EXCLUDED.active,
                closed = EXCLUDED.closed,
                archived = EXCLUDED.archived,
                competitive = EXCLUDED.competitive,
                volume = EXCLUDED.volume,
                volume_24hr = EXCLUDED.volume_24hr,
                volume_1wk = EXCLUDED.volume_1wk,
                volume_1mo = EXCLUDED.volume_1mo,
                volume_1yr = EXCLUDED.volume_1yr,
                open_interest = EXCLUDED.open_interest,
                liquidity_amm = EXCLUDED.liquidity_amm,
                liquidity_clob = EXCLUDED.liquidity_clob,
                raw_event = EXCLUDED.raw_event,
                updated_at = NOW()
            """,
            event_id,
            event.get("slug"),
            event.get("title"),
            event.get("description"),
            event.get("category"),
            tags,
            matched_tags,
            parse_dt(event.get("createdAt")),
            parse_dt(event.get("startDate")),
            parse_dt(event.get("endDate")),
            parse_dt(event.get("closedTime")),
            dur,
            event.get("active"),
            event.get("closed"),
            event.get("archived"),
            event.get("competitive"),
            as_float(event.get("volume")),
            as_float(event.get("volume24hr")),
            as_float(event.get("volume1wk")),
            as_float(event.get("volume1mo")),
            as_float(event.get("volume1yr")),
            as_float(event.get("openInterest")),
            as_float(event.get("liquidityAmm")),
            as_float(event.get("liquidityClob")),
            jsonb(event),
        )
        await conn.executemany(
            """
            INSERT INTO polymarket.event_target_tags (event_id, tag_slug)
            VALUES ($1, $2)
            ON CONFLICT (event_id, tag_slug) DO NOTHING
            """,
            [(event_id, tag_slug) for tag_slug in matched_tags],
        )

    def fetch_market_detail(self, market_id: str) -> dict[str, Any]:
        detail = self.get_json(f"{GAMMA_MARKET_API}/{market_id}")
        return detail if isinstance(detail, dict) else {}

    def build_market_record(self, event: dict[str, Any], raw_market: dict[str, Any]) -> MarketRecord | None:
        event_id = str(event.get("id") or event.get("event_id"))
        market_id = str(raw_market.get("id") or raw_market.get("market_id") or "")
        if not market_id:
            return None

        detail = self.fetch_market_detail(market_id)
        outcomes = as_text_list(detail.get("outcomes") or raw_market.get("outcomes"))
        token_ids = as_text_list(detail.get("clobTokenIds") or raw_market.get("clobTokenIds"))
        yes_token, no_token = extract_yes_no_tokens(outcomes, token_ids)
        if not yes_token:
            return None

        starts = [
            parse_dt(detail.get("createdAt")),
            parse_dt(detail.get("startDateIso")),
            parse_dt(raw_market.get("createdAt")),
            parse_dt(raw_market.get("startDate")),
            parse_dt(event.get("createdAt")),
        ]
        ends = [
            parse_dt(detail.get("closedTime")),
            parse_dt(raw_market.get("closedTime")),
            parse_dt(detail.get("endDateIso")),
            parse_dt(raw_market.get("endDate")),
            parse_dt(event.get("endDate")),
            self.data_end,
        ]
        start_candidates = [value for value in starts if value is not None]
        end_candidates = [value for value in ends if value is not None]
        history_start = max([self.data_start, *start_candidates]) if start_candidates else self.data_start
        history_end = min([self.data_end, *end_candidates]) if end_candidates else self.data_end

        return MarketRecord(
            event_id=event_id,
            market_id=market_id,
            condition_id=detail.get("conditionId") or raw_market.get("conditionId"),
            yes_token_id=yes_token,
            no_token_id=no_token,
            question=detail.get("question") or raw_market.get("question"),
            history_start=history_start,
            history_end=history_end,
            raw_market=raw_market,
            market_detail=detail,
        )

    async def upsert_market(self, conn: asyncpg.Connection, market: MarketRecord) -> None:
        detail = market.market_detail
        raw = market.raw_market
        outcomes = as_text_list(detail.get("outcomes") or raw.get("outcomes"))
        outcome_prices = as_float_list(detail.get("outcomePrices") or raw.get("outcomePrices"))
        closed = bool(detail.get("closed") or raw.get("closed"))
        resolved_outcome, did_happen, resolution_status = infer_resolution(outcomes, outcome_prices, closed)

        await conn.execute(
            """
            INSERT INTO polymarket.markets (
                market_id, event_id, condition_id, slug, question,
                group_item_title, group_item_threshold, market_type, format_type,
                outcomes, outcome_prices, yes_token_id, no_token_id,
                resolved_outcome, did_happen, resolution_status,
                active, closed, accepting_orders,
                start_at, end_at, closed_at,
                volume, volume_24hr, volume_1wk, liquidity_amm, liquidity_clob,
                best_bid, best_ask, spread, last_trade_price,
                raw_market, raw_market_detail, updated_at
            )
            VALUES (
                $1, $2, $3, $4, $5,
                $6, $7, $8, $9,
                $10, $11, $12, $13,
                $14, $15, $16,
                $17, $18, $19,
                $20, $21, $22,
                $23, $24, $25, $26, $27,
                $28, $29, $30, $31,
                $32::jsonb, $33::jsonb, NOW()
            )
            ON CONFLICT (market_id) DO UPDATE SET
                event_id = EXCLUDED.event_id,
                condition_id = EXCLUDED.condition_id,
                slug = EXCLUDED.slug,
                question = EXCLUDED.question,
                group_item_title = EXCLUDED.group_item_title,
                group_item_threshold = EXCLUDED.group_item_threshold,
                market_type = EXCLUDED.market_type,
                format_type = EXCLUDED.format_type,
                outcomes = EXCLUDED.outcomes,
                outcome_prices = EXCLUDED.outcome_prices,
                yes_token_id = EXCLUDED.yes_token_id,
                no_token_id = EXCLUDED.no_token_id,
                resolved_outcome = EXCLUDED.resolved_outcome,
                did_happen = EXCLUDED.did_happen,
                resolution_status = EXCLUDED.resolution_status,
                active = EXCLUDED.active,
                closed = EXCLUDED.closed,
                accepting_orders = EXCLUDED.accepting_orders,
                start_at = EXCLUDED.start_at,
                end_at = EXCLUDED.end_at,
                closed_at = EXCLUDED.closed_at,
                volume = EXCLUDED.volume,
                volume_24hr = EXCLUDED.volume_24hr,
                volume_1wk = EXCLUDED.volume_1wk,
                liquidity_amm = EXCLUDED.liquidity_amm,
                liquidity_clob = EXCLUDED.liquidity_clob,
                best_bid = EXCLUDED.best_bid,
                best_ask = EXCLUDED.best_ask,
                spread = EXCLUDED.spread,
                last_trade_price = EXCLUDED.last_trade_price,
                raw_market = EXCLUDED.raw_market,
                raw_market_detail = EXCLUDED.raw_market_detail,
                updated_at = NOW()
            """,
            market.market_id,
            market.event_id,
            market.condition_id,
            detail.get("slug") or raw.get("slug"),
            market.question,
            detail.get("groupItemTitle") or raw.get("groupItemTitle"),
            detail.get("groupItemThreshold") or raw.get("groupItemThreshold"),
            detail.get("marketType") or raw.get("marketType"),
            detail.get("formatType") or raw.get("formatType"),
            outcomes,
            outcome_prices,
            market.yes_token_id,
            market.no_token_id,
            resolved_outcome,
            did_happen,
            resolution_status,
            detail.get("active") if detail.get("active") is not None else raw.get("active"),
            closed,
            detail.get("acceptingOrders"),
            parse_dt(detail.get("startDateIso") or raw.get("startDate")),
            parse_dt(detail.get("endDateIso") or raw.get("endDate")),
            parse_dt(detail.get("closedTime") or raw.get("closedTime")),
            as_float(detail.get("volumeNum") or raw.get("volumeNum") or raw.get("volume")),
            as_float(detail.get("volume24hr") or raw.get("volume24hr")),
            as_float(detail.get("volume1wk") or raw.get("volume1wk")),
            as_float(detail.get("liquidityAmm") or raw.get("liquidityAmm")),
            as_float(detail.get("liquidityClob") or raw.get("liquidityClob")),
            as_float(detail.get("bestBid") or raw.get("bestBid")),
            as_float(detail.get("bestAsk") or raw.get("bestAsk")),
            as_float(detail.get("spread") or raw.get("spread")),
            as_float(detail.get("lastTradePrice") or raw.get("lastTradePrice")),
            jsonb(raw),
            jsonb(detail),
        )

    async def mark_market_state(
        self,
        conn: asyncpg.Connection,
        market: MarketRecord,
        status: str,
        row_count: int = 0,
        error: str | None = None,
    ) -> None:
        await conn.execute(
            """
            INSERT INTO polymarket.market_ingestion_state (
                market_id, event_id, last_run_id, status, history_start, history_end, row_count, last_error, updated_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
            ON CONFLICT (market_id) DO UPDATE SET
                last_run_id = EXCLUDED.last_run_id,
                status = EXCLUDED.status,
                history_start = EXCLUDED.history_start,
                history_end = EXCLUDED.history_end,
                row_count = EXCLUDED.row_count,
                last_error = EXCLUDED.last_error,
                updated_at = NOW()
            """,
            market.market_id,
            market.event_id,
            self.run_id,
            status,
            market.history_start,
            market.history_end,
            row_count,
            error,
        )

    async def market_already_complete(self, conn: asyncpg.Connection, market_id: str) -> bool:
        if self.args.force:
            return False
        status = await conn.fetchval(
            "SELECT status FROM polymarket.market_ingestion_state WHERE market_id = $1",
            market_id,
        )
        return status == "complete"

    def fetch_probability_history(self, market: MarketRecord) -> list[tuple[datetime, float]]:
        if not market.yes_token_id or market.history_start is None or market.history_end is None:
            return []
        if market.history_start >= market.history_end:
            return []

        start_ts = int(market.history_start.timestamp())
        end_ts = int(market.history_end.timestamp())
        rows: list[tuple[datetime, float]] = []
        cursor = start_ts
        while cursor <= end_ts:
            chunk_end = min(cursor + PRICE_CHUNK_SECONDS - 1, end_ts)
            payload = self.get_json(
                CLOB_PRICES_HISTORY_API,
                {
                    "market": market.yes_token_id,
                    "startTs": cursor,
                    "endTs": chunk_end,
                    "fidelity": PRICE_FIDELITY_MINUTES,
                },
            )
            if isinstance(payload, dict):
                for item in payload.get("history") or []:
                    timestamp_raw = item.get("t")
                    probability_raw = item.get("p")
                    try:
                        ts = datetime.fromtimestamp(float(timestamp_raw), tz=timezone.utc)
                        probability = float(probability_raw)
                    except (TypeError, ValueError, OSError):
                        continue
                    if 0.0 <= probability <= 1.0:
                        rows.append((ts, probability))
            cursor = chunk_end + 1
            time.sleep(self.args.sleep_seconds)

        rows = sorted(set(rows), key=lambda row: row[0])
        return rows

    async def insert_probability_rows(
        self,
        conn: asyncpg.Connection,
        market: MarketRecord,
        rows: list[tuple[datetime, float]],
    ) -> int:
        if not rows or not market.yes_token_id:
            return 0
        total_inserted = 0
        for start in range(0, len(rows), INSERT_BATCH_SIZE):
            batch = rows[start : start + INSERT_BATCH_SIZE]
            result = await conn.executemany(
                """
                INSERT INTO polymarket.market_probability_1m (
                    market_id, event_id, yes_token_id, ts, available_at, yes_probability, source_fidelity_minutes
                )
                VALUES ($1, $2, $3, $4, $4, $5, $6)
                ON CONFLICT (market_id, ts) DO UPDATE SET
                    yes_token_id = EXCLUDED.yes_token_id,
                    event_id = EXCLUDED.event_id,
                    available_at = EXCLUDED.available_at,
                    yes_probability = EXCLUDED.yes_probability,
                    source_fidelity_minutes = EXCLUDED.source_fidelity_minutes,
                    ingested_at = NOW()
                """,
                [
                    (
                        market.market_id,
                        market.event_id,
                        market.yes_token_id,
                        ts,
                        probability,
                        PRICE_FIDELITY_MINUTES,
                    )
                    for ts, probability in batch
                ],
            )
            # asyncpg executemany returns None on older versions; count attempted rows.
            total_inserted += len(batch)
        return total_inserted

    async def run(self) -> None:
        conn = await self.connect()
        try:
            await self.init_schema(conn)
            await self.create_run(conn)
            events = self.discover_events()
            print(f"\n[discover] eligible unique events={len(events)}")

            for event_index, record in enumerate(events, start=1):
                event = record.event
                event_id = str(event.get("id") or event.get("event_id"))
                markets_raw = event.get("markets") or []
                print(f"\n[event {event_index}/{len(events)}] {event_id} | markets={len(markets_raw)} | {event.get('title')}")
                await self.upsert_event(conn, record)

                for raw_market in markets_raw:
                    if self.args.max_markets and self.markets_seen >= self.args.max_markets:
                        print("[limit] max_markets reached")
                        await self.finish_run(conn, "partial_limit")
                        return

                    market = self.build_market_record(event, raw_market)
                    if market is None:
                        continue
                    self.markets_seen += 1
                    await self.upsert_market(conn, market)

                    if await self.market_already_complete(conn, market.market_id):
                        print(f"  [skip complete] market={market.market_id}")
                        continue

                    await self.mark_market_state(conn, market, "running")
                    try:
                        rows = self.fetch_probability_history(market)
                        inserted = await self.insert_probability_rows(conn, market, rows)
                        self.probability_rows += inserted
                        self.markets_ingested += 1
                        await self.mark_market_state(conn, market, "complete", row_count=inserted)
                        print(
                            f"  [market] {market.market_id} rows={inserted} "
                            f"range={market.history_start} -> {market.history_end}"
                        )
                    except Exception as exc:
                        await self.mark_market_state(conn, market, "error", error=repr(exc))
                        print(f"  [error] market={market.market_id}: {exc}")

            await self.finish_run(conn, "complete")
        except Exception as exc:
            try:
                await self.finish_run(conn, "error", repr(exc))
            except Exception:
                pass
            raise
        finally:
            await conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Polymarket 1-minute YES probability data into Postgres.")
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-events", type=int, default=0, help="Optional development limit.")
    parser.add_argument("--max-markets", type=int, default=0, help="Optional development limit.")
    parser.add_argument("--max-pages", type=int, default=0, help="Optional discovery page limit per tag.")
    parser.add_argument("--force", action="store_true", help="Re-fetch markets marked complete.")
    return parser


async def main_async() -> None:
    args = build_parser().parse_args()
    ingestor = PolymarketIngestor(args)
    try:
        await ingestor.run()
    except OSError as exc:
        print(f"DB connection failed before ingestion started: {exc}")
        print("No CSV/JSON fallback was written; fix the Postgres connection and rerun this script.")
        raise SystemExit(2) from exc
    finally:
        ingestor.close()


if __name__ == "__main__":
    asyncio.run(main_async())
