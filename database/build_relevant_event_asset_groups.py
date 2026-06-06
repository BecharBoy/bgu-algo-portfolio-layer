from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

import asyncpg
import httpx
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.db_connection import connect

ENV_PATH = REPO_ROOT / ".env"
SCHEMA_NAME = "checking_relevant_events"
PROMPT_VERSION = "asset-universe-v2"
EVENT_LIMIT = 5
CANDIDATE_POOL_LIMIT = 250
MIN_ASSETS_PER_EVENT = 4
TARGET_ASSETS_PER_EVENT = 8
MAX_ASSETS_PER_EVENT = 15

EXCLUDED_EVENT_TAGS = {
    "bitcoin",
    "crypto",
    "crypto-prices",
    "daily",
    "daily-close",
    "ethereum",
    "finance-updown",
    "hit-price",
    "multi-strikes",
    "pyth-finance",
    "recurring",
    "ripple",
    "solana",
    "stock-prices",
    "today",
    "up-or-down",
    "weekly",
    "xrp",
}

MACRO_EVENT_TAGS = {
    "economy",
    "economic-policy",
    "fed",
    "fed-rates",
    "gdp",
    "global-rates",
    "housing",
    "inflation",
    "jobs",
    "macro-indicators",
    "nfp",
    "nonfarm-payroll",
    "real-estate",
    "trade-war",
    "unemployment",
}

GEOPOLITICAL_EVENT_TAGS = {
    "china",
    "diplomacy-ceasefire",
    "foreign-policy",
    "geopolitics",
    "iran",
    "israel",
    "middle-east",
    "military-action",
    "oil",
    "politics",
    "russia",
    "strait-of-hormuz",
    "ukraine",
    "world",
}

COMPANY_EVENT_TAGS = {
    "ai",
    "big-tech",
    "business",
    "earnings",
    "equities",
    "fda",
    "finance",
    "ipo",
    "ipos",
    "stocks",
    "tech",
}

EVENT_CATEGORY_ORDER = ("macro", "geopolitics", "company")
INCLUDED_EVENT_TAGS = MACRO_EVENT_TAGS | GEOPOLITICAL_EVENT_TAGS | COMPANY_EVENT_TAGS


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class AssetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20)
    asset_name: str = Field(min_length=1, max_length=120)
    asset_class: Literal["stock", "etf"]
    reason: str = Field(min_length=20, max_length=500)

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.upper()


class EventAssetUniverse(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    universe_name: str = Field(min_length=1, max_length=200)
    universe_reason: str = Field(min_length=20, max_length=700)
    assets: list[AssetCandidate] = Field(
        min_length=MIN_ASSETS_PER_EVENT,
        max_length=MAX_ASSETS_PER_EVENT,
    )

    @model_validator(mode="after")
    def require_unique_symbols(self) -> EventAssetUniverse:
        symbols = [asset.symbol for asset in self.assets]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Asset universe contains duplicate symbols")
        return self


SYSTEM_PROMPT = f"""
You build a stock-market research universe for one prediction-market question.
The universe will be examined and backtested later. It is not a prediction or a trade.

CRITICAL RULES TO PREVENT HALLUCINATIONS:
1. THE EVENT TITLE IS YOUR ABSOLUTE SOURCE OF TRUTH. Ignore generic tags (like 'economy' or 'macro') if the event clearly asks about a specific company (e.g., Robinhood, Apple).
2. If the event is about a specific company, you MUST include that company's ticker first. Then include its direct competitors, major partners/suppliers, and specific sector ETFs.
3. DO NOT add generic mega-cap stocks (like AAPL, MSFT, SPY) just to fill space, unless they have a direct, specific economic link to the exact event question.
4. Return {MIN_ASSETS_PER_EVENT}-{MAX_ASSETS_PER_EVENT} unique assets. It is much better to return fewer highly relevant assets than to invent reasons for unrelated ones.
5. Your only job is to identify the "world" of US-listed stocks and ETFs worth researching for the supplied question. Do not predict Yes or No.
6. Return for each event both individual equities and sectors when relevant.
7. Every asset needs a specific reason explaining why it belongs in the research universe. Explain the direct economic link to the specific event question.
8. Omit any asset whose ticker or company identity you are not confident is real.

Return exactly the supplied JSON schema and no commentary outside the JSON.
""".strip()


CREATE_REVIEW_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.event_asset_llm_review (
    event_id       TEXT NOT NULL
        REFERENCES {SCHEMA_NAME}.source_events(event_id) ON DELETE CASCADE,
    model_name     TEXT NOT NULL,
    event_title    TEXT NOT NULL,
    llm_input      JSONB NOT NULL,
    llm_output     JSONB NOT NULL,
    human_valid    BOOLEAN,
    human_notes    TEXT,
    analyzed_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (event_id, model_name)
);
"""


def json_text(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, indent=indent)


def event_category(tags: list[str]) -> str:
    normalized_tags = {tag.lower() for tag in tags}
    if normalized_tags & COMPANY_EVENT_TAGS:
        return "company"
    if normalized_tags & GEOPOLITICAL_EVENT_TAGS:
        return "geopolitics"
    if normalized_tags & MACRO_EVENT_TAGS:
        return "macro"

    return "uncategorized"


def event_archetype(tags: list[str]) -> str:
    normalized_tags = {tag.lower() for tag in tags}
    if normalized_tags & {"nfp", "nonfarm-payroll", "jobs", "unemployment"}:
        return "macro_labor"
    if normalized_tags & {"fed", "fed-rates", "global-rates"}:
        return "macro_rates"
    if normalized_tags & {"inflation"}:
        return "macro_inflation"
    if normalized_tags & {"gdp"}:
        return "macro_growth"
    if normalized_tags & {"housing", "real-estate"}:
        return "macro_housing"
    if normalized_tags & {"strait-of-hormuz", "oil"}:
        return "geopolitics_energy"
    if normalized_tags & {"ukraine", "russia"}:
        return "geopolitics_europe"
    if normalized_tags & {"iran", "israel", "middle-east"}:
        return "geopolitics_middle_east"
    if normalized_tags & {"earnings"}:
        return "company_earnings"
    if normalized_tags & {"fda"}:
        return "company_fda"
    if normalized_tags & {"ipo", "ipos"}:
        return "company_ipo"
    return event_category(tags)


def select_balanced_events(events: list[asyncpg.Record]) -> list[asyncpg.Record]:
    buckets: dict[str, list[asyncpg.Record]] = {
        category: [] for category in EVENT_CATEGORY_ORDER
    }
    for event in events:
        buckets[event_category(list(event["tags"]))].append(event)

    selected: list[asyncpg.Record] = []
    seen_archetypes: set[str] = set()
    while len(selected) < EVENT_LIMIT:
        added_event = False
        for category in EVENT_CATEGORY_ORDER:
            if buckets[category]:
                selected_index = next(
                    (
                        index
                        for index, event in enumerate(buckets[category])
                        if event_archetype(list(event["tags"])) not in seen_archetypes
                    ),
                    0,
                )
                event = buckets[category].pop(selected_index)
                selected.append(event)
                seen_archetypes.add(event_archetype(list(event["tags"])))
                added_event = True
                if len(selected) == EVENT_LIMIT:
                    break
        if not added_event:
            break
    return selected


async def latest_research_events(conn: asyncpg.Connection) -> list[asyncpg.Record]:
    candidate_events = await conn.fetch(
        f"""
        SELECT
            e.event_id,
            e.title,
            e.created_at,
            e.end_at,
            e.tags,
            e.matched_tags,
            (ARRAY_AGG(q.market_id ORDER BY q.created_at, q.market_id))[1]
                AS representative_market_id,
            (ARRAY_AGG(q.question ORDER BY q.created_at, q.market_id))[1]
                AS representative_question,
            COUNT(*) AS related_market_count
        FROM {SCHEMA_NAME}.source_events e
        JOIN {SCHEMA_NAME}.source_questions q ON q.event_id = e.event_id
        WHERE NOT (e.tags && $1::TEXT[])
          AND e.tags && $2::TEXT[]
          AND e.end_at IS NOT NULL
          AND e.end_at < NOW()
          AND e.end_at >= '2026-04-01T00:00:00+00:00'::timestamptz
          AND e.end_at < '2026-05-01T00:00:00+00:00'::timestamptz
          AND e.created_at IS NOT NULL
          AND (e.end_at - e.created_at) > INTERVAL '3 days'
        GROUP BY
            e.event_id,
            e.title,
            e.created_at,
            e.end_at,
            e.tags,
            e.matched_tags
        ORDER BY e.created_at DESC NULLS LAST, e.event_id DESC
        LIMIT $3
        """,
        sorted(EXCLUDED_EVENT_TAGS),
        sorted(INCLUDED_EVENT_TAGS),
        CANDIDATE_POOL_LIMIT,
    )
    return select_balanced_events(list(candidate_events))


def build_llm_input(event: asyncpg.Record) -> dict[str, Any]:
    return {
        "prompt_version": PROMPT_VERSION,
        "event_id": event["event_id"],
        "event_category": event_category(list(event["tags"])),
        "event_title": event["title"],
        "market_id": event["representative_market_id"],
        "market_question": event["representative_question"],
        "created_at": event["created_at"],
        "end_at": event["end_at"],
        "tags": event["tags"],
        "matched_tags": event["matched_tags"],
        "related_market_count": event["related_market_count"],
    }


async def ask_llama(
    ollama: httpx.AsyncClient,
    model_name: str,
    llm_input: dict[str, Any],
) -> EventAssetUniverse:
    response = await ollama.post(
        "/api/chat",
        json={
            "model": model_name,
            "stream": False,
            "format": EventAssetUniverse.model_json_schema(),
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json_text(llm_input)},
            ],
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "seed": 42,
                "num_predict": 2500,
            },
        },
    )
    response.raise_for_status()

    response_body = response.json()
    content = response_body.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response is missing message.content")
    return EventAssetUniverse.model_validate_json(content)


async def save_review(
    conn: asyncpg.Connection,
    event: asyncpg.Record,
    model_name: str,
    llm_input: dict[str, Any],
    llm_output: EventAssetUniverse,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA_NAME}.event_asset_llm_review (
            event_id,
            model_name,
            event_title,
            llm_input,
            llm_output,
            analyzed_at
        )
        VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, NOW())
        ON CONFLICT (event_id, model_name) DO UPDATE SET
            event_title = EXCLUDED.event_title,
            llm_input = EXCLUDED.llm_input,
            llm_output = EXCLUDED.llm_output,
            human_valid = NULL,
            human_notes = NULL,
            analyzed_at = NOW()
        """,
        event["event_id"],
        model_name,
        event["title"],
        json_text(llm_input),
        llm_output.model_dump_json(),
    )


def print_review(
    index: int,
    total: int,
    event: asyncpg.Record,
    llm_input: dict[str, Any],
    llm_output: EventAssetUniverse,
) -> None:
    print("\n" + "=" * 100)
    print(f"EVENT {index}/{total}: {event['title']} (event_id={event['event_id']})")
    print("\nEXACT INPUT SENT TO LLAMA:")
    print(json_text(llm_input, indent=2))
    print("\nSTRUCTURED ASSET UNIVERSE FOR HUMAN REVIEW:")
    print(llm_output.model_dump_json(indent=2))


async def main() -> None:
    load_dotenv(ENV_PATH)
    ollama_host = required_env("OLLAMA_HOST").rstrip("/")
    ollama_model = required_env("OLLAMA_MODEL")

    conn = await connect()
    ollama = httpx.AsyncClient(
        base_url=ollama_host,
        timeout=httpx.Timeout(300),
    )
    try:
        await conn.execute(CREATE_REVIEW_TABLE_SQL)
        events = await latest_research_events(conn)
        if not events:
            raise RuntimeError("No eligible macro, geopolitical, or company events found")

        for index, event in enumerate(events, start=1):
            llm_input = build_llm_input(event)
            llm_output = await ask_llama(ollama, ollama_model, llm_input)
            await save_review(conn, event, ollama_model, llm_input, llm_output)
            print_review(index, len(events), event, llm_input, llm_output)

        print(
            f"\nSaved {len(events)} asset universes to "
            f"{SCHEMA_NAME}.event_asset_llm_review"
        )
    finally:
        await ollama.aclose()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
