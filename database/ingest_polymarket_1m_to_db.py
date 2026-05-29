from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx

from database.db_connection import connect
from database.polymarket_common import (
    DEFAULT_DATA_END,
    DEFAULT_DATA_START,
    INSERT_BATCH_SIZE,
    MAX_DURATION_DAYS,
    MIN_DURATION_DAYS,
    PRICE_CHUNK_SECONDS,
    PRICE_FIDELITY_MINUTES,
    TARGET_TAG_SLUGS,
    EventRecord,
    MarketRecord,
    duration_days,
    float_array,
    parse_dt,
    text_array,
    yes_no_tokens,
)
from database.polymarket_queries import market_is_complete
from database.polymarket_schema import init_polymarket_schema
from database.polymarket_writes import (
    create_ingestion_run,
    finish_ingestion_run,
    insert_probability_rows,
    mark_market_triggered_70,
    mark_market_state,
    set_market_yes_percentage,
    upsert_event,
    upsert_market,
)

sys.stdout.reconfigure(errors="replace")

GAMMA_EVENTS_KEYSET_API = "https://gamma-api.polymarket.com/events/keyset"
GAMMA_MARKET_API = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_HISTORY_API = "https://clob.polymarket.com/prices-history"
YES_TRIGGER_THRESHOLD = 0.70
PROBABILITY_EPSILON = 0.05


def trigger(market_name: str) -> None:
    print(f"[trigger] {market_name}")


def normalize_probability(value: Any, market_id: str) -> float | None:
    probability = float(value)
    if 0.0 <= probability <= 1.0:
        return probability
    if -PROBABILITY_EPSILON <= probability <= 1.0 + PROBABILITY_EPSILON:
        clipped = min(max(probability, 0.0), 1.0)
        print(f"  [probability clipped] market={market_id} raw={probability} clipped={clipped}")
        return clipped
    print(f"  [probability skipped] market={market_id} raw={probability}")
    return None


class PolymarketApi:
    def __init__(self, sleep_seconds: float) -> None:
        self.sleep_seconds = sleep_seconds
        self.client = httpx.AsyncClient(
            timeout=30,
            headers={"User-Agent": "my_traders-polymarket-db-ingest/1.0"},
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        response = await self.client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def sleep(self) -> None:
        await asyncio.sleep(self.sleep_seconds)


def eligible_event(event: dict[str, Any], data_start: datetime, data_end: datetime) -> bool:
    created = parse_dt(event.get("createdAt"))
    end = parse_dt(event.get("endDate"))
    event_duration = duration_days(event)
    if created is None or end is None or event_duration is None:
        return False
    if created < data_start or end > data_end:
        return False
    return MIN_DURATION_DAYS <= event_duration <= MAX_DURATION_DAYS


async def discover_events(
    api: PolymarketApi,
    *,
    data_start_text: str,
    data_start: datetime,
    data_end: datetime,
    max_pages: int,
    max_events: int,
) -> list[EventRecord]:
    records: dict[str, EventRecord] = {}
    for tag_slug in TARGET_TAG_SLUGS:
        print(f"\n[discover] tag={tag_slug}")
        cursor = None
        page = 0
        while True:
            params = {
                "limit": 500,
                "tag_slug": tag_slug,
                "start_date_min": data_start_text,
                "order": "createdAt",
                "ascending": "false",
            }
            if cursor:
                params["after_cursor"] = cursor

            body = await api.get_json(GAMMA_EVENTS_KEYSET_API, params)
            batch = body.get("events") or []
            cursor = body.get("next_cursor")
            page += 1

            kept = 0
            for event in batch:
                event_id = str(event.get("id") or event.get("event_id") or "")
                if not event_id or not eligible_event(event, data_start, data_end):
                    continue
                if event_id not in records:
                    records[event_id] = EventRecord(event=event, matched_tags=set())
                records[event_id].matched_tags.add(tag_slug)
                kept += 1

            print(f"  page={page} fetched={len(batch)} kept={kept} total_unique={len(records)}")
            if max_pages and page >= max_pages:
                break
            if not cursor or not batch:
                break
            await api.sleep()

    ordered = sorted(
        records.values(),
        key=lambda record: parse_dt(record.event.get("createdAt")) or data_start,
    )
    return ordered[:max_events] if max_events else ordered


async def fetch_market_detail(api: PolymarketApi, market_id: str) -> dict[str, Any]:
    detail = await api.get_json(f"{GAMMA_MARKET_API}/{market_id}")
    if not isinstance(detail, dict):
        raise TypeError(f"Gamma market detail for {market_id} is not a dict")
    return detail


async def build_market_record(
    api: PolymarketApi,
    *,
    event: dict[str, Any],
    raw_market: dict[str, Any],
    data_start: datetime,
    data_end: datetime,
) -> MarketRecord | None:
    event_id = str(event.get("id") or event.get("event_id"))
    market_id = str(raw_market.get("id") or raw_market.get("market_id") or "")
    if not market_id:
        raise ValueError(f"Market missing id inside event {event_id}")

    detail = await fetch_market_detail(api, market_id)
    outcomes = text_array(detail.get("outcomes") or raw_market.get("outcomes"))
    outcome_prices = float_array(detail.get("outcomePrices") or raw_market.get("outcomePrices"))
    token_ids = text_array(detail.get("clobTokenIds") or raw_market.get("clobTokenIds"))
    if token_ids:
        yes_token_id, no_token_id = yes_no_tokens(outcomes, token_ids)
    else:
        yes_token_id, no_token_id = None, None

    start_candidates = [
        parse_dt(detail.get("createdAt")),
        parse_dt(detail.get("startDateIso")),
        parse_dt(raw_market.get("createdAt")),
        parse_dt(raw_market.get("startDate")),
        parse_dt(event.get("createdAt")),
    ]
    end_candidates = [
        parse_dt(detail.get("closedTime")),
        parse_dt(raw_market.get("closedTime")),
        parse_dt(detail.get("endDateIso")),
        parse_dt(raw_market.get("endDate")),
        parse_dt(event.get("endDate")),
        data_end,
    ]
    history_start = max([data_start, *[value for value in start_candidates if value is not None]])
    history_end = min([data_end, *[value for value in end_candidates if value is not None]])

    return MarketRecord(
        event_id=event_id,
        market_id=market_id,
        condition_id=detail.get("conditionId") or raw_market.get("conditionId"),
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        question=detail.get("question") or raw_market.get("question"),
        outcomes=outcomes,
        outcome_prices=outcome_prices,
        history_start=history_start,
        history_end=history_end,
        raw_market=raw_market,
        market_detail=detail,
    )


async def fetch_probability_history(
    api: PolymarketApi,
    *,
    market: MarketRecord,
) -> list[tuple[datetime, float]]:
    if market.history_start >= market.history_end:
        return []

    rows: list[tuple[datetime, float]] = []
    cursor = int(market.history_start.timestamp())
    end_ts = int(market.history_end.timestamp())
    while cursor <= end_ts:
        chunk_end = min(cursor + PRICE_CHUNK_SECONDS - 1, end_ts)
        payload = await api.get_json(
            CLOB_PRICES_HISTORY_API,
            {
                "market": market.yes_token_id,
                "startTs": cursor,
                "endTs": chunk_end,
                "fidelity": PRICE_FIDELITY_MINUTES,
            },
        )
        history = payload.get("history") or []
        for item in history:
            ts = datetime.fromtimestamp(float(item["t"]), tz=timezone.utc)
            probability = normalize_probability(item["p"], market.market_id)
            if probability is None:
                continue
            rows.append((ts, probability))
        cursor = chunk_end + 1
        await api.sleep()

    return sorted(set(rows), key=lambda row: row[0])


async def run_ingestion(args: argparse.Namespace) -> None:
    data_start = parse_dt(args.data_start)
    data_end = parse_dt(args.data_end)
    if data_start is None or data_end is None:
        raise ValueError("data-start and data-end must be valid ISO timestamps")

    run_id = uuid.uuid4()
    events_discovered = 0
    markets_seen = 0
    markets_ingested = 0
    probability_rows = 0

    conn = await connect()
    api = PolymarketApi(sleep_seconds=args.sleep_seconds)

    await init_polymarket_schema(conn)
    await create_ingestion_run(
        conn,
        run_id=run_id,
        data_start=data_start,
        data_end=data_end,
        min_duration_days=MIN_DURATION_DAYS,
        max_duration_days=MAX_DURATION_DAYS,
        target_tag_slugs=TARGET_TAG_SLUGS,
    )

    event_records = await discover_events(
        api,
        data_start_text=args.data_start,
        data_start=data_start,
        data_end=data_end,
        max_pages=args.max_pages,
        max_events=args.max_events,
    )
    events_discovered = len(event_records)
    print(f"\n[discover] eligible unique events={events_discovered}")

    for event_index, record in enumerate(event_records, start=1):
        event = record.event
        event_id = str(event.get("id") or event.get("event_id"))
        raw_markets = event.get("markets") or []
        if not isinstance(raw_markets, list):
            raise TypeError(f"event.markets must be a list for event {event_id}")
        print(f"\n[event {event_index}/{events_discovered}] {event_id} | markets={len(raw_markets)} | {event.get('title')}")

        await upsert_event(conn, record)

        for raw_market in raw_markets:
            if args.max_markets and markets_seen >= args.max_markets:
                await finish_ingestion_run(
                    conn,
                    run_id=run_id,
                    status="partial_limit",
                    events_discovered=events_discovered,
                    markets_seen=markets_seen,
                    markets_ingested=markets_ingested,
                    probability_rows=probability_rows,
                )
                print("[limit] max-markets reached")
                await api.close()
                await conn.close()
                return

            if not isinstance(raw_market, dict):
                raise TypeError(f"market payload must be dict inside event {event_id}")

            market = await build_market_record(
                api,
                event=event,
                raw_market=raw_market,
                data_start=data_start,
                data_end=data_end,
            )
            if market is None:
                continue

            markets_seen += 1
            await upsert_market(conn, market)

            if market.yes_token_id is None:
                await mark_market_state(
                    conn,
                    market=market,
                    run_id=run_id,
                    status="no_yes_token",
                    row_count=0,
                )
                print(f"  [market metadata only] {market.market_id} no YES token | {market.question}")
                continue

            if not args.force and await market_is_complete(conn, market.market_id):
                print(f"  [skip complete] market={market.market_id}")
                continue

            await mark_market_state(
                conn,
                market=market,
                run_id=run_id,
                status="running",
                row_count=0,
            )
            rows = await fetch_probability_history(api, market=market)
            inserted = await insert_probability_rows(
                conn,
                market=market,
                rows=rows,
                source_fidelity_minutes=PRICE_FIDELITY_MINUTES,
                batch_size=INSERT_BATCH_SIZE,
            )
            if rows:
                await set_market_yes_percentage(
                    conn,
                    market_id=market.market_id,
                    yes_probability=rows[-1][1],
                )
                if any(probability >= YES_TRIGGER_THRESHOLD for _, probability in rows):
                    if await mark_market_triggered_70(conn, market.market_id):
                        trigger(market.question or market.market_id)
            await mark_market_state(
                conn,
                market=market,
                run_id=run_id,
                status="complete",
                row_count=inserted,
            )
            markets_ingested += 1
            probability_rows += inserted
            print(f"  [market] {market.market_id} rows={inserted} {market.history_start} -> {market.history_end}")

    await finish_ingestion_run(
        conn,
        run_id=run_id,
        status="complete",
        events_discovered=events_discovered,
        markets_seen=markets_seen,
        markets_ingested=markets_ingested,
        probability_rows=probability_rows,
    )
    await api.close()
    await conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest Polymarket 1-minute YES probabilities into Postgres.")
    parser.add_argument("--data-start", default=DEFAULT_DATA_START)
    parser.add_argument("--data-end", default=DEFAULT_DATA_END)
    parser.add_argument("--sleep-seconds", type=float, default=0.12)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--max-markets", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser


async def main_async() -> None:
    await run_ingestion(build_parser().parse_args())


if __name__ == "__main__":
    asyncio.run(main_async())
