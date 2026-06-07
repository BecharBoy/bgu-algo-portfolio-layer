from __future__ import annotations

import csv
from typing import Any
from database.backtesting.repositories import json_value
from database.backtesting.repositories.market_decisions import (
    accepted_market_ids,
    reusable_market_decision,
    save_market_decision,
)
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repository import candidate_events, event_markets
from LLM.remove_unwanted_markets import SYSTEM_PROMPT as FILTER_PROMPT, classify_markets
from main_backtesting.models import SourceEvent, SourceMarket
from main_backtesting.utils import chunks, input_hash


async def write_deleted_market_log(self, conn: Any) -> None:
    rows = await conn.fetch(
        """
        SELECT d.market_id, d.event_id, d.event_title, d.market_question,
               d.reason, d.model_name, d.prompt_version, d.processed_at
        FROM checking_relevant_events.historical_run_market_decisions r
        JOIN checking_relevant_events.historical_market_decisions d
          ON d.input_hash = r.input_hash
        WHERE r.run_id = $1 AND NOT d.relevant
        ORDER BY d.processed_at, d.market_id
        """,
        self.run_id,
    )
    path = self.run_dir / "logs" / "deleted_non_relevant_markets.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "market_id",
        "event_id",
        "event_title",
        "market_question",
        "reason",
        "model_name",
        "prompt_version",
        "processed_at",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


async def run_events(self, conn: Any) -> list[SourceEvent]:
    selected_event_ids = set(self.config.selected_event_ids)
    events = await candidate_events(
        conn,
        start=self.config.start,
        end=self.config.end,
        minimum_days_remaining=self.config.minimum_days_remaining,
        maximum_days_remaining=self.config.maximum_days_remaining,
        included_tags=sorted(self.config.included_tags),
        excluded_tags=sorted(self.config.excluded_tags),
        limit=0 if selected_event_ids else self.config.maximum_events,
    )
    if selected_event_ids:
        events = [event for event in events if event.event_id in selected_event_ids]
        missing = selected_event_ids - {event.event_id for event in events}
        if missing:
            raise ValueError(
                "Selected smoke-test events are not eligible under the configured "
                f"date/tag/5-to-60-day filters: {sorted(missing)}"
            )
        if self.config.maximum_events:
            events = events[: self.config.maximum_events]
    return events


async def accepted_markets(self, conn: Any) -> list[SourceMarket]:
    accepted = set(await accepted_market_ids(conn, self.run_id))
    return [
        market
        for market in await candidate_markets(self, conn)
        if market.market_id in accepted
    ]


async def candidate_markets(self, conn: Any) -> list[SourceMarket]:
    markets: list[SourceMarket] = []
    for event in await run_events(self, conn):
        markets.extend(
            market
            for market in await event_markets(conn, event)
            if self.config.minimum_days_remaining
            < (market.end_at - market.created_at).total_seconds() / 86_400
            <= self.config.maximum_days_remaining
        )
    return markets


async def run(self, conn: Any) -> None:
    markets = await candidate_markets(self, conn)
    for batch_index, batch in enumerate(chunks(markets, self.config.event_filter_batch_size), 1):
        self.current_work_key = f"batch:{batch_index}"
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="event_filter",
            work_key=self.current_work_key,
            payload={"market_ids": [market.market_id for market in batch]},
        ):
            continue
        payloads = [
            {
                "market_id": market.market_id,
                "event_id": market.event_id,
                "event_title": market.event_title,
                "market_question": market.question,
                "tags": market.tags,
                "created_at": market.created_at,
                "end_at": market.end_at,
            }
            for market in batch
        ]
        batch_llm_input = {
            "system_prompt": FILTER_PROMPT
            + "\nJudge each specific market question independently. Return exactly one "
            "decision for every supplied market_id.",
            "payload": {"markets": payloads},
        }
        identities = {
            market.market_id: input_hash(
                {
                    "task": "market_filter",
                    "model": self.ollama.model_name,
                    "prompt_version": self.config.event_filter_prompt_version,
                    "model_input": batch_llm_input,
                    "market_id": market.market_id,
                }
            )
            for market in batch
        }
        cached = {
            market.market_id: await reusable_market_decision(
                conn, identities[market.market_id]
            )
            for market in batch
        }
        cached_count = sum(item is not None for item in cached.values())
        if cached_count not in {0, len(batch)}:
            raise RuntimeError(
                f"Partial exact event-filter batch cache for {self.current_work_key}"
            )
        if cached_count == len(batch):
            async with conn.transaction():
                for market in batch:
                    item = cached[market.market_id]
                    identity = identities[market.market_id]
                    await save_market_decision(
                        conn,
                        run_id=self.run_id,
                        input_hash=identity,
                        market_id=market.market_id,
                        event_id=market.event_id,
                        event_title=market.event_title,
                        market_question=market.question,
                        model_name=item["model_name"],
                        prompt_version=item["prompt_version"],
                        llm_input=json_value(item["llm_input"]),
                        llm_output=json_value(item["llm_output"]),
                        relevant=item["relevant"],
                        reason=item["reason"],
                    )
        else:
            decisions = await classify_markets(self.ollama, batch)
            by_id = {item.market_id: item for item in decisions}
            batch_llm_output = {
                "decisions": [decision.model_dump(mode="json") for decision in decisions]
            }
            async with conn.transaction():
                for market in batch:
                    decision = by_id[market.market_id]
                    await save_market_decision(
                        conn,
                        run_id=self.run_id,
                        input_hash=identities[market.market_id],
                        market_id=market.market_id,
                        event_id=market.event_id,
                        event_title=market.event_title,
                        market_question=market.question,
                        model_name=self.ollama.model_name,
                        prompt_version=self.config.event_filter_prompt_version,
                        llm_input=batch_llm_input,
                        llm_output=batch_llm_output,
                        relevant=decision.relevant_to_financial_markets,
                        reason=decision.reason,
                    )
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="event_filter",
            work_key=self.current_work_key,
            result={"market_count": len(batch)},
        )
        await write_deleted_market_log(self, conn)
        print(f"[market-filter batch {batch_index}] markets={len(batch)}")
