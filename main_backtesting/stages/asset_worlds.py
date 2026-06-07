from __future__ import annotations

from datetime import datetime
from typing import Any
from database.backtesting.repositories import json_value
from database.backtesting.repositories.probabilities import run_passes
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import reusable_world, save_world
from LLM.build_world import SYSTEM_PROMPT as WORLD_PROMPT, assets_from_world, build_asset_worlds
from main_backtesting.models import Asset, SourceMarket
from main_backtesting.utils import chunks, input_hash

from main_backtesting.stages.event_filter import accepted_markets


async def run(self, conn: Any) -> None:
    market_map = {market.market_id: market for market in await accepted_markets(self, conn)}
    passes = list(await run_passes(conn, self.run_id))
    for batch_index, batch in enumerate(chunks(passes, self.config.asset_world_batch_size), 1):
        self.current_work_key = f"batch:{batch_index}"
        if not await start_work(
            conn,
            run_id=self.run_id,
            stage="asset_worlds",
            work_key=self.current_work_key,
            payload={"passes": [(row["market_id"], row["pass_number"]) for row in batch]},
        ):
            continue
        requests: list[tuple[str, SourceMarket, datetime]] = []
        rows_by_request: dict[str, Any] = {}
        payloads: list[dict[str, Any]] = []
        for row in batch:
            market = market_map[row["market_id"]]
            request_id = f"{market.market_id}:{row['pass_number']}"
            payload = {
                "request_id": request_id,
                "event_title": market.event_title,
                "market_question": market.question,
                "tags": market.tags,
                "market_created_at": market.created_at,
                "market_end_at": market.end_at,
                "historical_as_of": row["above_at"],
            }
            requests.append((request_id, market, row["above_at"]))
            rows_by_request[request_id] = row
            payloads.append(payload)
        batch_llm_input = {
            "system_prompt": WORLD_PROMPT
            + "\nBuild one independent world for every request_id and echo each request_id.",
            "payload": {"requests": payloads},
        }
        identities = {
            request_id: input_hash(
                {
                    "task": "asset_world",
                    "model": self.ollama.model_name,
                    "prompt_version": self.config.asset_world_prompt_version,
                    "model_input": batch_llm_input,
                    "request_id": request_id,
                }
            )
            for request_id, _, _ in requests
        }
        cached = {
            request_id: await reusable_world(conn, identities[request_id])
            for request_id, _, _ in requests
        }
        cached_count = sum(item is not None for item in cached.values())
        if cached_count not in {0, len(requests)}:
            raise RuntimeError(
                f"Partial exact asset-world batch cache for {self.current_work_key}"
            )
        if cached_count == len(requests):
            async with conn.transaction():
                for request_id, market, as_of in requests:
                    item = cached[request_id]
                    row = rows_by_request[request_id]
                    assets_rows = await conn.fetch(
                        """
                        SELECT symbol, asset_name, asset_class, reason
                        FROM checking_relevant_events.historical_asset_world_assets
                        WHERE world_id = $1 ORDER BY symbol
                        """,
                        item["world_id"],
                    )
                    await save_world(
                        conn,
                        run_id=self.run_id,
                        input_hash=identities[request_id],
                        market=market,
                        pass_number=row["pass_number"],
                        as_of=as_of,
                        model_name=item["model_name"],
                        prompt_version=item["prompt_version"],
                        llm_input=json_value(item["llm_input"]),
                        llm_output=json_value(item["llm_output"]),
                        universe_name=item["universe_name"],
                        universe_reason=item["universe_reason"],
                        assets=[Asset(**dict(asset_row)) for asset_row in assets_rows],
                    )
        else:
            worlds = await build_asset_worlds(self.ollama, requests)
            by_id = {world.request_id: world for world in worlds}
            batch_llm_output = {
                "worlds": [world.model_dump(mode="json") for world in worlds]
            }
            async with conn.transaction():
                for request_id, market, as_of in requests:
                    row = rows_by_request[request_id]
                    world = by_id[request_id]
                    await save_world(
                        conn,
                        run_id=self.run_id,
                        input_hash=identities[request_id],
                        market=market,
                        pass_number=row["pass_number"],
                        as_of=as_of,
                        model_name=self.ollama.model_name,
                        prompt_version=self.config.asset_world_prompt_version,
                        llm_input=batch_llm_input,
                        llm_output=batch_llm_output,
                        universe_name=world.universe_name,
                        universe_reason=world.universe_reason,
                        assets=assets_from_world(world),
                    )
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="asset_worlds",
            work_key=self.current_work_key,
            result={"world_count": len(batch)},
        )
        print(f"[world batch {batch_index}] worlds={len(batch)}")
