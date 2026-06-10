from __future__ import annotations

from datetime import datetime
from typing import Any
from database.backtesting.repositories.probabilities import run_passes
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import (
    ib_tradable_assets,
    link_run_world,
    reusable_world,
    save_world,
)
from LLM.build_world import (
    SYSTEM_PROMPT as WORLD_PROMPT,
    IBAssetCatalogIndex,
    assets_from_world,
    build_asset_worlds,
)
from main_backtesting.models import SourceMarket
from main_backtesting.utils import chunks, input_hash

from main_backtesting.stages.event_filter import accepted_markets


async def run(self, conn: Any) -> None:
    tradable_assets = await ib_tradable_assets(conn)
    if not tradable_assets:
        raise RuntimeError(
            "The IB-confirmed tradable universe is missing or empty. Populate "
            "checking_relevant_events.ib_assets with "
            "`.venv\\Scripts\\python.exe -m database.map_ib_assets` before running "
            "the asset-world stage."
        )
    catalog_hash = input_hash([asset.prompt_record() for asset in tradable_assets])
    asset_catalog = IBAssetCatalogIndex(tradable_assets)
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
        payloads_by_request: dict[str, dict[str, Any]] = {}
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
            payloads_by_request[request_id] = payload
            payloads.append(payload)
        batch_llm_input = {
            "system_prompt": WORLD_PROMPT
            + "\nBuild one independent world for every request_id and echo each request_id.",
            "payload": {"requests": payloads},
            "selection_mode": "discover_then_select_from_relevant_ib_catalog",
            "ib_asset_catalog_count": len(tradable_assets),
            "ib_asset_catalog_hash": catalog_hash,
        }
        identities = {
            request_id: input_hash(
                {
                    "task": "asset_world",
                    "model": self.ollama.model_name,
                    "prompt_version": self.config.asset_world_prompt_version,
                    "ib_asset_catalog_hash": catalog_hash,
                    "model_input": payloads_by_request[request_id],
                }
            )
            for request_id, _, _ in requests
        }
        cached = {
            request_id: await reusable_world(conn, identities[request_id])
            for request_id, _, _ in requests
        }
        cached_requests = [request for request in requests if cached[request[0]] is not None]
        missing_requests = [request for request in requests if cached[request[0]] is None]
        if cached_requests:
            async with conn.transaction():
                for request_id, market, as_of in cached_requests:
                    item = cached[request_id]
                    row = rows_by_request[request_id]
                    await link_run_world(
                        conn,
                        run_id=self.run_id,
                        market_id=market.market_id,
                        pass_number=row["pass_number"],
                        world_id=item["world_id"],
                    )
        if missing_requests:
            worlds = await build_asset_worlds(
                self.ollama,
                missing_requests,
                tradable_assets=asset_catalog,
            )
            by_id = {world.request_id: world for world in worlds}
            batch_llm_output = {
                "worlds": [world.model_dump(mode="json") for world in worlds]
            }
            async with conn.transaction():
                for request_id, market, as_of in missing_requests:
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
