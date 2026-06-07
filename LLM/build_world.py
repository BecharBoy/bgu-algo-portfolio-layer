from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from LLM.ollama_client import OllamaClient
from main_backtesting.models import Asset, SourceMarket


class AssetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20)
    asset_name: str = Field(min_length=1, max_length=120)
    asset_class: Literal["stock", "etf"]
    reason: str = Field(min_length=20, max_length=500)

    @field_validator("symbol")
    @classmethod
    def uppercase_symbol(cls, value: str) -> str:
        return value.upper()


class AssetWorld(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    universe_name: str = Field(min_length=1, max_length=200)
    universe_reason: str = Field(min_length=20, max_length=700)
    assets: list[AssetCandidate] = Field(min_length=4, max_length=15)

    @model_validator(mode="after")
    def unique_symbols(self) -> AssetWorld:
        symbols = [asset.symbol for asset in self.assets]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Asset world contains duplicate symbols")
        return self


class BatchedAssetWorld(AssetWorld):
    request_id: str


class BatchedAssetWorlds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worlds: list[BatchedAssetWorld]


SYSTEM_PROMPT = """
Build the research world of US-listed stocks and equity ETFs that have a concrete
economic relationship to the supplied prediction-market question.

This is asset selection only. Do not predict Yes or No, provide direction, confidence,
position sizing, or trading advice. Prefer individual companies and include sector or
country ETFs only when useful. Every asset requires a specific reason. Do not add famous
tickers merely to fill the list. Return at least four assets and only the supplied JSON schema.
""".strip()


async def build_asset_world(
    ollama: OllamaClient,
    market: SourceMarket,
    *,
    as_of: datetime | None = None,
) -> AssetWorld:
    return await ollama.structured(
        system_prompt=SYSTEM_PROMPT,
        payload={
            "event_title": market.event_title,
            "market_question": market.question,
            "tags": market.tags,
            "market_created_at": market.created_at,
            "market_end_at": market.end_at,
            "historical_as_of": as_of,
        },
        response_model=AssetWorld,
        max_tokens=1000,
    )


async def build_asset_worlds(
    ollama: OllamaClient,
    requests: list[tuple[str, SourceMarket, datetime]],
) -> list[BatchedAssetWorld]:
    if not requests:
        return []
    response = await ollama.structured(
        system_prompt=SYSTEM_PROMPT
        + "\nBuild one independent world for every request_id and echo each request_id.",
        payload={
            "requests": [
                {
                    "request_id": request_id,
                    "event_title": market.event_title,
                    "market_question": market.question,
                    "tags": market.tags,
                    "market_created_at": market.created_at,
                    "market_end_at": market.end_at,
                    "historical_as_of": as_of,
                }
                for request_id, market, as_of in requests
            ]
        },
        response_model=BatchedAssetWorlds,
        max_tokens=max(1200, len(requests) * 1200),
    )
    expected = {request_id for request_id, _, _ in requests}
    actual = {item.request_id for item in response.worlds}
    if len(actual) != len(response.worlds) or actual != expected:
        raise ValueError(f"World batch identifiers mismatch: expected={expected} actual={actual}")
    return response.worlds


def assets_from_world(world: AssetWorld) -> list[Asset]:
    return [
        Asset(
            symbol=item.symbol,
            asset_name=item.asset_name,
            asset_class=item.asset_class,
            reason=item.reason,
        )
        for item in world.assets
    ]
