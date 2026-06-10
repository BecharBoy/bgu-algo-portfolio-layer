from __future__ import annotations

from datetime import datetime
import re
from typing import Callable, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    field_validator,
    model_validator,
)

from LLM.ollama_client import OllamaClient
from main_backtesting.models import Asset, IBTradableAsset, SourceMarket


class AssetCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20)
    asset_name: str = Field(min_length=1, max_length=120)
    asset_class: Literal["stock", "etf"]
    relationship_type: Literal[
        "direct_company",
        "customer",
        "supplier",
        "distributor",
        "partner",
        "competitor",
        "substitute",
        "complement",
        "creditor",
        "investor",
        "landlord_tenant",
        "sector_etf",
        "country_etf",
        "commodity_proxy",
        "other_specific",
    ]
    reason: str = Field(
        min_length=20,
        max_length=500,
        description=(
            "Specific causal economic relationship to the exact question. Generic claims "
            "that an asset may be affected are insufficient."
        ),
    )

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
    def require_unique_symbols(self) -> AssetWorld:
        symbols = [
            asset.symbol.strip().upper().replace(".", " ").replace("$", " ")
            for asset in self.assets
        ]
        if len(symbols) != len(set(symbols)):
            raise ValueError("Asset world contains duplicate symbols")
        return self


class BatchedAssetWorld(AssetWorld):
    request_id: str


class BatchedAssetWorlds(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worlds: list[BatchedAssetWorld]



class ClusteredRoutingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    event_archetype: Literal["Macroeconomic / Fed", "Geopolitical / Conflict", "Earnings / Financials", "Regulatory / Legal", "Product Releases / Tech", "Asset Prices / Targets", "Other"]
    primary_direct_companies: list[str] = Field(default_factory=list, max_length=5)
    custom_related_companies: list[str] = Field(default_factory=list, max_length=15)
    impacted_ib_industries: list[str] = Field(default_factory=list, max_length=5)
    impacted_ib_categories: list[str] = Field(default_factory=list, max_length=5)
    rationale: str = Field(min_length=20, max_length=700)

class CatalogSearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    direct_entities: list[str] = Field(default_factory=list, max_length=10)
    related_entities: list[str] = Field(default_factory=list, max_length=20)
    ticker_hints: list[str] = Field(default_factory=list, max_length=20)
    industries: list[str] = Field(default_factory=list, max_length=15)
    etf_themes: list[str] = Field(default_factory=list, max_length=15)
    economic_keywords: list[str] = Field(default_factory=list, max_length=25)
    rationale: str = Field(min_length=20, max_length=700)

    @field_validator("ticker_hints")
    @classmethod
    def uppercase_ticker_hints(cls, values: list[str]) -> list[str]:
        return [value.upper() for value in values]


SYSTEM_PROMPT = """
Build a cross-sectional research world of 4-15 stocks and equity ETFs that have concrete
economic relationships to the supplied prediction-market question.

This is asset selection only. Do not predict Yes or No, provide direction, confidence,
position sizing, or trading advice. Choose only real tickers available in the IB-confirmed
tradable universe. The final output is validated against that universe.

Think broadly about the event's economic relationship graph. Relevant relationships can
include the directly named company, customers, suppliers, distributors, partners,
competitors, substitutes, complements, creditors, investors, landlords, tenants, and
specific sector or country ETFs. Competitors are valid when the event changes shared
demand, supply, pricing, market share, or substitution economics; being in the same broad
industry alone is not enough.

For company earnings questions, build the world around the reporting company and the
companies whose economics are materially connected to its results. Do not assume whether
earnings will beat or miss.

For geopolitical questions, infer the economic transmission channels from the exact
question, such as commodities, supply chains, shipping, defense demand, sanctions,
regional revenue, or country exposure. Decide the assets yourself; do not default to
famous companies or broad-market ETFs.

Every reason must state the relationship and causal transmission channel. Phrases such as
"may be affected", "potential applications", or "has exposure" are insufficient unless
the reason explains exactly why. Classify every asset with the supplied relationship_type;
use other_specific only when none of the defined relationships fits and explain it fully.
Return 4-15 unique assets and only the supplied JSON schema.
""".strip()


CLUSTERED_ROUTER_PROMPT = """
You are an elite financial analyst at Morgan Stanley. Your mission is to analyze a prediction market event and build a research routing plan. Identify the event archetype. Name the EXACT primary companies involved, their direct competitors, and select the 1-2 most impacted industry subsectors/categories from the provided list. Do not make shit up.
Return only the supplied JSON schema.
""".strip()

CLUSTERED_SELECTION_PROMPT = (
    SYSTEM_PROMPT
    + """

You are the lead analyst. You routed a market event and retrieved the following IB-verified assets based on the exact companies and subsectors you requested. The user payload contains these strongest matching catalog records in available_ib_assets. Select the best 4-15 assets only from that list. Copy symbol, asset_name, and asset_class exactly. Write a factual, concrete economic reason for each. Do not make shit up. NEVER provide trading advice, buy/sell signals, or technical analysis.
""".rstrip()
)
CATALOG_SEARCH_PROMPT = """
Create a search plan for finding economically related stocks and equity ETFs inside a
complete Interactive Brokers asset catalog.

Do not select the final assets yet. Identify the directly named companies, economically
related companies, likely ticker hints, relevant industries, specific ETF themes, and
economic transmission keywords. Think about customers, suppliers, partners, competitors,
substitutes, complements, creditors, investors, landlords, tenants, commodities, country
exposure, and sector exposure. Avoid generic words that would match almost every company.
Return only the supplied JSON schema.
""".strip()

CATALOG_SELECTION_PROMPT = (
    SYSTEM_PROMPT
    + """

The user payload contains available_ib_assets. This is the authoritative shortlist of
real, currently IB-tradable assets relevant to the question. Choose every final asset
only from that list. Copy its symbol, asset_name, and asset_class exactly. Do not invent
or modify tickers. The discovered_world is research input, not an allowed-symbol list;
use it to select the strongest 4-15 assets available in available_ib_assets. Preserve the
discovered relationship_type and concrete causal reason for every retained discovered
asset.
""".rstrip()
)

FULL_CATALOG_SELECTION_PROMPT = (
    SYSTEM_PROMPT
    + """

The system searched the complete IB-confirmed stock and equity-ETF catalog using your
economic search plan. The user payload contains the strongest matching catalog records in
available_ib_assets. Select the best 4-15 assets only from that list. Copy symbol,
asset_name, and asset_class exactly. Do not assume that a keyword or industry match is
economically relevant; include an asset only when you can explain a concrete relationship
and causal transmission channel to the exact market question.
""".rstrip()
)

MAX_MISSING_BATCH_RETRIES = 3
MAX_INVALID_SYMBOL_RETRIES = 3
CATALOG_RETRIEVAL_LIMIT = 80
CATALOG_TOKEN_RE = re.compile(r"[A-Z0-9]+")
CATALOG_STOP_WORDS = {
    "A",
    "AN",
    "AND",
    "ARE",
    "CLASS",
    "CO",
    "COMMON",
    "COMPANY",
    "CORP",
    "CORPORATION",
    "ETF",
    "FUND",
    "HOLDINGS",
    "INC",
    "INCORPORATED",
    "LTD",
    "OF",
    "ORDINARY",
    "SHARE",
    "SHARES",
    "STOCK",
    "THE",
}


class IBAssetCatalogIndex:
    def __init__(self, assets: list[IBTradableAsset]) -> None:
        self.assets = tuple(assets)
        self.by_symbol = {ib_symbol_key(asset.symbol): asset for asset in assets}
        self.name_text: dict[str, str] = {}
        self.name_tokens: dict[str, set[str]] = {}
        self.metadata_tokens: dict[str, set[str]] = {}
        self.name_token_symbols: dict[str, set[str]] = {}
        self.metadata_token_symbols: dict[str, set[str]] = {}
        for asset in assets:
            key = ib_symbol_key(asset.symbol)
            self.name_text[key] = normalized_catalog_text(asset.asset_name)
            name_tokens = catalog_tokens(asset.symbol, asset.asset_name)
            metadata_tokens = catalog_tokens(
                asset.industry,
                asset.category,
                asset.subcategory,
            )
            self.name_tokens[key] = name_tokens
            self.metadata_tokens[key] = metadata_tokens
            for token in name_tokens:
                self.name_token_symbols.setdefault(token, set()).add(key)
            for token in metadata_tokens:
                self.metadata_token_symbols.setdefault(token, set()).add(key)


def ib_asset_catalog_index(
    assets: list[IBTradableAsset] | IBAssetCatalogIndex,
) -> IBAssetCatalogIndex:
    if isinstance(assets, IBAssetCatalogIndex):
        return assets
    return IBAssetCatalogIndex(assets)


def ib_symbol_key(symbol: str) -> str:
    return symbol.strip().upper().replace(".", " ").replace("$", " ")


def normalized_catalog_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(CATALOG_TOKEN_RE.findall(value.upper()))


def catalog_tokens(*values: str | None) -> set[str]:
    return {
        token
        for value in values
        for token in CATALOG_TOKEN_RE.findall((value or "").upper())
        if len(token) > 1 and token not in CATALOG_STOP_WORDS
    }


def retrieve_catalog_candidates(
    market: SourceMarket,
    plan: CatalogSearchPlan,
    catalog: IBAssetCatalogIndex,
    *,
    limit: int = CATALOG_RETRIEVAL_LIMIT,
) -> list[IBTradableAsset]:
    direct_phrases = [normalized_catalog_text(value) for value in plan.direct_entities]
    related_phrases = [normalized_catalog_text(value) for value in plan.related_entities]
    direct_tokens = catalog_tokens(*plan.direct_entities)
    related_tokens = catalog_tokens(*plan.related_entities)
    industry_tokens = catalog_tokens(*plan.industries)
    theme_tokens = catalog_tokens(*plan.etf_themes)
    economic_tokens = catalog_tokens(*plan.economic_keywords)
    market_tokens = catalog_tokens(market.event_title, market.question, *market.tags)
    candidate_keys = {
        ib_symbol_key(symbol)
        for symbol in plan.ticker_hints
        if ib_symbol_key(symbol) in catalog.by_symbol
    }
    for token in (
        direct_tokens
        | related_tokens
        | industry_tokens
        | theme_tokens
        | economic_tokens
        | market_tokens
    ):
        candidate_keys.update(catalog.name_token_symbols.get(token, ()))
        candidate_keys.update(catalog.metadata_token_symbols.get(token, ()))

    ticker_hints = {ib_symbol_key(symbol) for symbol in plan.ticker_hints}
    ranked: list[tuple[int, str, IBTradableAsset]] = []
    for key in candidate_keys:
        asset = catalog.by_symbol[key]
        name_tokens = catalog.name_tokens[key]
        metadata_tokens = catalog.metadata_tokens[key]
        name_text = catalog.name_text[key]
        score = 0
        if key in ticker_hints:
            score += 20_000
        score += sum(2_000 for phrase in direct_phrases if phrase and phrase in name_text)
        score += sum(1_000 for phrase in related_phrases if phrase and phrase in name_text)
        score += len(direct_tokens & name_tokens) * 500
        score += len(related_tokens & name_tokens) * 300
        score += len(industry_tokens & metadata_tokens) * 180
        score += len(economic_tokens & metadata_tokens) * 80
        score += len(market_tokens & name_tokens) * 60
        score += len(market_tokens & metadata_tokens) * 30
        if asset.asset_class == "etf":
            score += len(theme_tokens & (name_tokens | metadata_tokens)) * 220
        if score > 0:
            ranked.append((score, asset.symbol, asset))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [asset for _, _, asset in ranked[:limit]]


def invalid_ib_symbols(world: AssetWorld, tradable_symbols: set[str] | None) -> list[str]:
    if tradable_symbols is None:
        return []
    allowed = {ib_symbol_key(symbol) for symbol in tradable_symbols}
    return [
        asset.symbol
        for asset in world.assets
        if ib_symbol_key(asset.symbol) not in allowed
    ]


def verified_discovered_assets(
    discovered_world: AssetWorld,
    tradable_assets: list[IBTradableAsset] | IBAssetCatalogIndex,
) -> list[IBTradableAsset]:
    catalog = ib_asset_catalog_index(tradable_assets)
    return [
        catalog.by_symbol[key]
        for asset in discovered_world.assets
        if (key := ib_symbol_key(asset.symbol)) in catalog.by_symbol
    ]


def invalid_catalog_symbols(
    world: AssetWorld,
    available_assets: list[IBTradableAsset],
) -> list[str]:
    allowed = {ib_symbol_key(asset.symbol) for asset in available_assets}
    return [
        asset.symbol
        for asset in world.assets
        if ib_symbol_key(asset.symbol) not in allowed
    ]


def canonicalize_catalog_world(
    world: AssetWorld,
    available_assets: list[IBTradableAsset],
    discovered_world: AssetWorld | None = None,
) -> AssetWorld:
    by_symbol = {ib_symbol_key(asset.symbol): asset for asset in available_assets}
    discovered_by_symbol = {
        ib_symbol_key(asset.symbol): asset
        for asset in discovered_world.assets
    } if discovered_world is not None else {}
    assets = []
    for selected in world.assets:
        key = ib_symbol_key(selected.symbol)
        canonical = by_symbol[key]
        discovered = discovered_by_symbol.get(key)
        updates = {
            "symbol": canonical.symbol,
            "asset_name": canonical.asset_name,
            "asset_class": canonical.asset_class,
        }
        if discovered is not None:
            updates["relationship_type"] = discovered.relationship_type
            updates["reason"] = discovered.reason
        assets.append(
            selected.model_copy(update=updates)
        )
    return world.model_copy(update={"assets": assets})


def catalog_asset_world_model(
    available_assets: list[IBTradableAsset],
) -> type[AssetWorld]:
    symbols = tuple(asset.symbol for asset in available_assets)
    allowed_symbol = Literal.__getitem__(symbols)
    candidate_model = create_model(
        "CatalogAssetCandidate",
        __base__=AssetCandidate,
        symbol=(allowed_symbol, ...),
    )
    return create_model(
        "CatalogAssetWorld",
        __base__=AssetWorld,
        assets=(list[candidate_model], Field(min_length=4, max_length=15)),
    )


def catalog_selection_payload(
    request_id: str | None,
    market: SourceMarket,
    as_of: datetime,
    discovered_world: AssetWorld,
    available_assets: list[IBTradableAsset],
    *,
    rejected_symbols: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_title": market.event_title,
        "market_question": market.question,
        "tags": market.tags,
        "market_created_at": market.created_at,
        "market_end_at": market.end_at,
        "historical_as_of": as_of,
        "discovered_world": discovered_world.model_dump(mode="json"),
        "available_ib_assets": [asset.prompt_record() for asset in available_assets],
    }
    if request_id is not None:
        payload["request_id"] = request_id
    if rejected_symbols:
        payload["rejected_symbols_not_in_available_ib_assets"] = rejected_symbols
        payload["replacement_instruction"] = (
            "Replace every rejected symbol using only available_ib_assets and return "
            "a complete 4-15 asset world."
        )
    return payload


async def select_asset_world_from_catalog(
    ollama: OllamaClient,
    market: SourceMarket,
    as_of: datetime,
    discovered_world: AssetWorld,
    available_assets: list[IBTradableAsset],
) -> AssetWorld:
    if len(available_assets) < 4:
        raise ValueError(
            "Fewer than four relevant IB-tradable candidates were found for "
            f"market {market.market_id}: {len(available_assets)}"
        )
    rejected_symbols: list[str] = []
    response_model = catalog_asset_world_model(available_assets)
    for attempt in range(MAX_INVALID_SYMBOL_RETRIES + 1):
        payload = catalog_selection_payload(
            None,
            market,
            as_of,
            discovered_world,
            available_assets,
            rejected_symbols=rejected_symbols,
        )
        if attempt:
            payload["schema_retry_instruction"] = (
                "The previous response violated the allowed-symbol JSON schema. Use only "
                "an exact symbol enum value from available_ib_assets."
            )
        try:
            world = await ollama.structured(
                system_prompt=CATALOG_SELECTION_PROMPT,
                payload=payload,
                response_model=response_model,
                max_tokens=2200,
            )
        except ValidationError:
            continue
        rejected_symbols = invalid_catalog_symbols(world, available_assets)
        if not rejected_symbols:
            return canonicalize_catalog_world(world, available_assets, discovered_world)
    raise ValueError(
        "Asset world still contains symbols outside the supplied IB candidate catalog "
        f"after {MAX_INVALID_SYMBOL_RETRIES + 1} attempts: {rejected_symbols}"
    )


async def build_catalog_retrieval_world(
    ollama: OllamaClient,
    market: SourceMarket,
    *,
    as_of: datetime,
    catalog: IBAssetCatalogIndex,
    progress: Callable[[str, dict[str, object]], None] | None = None,
) -> tuple[AssetWorld, CatalogSearchPlan, list[IBTradableAsset]]:
    search_payload = {
        "event_title": market.event_title,
        "market_question": market.question,
        "tags": market.tags,
        "market_created_at": market.created_at,
        "market_end_at": market.end_at,
        "historical_as_of": as_of,
        "catalog_asset_count": len(catalog.assets),
        "catalog_fields": [
            "symbol",
            "asset_name",
            "asset_class",
            "primary_exchange",
            "industry",
            "category",
            "subcategory",
        ],
    }
    if progress:
        progress("search-plan-start", {"catalog_assets": len(catalog.assets)})
    plan = await ollama.structured(
        system_prompt=CATALOG_SEARCH_PROMPT,
        payload=search_payload,
        response_model=CatalogSearchPlan,
        max_tokens=1200,
    )
    if progress:
        progress(
            "search-plan-complete",
            {
                "direct_entities": plan.direct_entities,
                "related_entities": plan.related_entities,
                "industries": plan.industries,
                "etf_themes": plan.etf_themes,
                "ticker_hints": plan.ticker_hints,
            },
        )
    candidates = retrieve_catalog_candidates(market, plan, catalog)
    if progress:
        progress(
            "candidates-retrieved",
            {
                "count": len(candidates),
                "symbols": [asset.symbol for asset in candidates],
            },
        )
    if len(candidates) < 4:
        raise ValueError(
            "Full-catalog retrieval found fewer than four candidates for "
            f"market {market.market_id}: {len(candidates)}"
        )
    response_model = catalog_asset_world_model(candidates)
    selection_payload = {
        **search_payload,
        "catalog_search_plan": plan.model_dump(mode="json"),
        "available_ib_assets": [asset.prompt_record() for asset in candidates],
    }
    for attempt in range(MAX_INVALID_SYMBOL_RETRIES + 1):
        if progress:
            progress(
                "selection-start",
                {"attempt": attempt + 1, "available_candidates": len(candidates)},
            )
        if attempt:
            selection_payload["schema_retry_instruction"] = (
                "Use only an exact allowed symbol from available_ib_assets."
            )
        try:
            world = await ollama.structured(
                system_prompt=FULL_CATALOG_SELECTION_PROMPT,
                payload=selection_payload,
                response_model=response_model,
                max_tokens=2200,
            )
        except ValidationError:
            if progress:
                progress("selection-schema-retry", {"attempt": attempt + 1})
            continue
        canonical_world = canonicalize_catalog_world(world, candidates)
        if progress:
            progress(
                "selection-complete",
                {"symbols": [asset.symbol for asset in canonical_world.assets]},
            )
        return canonical_world, plan, candidates
    raise ValueError(
        "Catalog-retrieval selection failed the allowed-symbol schema after "
        f"{MAX_INVALID_SYMBOL_RETRIES + 1} attempts"
    )


async def select_asset_worlds_from_catalog(
    ollama: OllamaClient,
    requests: list[tuple[str, SourceMarket, datetime]],
    discovered_worlds: list[BatchedAssetWorld],
    available_by_request: dict[str, list[IBTradableAsset]],
) -> list[BatchedAssetWorld]:
    discovered_by_request = {world.request_id: world for world in discovered_worlds}
    selected_worlds = []
    for request_id, market, as_of in requests:
        selected = await select_asset_world_from_catalog(
            ollama,
            market,
            as_of,
            discovered_by_request[request_id],
            available_by_request[request_id],
        )
        selected_worlds.append(
            BatchedAssetWorld(request_id=request_id, **selected.model_dump())
        )
    return selected_worlds


async def build_asset_world(
    ollama: OllamaClient,
    market: SourceMarket,
    *,
    as_of: datetime | None = None,
    tradable_symbols: set[str] | None = None,
    initial_invalid_symbols: list[str] | None = None,
) -> AssetWorld:
    invalid_symbols = list(initial_invalid_symbols or [])
    for attempt in range(MAX_INVALID_SYMBOL_RETRIES + 1):
        payload = {
            "event_title": market.event_title,
            "market_question": market.question,
            "tags": market.tags,
            "market_created_at": market.created_at,
            "market_end_at": market.end_at,
            "historical_as_of": as_of,
        }
        if invalid_symbols:
            payload["rejected_symbols_not_in_ib_tradable_universe"] = invalid_symbols
            payload["replacement_instruction"] = (
                "Replace every rejected symbol with a different economically related "
                "IB-tradable stock or ETF. Return a complete 4-15 asset world."
            )
        world = await ollama.structured(
            system_prompt=SYSTEM_PROMPT,
            payload=payload,
            response_model=AssetWorld,
            max_tokens=2200,
        )
        invalid_symbols = invalid_ib_symbols(world, tradable_symbols)
        if not invalid_symbols:
            return world
    raise ValueError(
        "Asset world still contains symbols outside the IB-confirmed tradable universe "
        f"after {MAX_INVALID_SYMBOL_RETRIES + 1} attempts: {invalid_symbols}"
    )


async def _build_single_asset_world(
    ollama: OllamaClient,
    request_id: str,
    market: SourceMarket,
    as_of: datetime,
    *,
    tradable_symbols: set[str] | None = None,
    initial_invalid_symbols: list[str] | None = None,
) -> BatchedAssetWorld:
    world = await build_asset_world(
        ollama,
        market,
        as_of=as_of,
        tradable_symbols=tradable_symbols,
        initial_invalid_symbols=initial_invalid_symbols,
    )
    return BatchedAssetWorld(request_id=request_id, **world.model_dump())


async def build_asset_worlds(
    ollama: OllamaClient,
    requests: list[tuple[str, SourceMarket, datetime]],
    *,
    tradable_symbols: set[str] | None = None,
    tradable_assets: list[IBTradableAsset] | IBAssetCatalogIndex | None = None,
    progress: Callable[[str, dict[str, object]], None] | None = None,
) -> list[BatchedAssetWorld]:
    if progress:
        progress("discovery-start", {"requests": len(requests)})
    discovered_worlds = await _build_asset_worlds(
        ollama,
        requests,
        missing_retry_attempt=0,
        tradable_symbols=None if tradable_assets is not None else tradable_symbols,
    )
    if progress:
        progress(
            "discovery-complete",
            {
                "symbols": [
                    asset.symbol
                    for world in discovered_worlds
                    for asset in world.assets
                ]
            },
        )
    if tradable_assets is None:
        return discovered_worlds
    catalog = ib_asset_catalog_index(tradable_assets)
    verified_discoveries = []
    available_by_request = {}
    for (request_id, market, as_of), discovered_world in zip(requests, discovered_worlds):
        if len(verified_discovered_assets(discovered_world, catalog)) < 4:
            invalid_symbols = invalid_ib_symbols(
                discovered_world,
                set(catalog.by_symbol),
            )
            if progress:
                progress(
                    "correction-start",
                    {"request_id": request_id, "invalid_symbols": invalid_symbols},
                )
            corrected = await build_asset_world(
                ollama,
                market,
                as_of=as_of,
                tradable_symbols=set(catalog.by_symbol),
                initial_invalid_symbols=invalid_symbols,
            )
            discovered_world = BatchedAssetWorld(
                request_id=request_id,
                **corrected.model_dump(),
            )
            if progress:
                progress(
                    "correction-complete",
                    {"symbols": [asset.symbol for asset in discovered_world.assets]},
                )
        available = verified_discovered_assets(discovered_world, catalog)
        if len(available) < 4:
            raise ValueError(
                "Asset discovery did not produce four IB-confirmed stocks/ETFs for "
                f"market {market.market_id}: {len(available)}"
            )
        if progress:
            progress(
                "ib-validation-complete",
                {
                    "request_id": request_id,
                    "verified_count": len(available),
                    "symbols": [asset.symbol for asset in available],
                },
            )
        verified_discoveries.append(discovered_world)
        available_by_request[request_id] = available
    if progress:
        progress("final-selection-start", {"requests": len(requests)})
    selected_worlds = await select_asset_worlds_from_catalog(
        ollama,
        requests,
        verified_discoveries,
        available_by_request,
    )
    if progress:
        progress(
            "final-selection-complete",
            {
                "symbols": [
                    asset.symbol
                    for world in selected_worlds
                    for asset in world.assets
                ]
            },
        )
    return selected_worlds


async def _build_asset_worlds(
    ollama: OllamaClient,
    requests: list[tuple[str, SourceMarket, datetime]],
    *,
    missing_retry_attempt: int,
    tradable_symbols: set[str] | None,
) -> list[BatchedAssetWorld]:
    if not requests:
        return []
    if len(requests) == 1:
        request_id, market, as_of = requests[0]
        return [
            await _build_single_asset_world(
                ollama,
                request_id,
                market,
                as_of,
                tradable_symbols=tradable_symbols,
            )
        ]
    try:
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
        seen: dict[str, BatchedAssetWorld] = {}
        invalid: dict[str, list[str]] = {}
        for world in response.worlds:
            if world.request_id in expected:
                rejected = invalid_ib_symbols(world, tradable_symbols)
                if rejected:
                    invalid[world.request_id] = rejected
                else:
                    seen.setdefault(world.request_id, world)
        for request_id, market, as_of in requests:
            if request_id in invalid:
                seen[request_id] = await _build_single_asset_world(
                    ollama,
                    request_id,
                    market,
                    as_of,
                    tradable_symbols=tradable_symbols,
                    initial_invalid_symbols=invalid[request_id],
                )
        missing = [request for request in requests if request[0] not in seen]
        if missing:
            if missing_retry_attempt >= MAX_MISSING_BATCH_RETRIES:
                print(
                    f"[asset-world] Ollama dropped {len(missing)} world(s) after "
                    f"{MAX_MISSING_BATCH_RETRIES} batch retries, building them individually"
                )
                retried = [
                    await _build_single_asset_world(
                        ollama,
                        request_id,
                        market,
                        as_of,
                        tradable_symbols=tradable_symbols,
                    )
                    for request_id, market, as_of in missing
                ]
            else:
                print(
                    f"[asset-world] Ollama dropped {len(missing)} world(s) from batch of "
                    f"{len(requests)}, retrying only missing request IDs "
                    f"({missing_retry_attempt + 1}/{MAX_MISSING_BATCH_RETRIES})"
                )
                retried = await _build_asset_worlds(
                    ollama,
                    missing,
                    missing_retry_attempt=missing_retry_attempt + 1,
                    tradable_symbols=tradable_symbols,
                )
            seen.update({world.request_id: world for world in retried})
        return [seen[request_id] for request_id, _, _ in requests]
    except (ValidationError, ValueError) as exc:
        midpoint = len(requests) // 2
        print(
            f"[asset-world] Batch of {len(requests)} failed ({type(exc).__name__}), "
            f"splitting into {midpoint}+{len(requests) - midpoint}"
        )
        left = await _build_asset_worlds(
            ollama,
            requests[:midpoint],
            missing_retry_attempt=missing_retry_attempt,
            tradable_symbols=tradable_symbols,
        )
        right = await _build_asset_worlds(
            ollama,
            requests[midpoint:],
            missing_retry_attempt=missing_retry_attempt,
            tradable_symbols=tradable_symbols,
        )
        return left + right


def assets_from_world(world: AssetWorld) -> list[Asset]:
    return [
        Asset(
            symbol=item.symbol,
            asset_name=item.asset_name,
            asset_class=item.asset_class,
            reason=f"[{item.relationship_type}] {item.reason}",
        )
        for item in world.assets
    ]


def retrieve_clustered_candidates(
    plan: ClusteredRoutingPlan,
    catalog: IBAssetCatalogIndex,
    *,
    limit: int = CATALOG_RETRIEVAL_LIMIT,
) -> list[IBTradableAsset]:
    direct_phrases = [normalized_catalog_text(value) for value in plan.primary_direct_companies]
    related_phrases = [normalized_catalog_text(value) for value in plan.custom_related_companies]
    
    candidate_keys = set()
    
    for phrase in direct_phrases + related_phrases:
        if not phrase:
            continue
        tokens = catalog_tokens(phrase)
        for token in tokens:
            candidate_keys.update(catalog.name_token_symbols.get(token, ()))

    industry_tokens = catalog_tokens(*plan.impacted_ib_industries)
    category_tokens = catalog_tokens(*plan.impacted_ib_categories)
    
    for token in (industry_tokens | category_tokens):
        candidate_keys.update(catalog.metadata_token_symbols.get(token, ()))

    ranked: list[tuple[int, str, IBTradableAsset]] = []
    for key in candidate_keys:
        asset = catalog.by_symbol[key]
        name_tokens = catalog.name_tokens[key]
        metadata_tokens = catalog.metadata_tokens[key]
        name_text = catalog.name_text[key]
        score = 0
        
        score += sum(20_000 for phrase in direct_phrases if phrase and phrase in name_text)
        score += sum(10_000 for phrase in related_phrases if phrase and phrase in name_text)
        
        score += len(industry_tokens & metadata_tokens) * 100
        score += len(category_tokens & metadata_tokens) * 100
        
        if score > 0:
            ranked.append((score, asset.symbol, asset))
            
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [asset for _, _, asset in ranked[:limit]]

async def build_clustered_routing_world(
    ollama: OllamaClient,
    market: SourceMarket,
    *,
    as_of: datetime,
    catalog: IBAssetCatalogIndex,
    progress: Callable[[str, dict[str, object]], None] | None = None,
) -> tuple[AssetWorld, ClusteredRoutingPlan, list[IBTradableAsset]]:
    available_industries = list({a.industry for a in catalog.assets if a.industry})
    available_categories = list({a.category for a in catalog.assets if a.category})
    
    routing_payload = {
        "event_title": market.event_title,
        "market_question": market.question,
        "tags": market.tags,
        "market_created_at": market.created_at,
        "market_end_at": market.end_at,
        "historical_as_of": as_of,
        "available_ib_industries": available_industries,
        "available_ib_categories": available_categories,
    }
    
    if progress:
        progress("routing-plan-start", {"catalog_assets": len(catalog.assets)})
        
    plan = await ollama.structured(
        system_prompt=CLUSTERED_ROUTER_PROMPT,
        payload=routing_payload,
        response_model=ClusteredRoutingPlan,
        max_tokens=1200,
    )
    
    if progress:
        progress(
            "routing-plan-complete",
            {
                "event_archetype": plan.event_archetype,
                "primary_direct_companies": plan.primary_direct_companies,
                "custom_related_companies": plan.custom_related_companies,
                "impacted_ib_industries": plan.impacted_ib_industries,
                "impacted_ib_categories": plan.impacted_ib_categories,
            },
        )
        
    candidates = retrieve_clustered_candidates(plan, catalog)
    
    if progress:
        progress(
            "candidates-retrieved",
            {
                "count": len(candidates),
                "symbols": [asset.symbol for asset in candidates],
            },
        )
        
    if len(candidates) < 4:
        raise ValueError(
            "Clustered retrieval found fewer than four candidates for "
            f"market {market.market_id}: {len(candidates)}"
        )
        
    response_model = catalog_asset_world_model(candidates)
    selection_payload = {
        **routing_payload,
        "clustered_routing_plan": plan.model_dump(mode="json"),
        "available_ib_assets": [asset.prompt_record() for asset in candidates],
    }
    
    for attempt in range(MAX_INVALID_SYMBOL_RETRIES + 1):
        if progress:
            progress(
                "selection-start",
                {"attempt": attempt + 1, "available_candidates": len(candidates)},
            )
        if attempt:
            selection_payload["schema_retry_instruction"] = (
                "Use only an exact allowed symbol from available_ib_assets."
            )
        try:
            world = await ollama.structured(
                system_prompt=CLUSTERED_SELECTION_PROMPT,
                payload=selection_payload,
                response_model=response_model,
                max_tokens=2200,
            )
        except ValidationError:
            if progress:
                progress("selection-schema-retry", {"attempt": attempt + 1})
            continue
            
        canonical_world = canonicalize_catalog_world(world, candidates)
        if progress:
            progress(
                "selection-complete",
                {"symbols": [asset.symbol for asset in canonical_world.assets]},
            )
        return canonical_world, plan, candidates
        
    raise ValueError(
        "Clustered-routing selection failed the allowed-symbol schema after "
        f"{MAX_INVALID_SYMBOL_RETRIES + 1} attempts"
    )

