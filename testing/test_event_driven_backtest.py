from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from database.backtesting.historical_repository import prior_ml_observations
from database.backtesting.repositories.machine_learning import save_model_snapshot
from database.backtesting.repositories.trades import select_walk_forward_momentum_parameters
from database.backtesting.repositories._shared import json_text
from database.backtesting.market_data import (
    _download_prices,
    benchmark_symbol,
    next_bar_after,
    yahoo_request_bounds,
)
from database.backtesting.repositories.prices import _uncovered_intervals
from database.backtesting.news import (
    GdeltNewsClient,
    GdeltRateLimitError,
    PostgresGdeltSearchLimiter,
)
from database.backtesting.polymarket import PolymarketHistoryClient, hourly_as_of_points
from database.backtesting.repository import candidate_events
from database.backtesting.security_master import (
    SecurityMaster,
    SecurityMasterEntry,
    entries_from_text,
    yfinance_symbol,
)
from LLM.build_world import (
    AssetWorld,
    BatchedAssetWorlds,
    CatalogSearchPlan,
    IBAssetCatalogIndex,
    assets_from_world,
    build_asset_world,
    build_asset_worlds,
    catalog_asset_world_model,
    retrieve_catalog_candidates,
)
from LLM.remove_unwanted_markets import classify_markets, is_speech_word_market
import main_backtesting.engine as engine_module
import main_backtesting.stages.prices as prices_stage
from main_backtesting.config import BacktestConfig
from main_backtesting.engine import (
    HistoricalBacktestEngine,
    detect_passes,
    validate_pipeline_integrity,
)
from main_backtesting.stages import STAGES, STAGE_FUNCTIONS
from main_backtesting.models import (
    Asset,
    IBTradableAsset,
    MLModelSnapshot,
    MLObservation,
    PriceBar,
    ProbabilityPoint,
    SourceMarket,
)
from main_backtesting.reporting import create_probability_graph, create_trade_graph
from main_backtesting.stages.prices import (
    _merge_requests,
    download_and_save_prices,
    write_rejected_asset_log,
)
from main_backtesting.stages.simulation import polymarket_volume_quality, simulate_one_trade
from main_backtesting.utils import event_archetype
from strategies.event_driven_long import (
    EventDrivenLongStrategy,
    average_true_range,
    ib_style_commission,
    rate_of_change,
)
from strategies.event_driven_ml import build_observation, close_as_of, predict, train_snapshot

START = datetime(2026, 1, 2, 14, tzinfo=timezone.utc)


class FakeGdeltScheduleDatabase:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.last_request_started_at: datetime | None = None
        self.last_request_completed_at: datetime | None = None
        self.last_status: int | None = None
        self.request_starts: list[datetime] = []

    async def connect(self) -> FakeGdeltScheduleConnection:
        return FakeGdeltScheduleConnection(self)


class FakeGdeltScheduleConnection:
    def __init__(self, database: FakeGdeltScheduleDatabase) -> None:
        self.database = database

    async def execute(self, query: str, *arguments: object) -> str:
        if "pg_advisory_unlock" in query:
            self.database.lock.release()
        elif "pg_advisory_lock" in query:
            await self.database.lock.acquire()
        elif "last_request_completed_at = clock_timestamp()" in query:
            self.database.last_request_completed_at = datetime.now(timezone.utc)
            self.database.last_status = int(arguments[1])
        return "OK"

    async def fetchrow(self, query: str, *arguments: object) -> dict[str, object]:
        return {
            "last_request_started_at": self.database.last_request_started_at,
            "last_request_completed_at": self.database.last_request_completed_at,
            "database_now": datetime.now(timezone.utc),
        }

    async def fetchval(self, query: str, *arguments: object) -> datetime:
        request_timestamp = datetime.now(timezone.utc)
        self.database.last_request_started_at = request_timestamp
        self.database.request_starts.append(request_timestamp)
        return request_timestamp

    async def close(self) -> None:
        return None


class FakeGdeltHttpClient:
    def __init__(self, statuses: list[int], *, response_delay: float = 0.0) -> None:
        self.statuses = list(statuses)
        self.response_delay = response_delay
        self.calls: list[datetime] = []

    async def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.calls.append(datetime.now(timezone.utc))
        await asyncio.sleep(self.response_delay)
        status = self.statuses.pop(0)
        body = (
            "Please limit requests to one every 5 seconds."
            if status == 429
            else '{"articles": []}'
        )
        return httpx.Response(
            status,
            text=body,
            headers={"server": "GDELT Server"},
            request=httpx.Request("GET", url),
        )

    async def aclose(self) -> None:
        return None


async def gdelt_search(client: GdeltNewsClient) -> list[dict[str, object]]:
    return await client._search(
        query='"financial markets"',
        start=START - timedelta(days=1),
        end=START,
        max_records=1,
    )


def gdelt_rate_limit_error() -> GdeltRateLimitError:
    return GdeltRateLimitError(
        status_code=429,
        response_body="Please limit requests to one every 5 seconds.",
        response_headers={"server": "GDELT Server"},
        request_timestamp=START,
        previous_request_timestamp=START - timedelta(seconds=6),
        previous_completion_timestamp=START - timedelta(seconds=5.5),
        minimum_interval_seconds=5.5,
    )


def probability_points(values: list[float]) -> list[ProbabilityPoint]:
    return [
        ProbabilityPoint(START + timedelta(hours=index), value)
        for index, value in enumerate(values)
    ]


def price_bar(index: int, *, open_price: float = 100.0, high: float = 102.0, low: float = 99.0) -> PriceBar:
    return PriceBar(
        timestamp=START + timedelta(hours=index),
        open=open_price,
        high=high,
        low=low,
        close=101.0,
        volume=1_000.0,
    )


def test_only_the_first_threshold_crossing_is_recorded() -> None:
    passes = detect_passes(
        "market-1",
        probability_points([0.40, 0.56, 0.70, 0.55, 0.54, 0.60, 0.52, 0.80]),
        0.55,
    )

    assert [item.pass_number for item in passes] == [1]
    assert passes[0].above_probability == pytest.approx(0.56)
    assert passes[0].fell_below_probability == pytest.approx(0.55)


def test_market_starting_above_threshold_creates_first_pass() -> None:
    passes = detect_passes("market-1", probability_points([0.80, 0.81]), 0.55)
    assert len(passes) == 1
    assert passes[0].pass_number == 1
    assert passes[0].above_at == START


def test_hourly_probability_never_uses_an_observation_before_it_was_available() -> None:
    points = hourly_as_of_points(
        [
            (START + timedelta(minutes=45), 0.60),
            (START + timedelta(hours=1, minutes=20), 0.70),
        ],
        history_end=START + timedelta(hours=3),
    )
    assert [point.timestamp for point in points] == [
        START + timedelta(hours=1),
        START + timedelta(hours=2),
    ]
    assert points[0].probability == pytest.approx(0.60)
    assert points[0].available_at == START + timedelta(minutes=45)
    assert points[1].probability == pytest.approx(0.70)
    assert all(point.available_at <= point.timestamp for point in points)


def test_hourly_volume_uses_only_the_previous_completed_hour() -> None:
    points = hourly_as_of_points(
        [
            (START + timedelta(minutes=45), 0.60),
            (START + timedelta(hours=1, minutes=20), 0.70),
        ],
        history_end=START + timedelta(hours=3),
        volume_rows=[
            (START + timedelta(minutes=15), 25.0),
            (START + timedelta(minutes=50), 75.0),
            (START + timedelta(hours=1, minutes=10), 200.0),
        ],
    )
    assert points[0].volume_usdc == pytest.approx(100.0)
    assert points[1].volume_usdc == pytest.approx(200.0)


def test_polymarket_volume_quality_requires_real_pre_entry_volume() -> None:
    unavailable = polymarket_volume_quality(
        [ProbabilityPoint(START, 0.60)],
        trigger_at=START,
        minimum_pre_entry_usdc=10.0,
        concentration_minimum_usdc=1_000.0,
        max_single_hour_share=0.95,
    )
    insignificant = polymarket_volume_quality(
        [ProbabilityPoint(START, 0.60, volume_usdc=9.99)],
        trigger_at=START,
        minimum_pre_entry_usdc=10.0,
        concentration_minimum_usdc=1_000.0,
        max_single_hour_share=0.95,
    )

    assert unavailable["reason"] == "polymarket_volume_unavailable"
    assert insignificant["reason"] == "polymarket_volume_not_significant"
    assert not unavailable["allowed"]
    assert not insignificant["allowed"]


def test_polymarket_volume_quality_rejects_only_large_dominating_hour() -> None:
    concentrated = polymarket_volume_quality(
        [
            ProbabilityPoint(START, 0.54, volume_usdc=20.0),
            ProbabilityPoint(START + timedelta(hours=1), 0.60, volume_usdc=1_000.0),
        ],
        trigger_at=START + timedelta(hours=1),
        minimum_pre_entry_usdc=10.0,
        concentration_minimum_usdc=1_000.0,
        max_single_hour_share=0.95,
    )
    healthy = polymarket_volume_quality(
        [
            ProbabilityPoint(START, 0.54, volume_usdc=100.0),
            ProbabilityPoint(START + timedelta(hours=1), 0.60, volume_usdc=1_000.0),
        ],
        trigger_at=START + timedelta(hours=1),
        minimum_pre_entry_usdc=10.0,
        concentration_minimum_usdc=1_000.0,
        max_single_hour_share=0.95,
    )

    assert concentrated["reason"] == "polymarket_volume_single_hour_concentration"
    assert not concentrated["allowed"]
    assert healthy["reason"] == "polymarket_volume_quality_confirmed"
    assert healthy["allowed"]


def test_polymarket_volume_408_is_retried_without_failing_probabilities(monkeypatch) -> None:
    async def run() -> tuple[list[tuple[datetime, float]] | None, str, int]:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            async def get(self, url: str, **kwargs: object) -> httpx.Response:
                self.calls += 1
                status = 408 if self.calls == 1 else 200
                return httpx.Response(
                    status,
                    json=[],
                    request=httpx.Request("GET", url),
                )

        async def no_sleep(*args: object, **kwargs: object) -> None:
            return None

        client = PolymarketHistoryClient()
        await client.client.aclose()
        fake = FakeClient()
        client.client = fake  # type: ignore[assignment]
        monkeypatch.setattr("database.backtesting.polymarket.asyncio.sleep", no_sleep)
        rows = await client._trade_volumes(
            condition_id="condition",
            start=START,
            end=START + timedelta(days=1),
        )
        return rows, client.volume_status, fake.calls

    rows, status, calls = asyncio.run(run())
    assert rows == []
    assert status == "complete"
    assert calls == 2


def test_probability_graph_accepts_double_dollar_market_title(tmp_path) -> None:
    path = create_probability_graph(
        market_id="1334700",
        question="Will S&P 500 (SPX) hit $$6,900 (HIGH) in February 2026?",
        probabilities=[ProbabilityPoint(START, 0.60)],
        passes=[],
        event_end=START + timedelta(days=1),
        final_outcome=None,
        graph_dir=tmp_path,
    )
    assert path.exists()


def asset_world_test_assets() -> list[dict[str, str]]:
    return [
        {
            "symbol": "AAPL",
            "asset_name": "Apple",
            "asset_class": "stock",
            "relationship_type": "direct_company",
            "reason": "Apple is the company directly named by this specific market question.",
        },
        {
            "symbol": "MSFT",
            "asset_name": "Microsoft",
            "asset_class": "stock",
            "relationship_type": "competitor",
            "reason": "Microsoft competes for the same customers and technology spending.",
        },
        {
            "symbol": "AMZN",
            "asset_name": "Amazon",
            "asset_class": "stock",
            "relationship_type": "partner",
            "reason": "Amazon has a material commercial partnership connected to the event.",
        },
        {
            "symbol": "XLK",
            "asset_name": "Technology Select Sector SPDR Fund",
            "asset_class": "etf",
            "relationship_type": "sector_etf",
            "reason": "XLK represents the specific technology-sector transmission channel.",
        },
    ]


def test_asset_world_requires_four_unique_symbols() -> None:
    duplicate_assets = asset_world_test_assets()
    duplicate_assets[-1] = {
        **duplicate_assets[-1],
        "symbol": "AAPL",
        "asset_name": "Apple duplicate",
    }
    with pytest.raises(ValueError, match="duplicate symbols"):
        AssetWorld.model_validate(
            {
                "universe_name": "Test world",
                "universe_reason": "Assets with direct exposure to the supplied market question.",
                "assets": duplicate_assets,
            }
        )


def test_asset_world_batch_retries_only_missing_request_ids() -> None:
    class PartialWorldOllama:
        def __init__(self) -> None:
            self.request_ids: list[list[str]] = []

        async def structured(
            self,
            *,
            payload,
            response_model,
            **kwargs,
        ):
            if response_model is AssetWorld:
                return AssetWorld.model_validate(
                    {
                        "universe_name": "Single retry world",
                        "universe_reason": "Assets with direct exposure to this market question.",
                        "assets": asset_world_test_assets(),
                    }
                )
            request_ids = [item["request_id"] for item in payload["requests"]]
            self.request_ids.append(request_ids)
            return BatchedAssetWorlds.model_validate(
                {
                    "worlds": [
                        {
                            "request_id": request_ids[0],
                            "universe_name": "Partial batch world",
                            "universe_reason": "Assets with direct exposure to this market question.",
                            "assets": asset_world_test_assets(),
                        }
                    ]
                }
            )

    market = SourceMarket(
        market_id="market",
        event_id="event",
        event_title="Event",
        question="Question?",
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["finance"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )
    ollama = PartialWorldOllama()
    worlds = asyncio.run(
        build_asset_worlds(
            ollama,
            [(f"request-{index}", market, START) for index in range(4)],
        )
    )
    assert [world.request_id for world in worlds] == [
        "request-0",
        "request-1",
        "request-2",
        "request-3",
    ]
    assert ollama.request_ids == [
        ["request-0", "request-1", "request-2", "request-3"],
        ["request-1", "request-2", "request-3"],
        ["request-2", "request-3"],
    ]


def test_asset_world_batch_falls_back_to_singles_after_bounded_retries() -> None:
    class DroppingWorldOllama:
        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        async def structured(
            self,
            *,
            response_model,
            **kwargs,
        ):
            if response_model is BatchedAssetWorlds:
                self.batch_calls += 1
                return BatchedAssetWorlds(worlds=[])
            self.single_calls += 1
            return AssetWorld.model_validate(
                {
                    "universe_name": "Single fallback world",
                    "universe_reason": "Assets with direct exposure to this market question.",
                    "assets": asset_world_test_assets(),
                }
            )

    market = SourceMarket(
        market_id="market",
        event_id="event",
        event_title="Event",
        question="Question?",
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["finance"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )
    ollama = DroppingWorldOllama()
    worlds = asyncio.run(
        build_asset_worlds(
            ollama,
            [(f"request-{index}", market, START) for index in range(3)],
        )
    )

    assert [world.request_id for world in worlds] == [
        "request-0",
        "request-1",
        "request-2",
    ]
    assert ollama.batch_calls == 4
    assert ollama.single_calls == 3


def test_asset_world_retries_symbols_outside_ib_tradable_universe() -> None:
    class RetryingWorldOllama:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        async def structured(self, *, payload, **kwargs):
            self.payloads.append(payload)
            assets = asset_world_test_assets()
            if len(self.payloads) == 1:
                assets[0] = {**assets[0], "symbol": "FAKE"}
            return AssetWorld.model_validate(
                {
                    "universe_name": "IB validated world",
                    "universe_reason": "Assets connected through concrete economic relationships.",
                    "assets": assets,
                }
            )

    market = SourceMarket(
        market_id="market",
        event_id="event",
        event_title="Event",
        question="Question?",
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["finance"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )
    ollama = RetryingWorldOllama()
    world = asyncio.run(
        build_asset_world(
            ollama,
            market,
            as_of=START,
            tradable_symbols={"AAPL", "MSFT", "AMZN", "XLK"},
        )
    )

    assert [asset.symbol for asset in world.assets] == ["AAPL", "MSFT", "AMZN", "XLK"]
    assert ollama.payloads[1]["rejected_symbols_not_in_ib_tradable_universe"] == ["FAKE"]
    assert assets_from_world(world)[0].reason.startswith("[direct_company]")


def test_asset_world_final_selection_uses_supplied_ib_asset_catalog() -> None:
    class CatalogAwareOllama:
        def __init__(self) -> None:
            self.catalog_payload: dict[str, object] | None = None

        async def structured(self, *, payload, **kwargs):
            if "available_ib_assets" not in payload:
                if "rejected_symbols_not_in_ib_tradable_universe" in payload:
                    return AssetWorld.model_validate(
                        {
                            "universe_name": "Corrected discovered relationships",
                            "universe_reason": "Relationships corrected to use IB-confirmed assets.",
                            "assets": asset_world_test_assets(),
                        }
                    )
                assets = asset_world_test_assets()
                assets[0] = {
                    **assets[0],
                    "symbol": "FAKE",
                    "asset_name": "Apple",
                }
                return AssetWorld.model_validate(
                    {
                        "universe_name": "Discovered relationships",
                        "universe_reason": "Potential relationships discovered before IB validation.",
                        "assets": assets,
                    }
                )
            self.catalog_payload = payload
            return AssetWorld.model_validate(
                {
                    "universe_name": "Catalog selected world",
                    "universe_reason": "Final assets selected only from the supplied IB catalog.",
                    "assets": [
                        {
                            **asset,
                            "asset_name": "LLM-provided name is replaced",
                            "relationship_type": "direct_company",
                            "reason": "Generic final-selector reason that must not replace discovery.",
                        }
                        for asset in asset_world_test_assets()
                    ],
                }
            )

    market = SourceMarket(
        market_id="market",
        event_id="event",
        event_title="Will Apple beat quarterly earnings?",
        question="Will Apple beat quarterly earnings?",
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["earnings", "tech"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )
    catalog = [
        IBTradableAsset("AAPL", "Apple Inc.", "stock", "NASDAQ", "COMMON", "Technology"),
        IBTradableAsset("MSFT", "Microsoft Corporation", "stock", "NASDAQ", "COMMON", "Technology"),
        IBTradableAsset("AMZN", "Amazon.com, Inc.", "stock", "NASDAQ", "COMMON", "Retail"),
        IBTradableAsset("XLK", "Technology Select Sector SPDR Fund", "etf", "ARCA", "ETF"),
        IBTradableAsset("NVDA", "NVIDIA Corporation", "stock", "NASDAQ", "COMMON", "Technology"),
    ]
    ollama = CatalogAwareOllama()
    worlds = asyncio.run(
        build_asset_worlds(
            ollama,
            [("request", market, START)],
            tradable_assets=catalog,
        )
    )

    assert ollama.catalog_payload is not None
    available = ollama.catalog_payload["available_ib_assets"]
    assert {item["symbol"] for item in available} <= {asset.symbol for asset in catalog}
    assert "FAKE" not in {item["symbol"] for item in available}
    assert [asset.symbol for asset in worlds[0].assets] == ["AAPL", "MSFT", "AMZN", "XLK"]
    assert worlds[0].assets[0].asset_name == "Apple Inc."
    assert worlds[0].assets[-1].asset_class == "etf"
    assert worlds[0].assets[1].relationship_type == "competitor"
    assert worlds[0].assets[1].reason == asset_world_test_assets()[1]["reason"]


def test_catalog_asset_world_schema_rejects_symbol_outside_available_assets() -> None:
    catalog = [
        IBTradableAsset("AAPL", "Apple Inc.", "stock", "NASDAQ", "COMMON"),
        IBTradableAsset("MSFT", "Microsoft Corporation", "stock", "NASDAQ", "COMMON"),
        IBTradableAsset("AMZN", "Amazon.com, Inc.", "stock", "NASDAQ", "COMMON"),
        IBTradableAsset("XLK", "Technology Select Sector SPDR Fund", "etf", "ARCA", "ETF"),
    ]
    schema = catalog_asset_world_model(catalog)
    invalid_assets = asset_world_test_assets()
    invalid_assets[0] = {**invalid_assets[0], "symbol": "FAKE"}

    with pytest.raises(ValueError):
        schema.model_validate(
            {
                "universe_name": "Invalid catalog world",
                "universe_reason": "This world contains a symbol outside the supplied catalog.",
                "assets": invalid_assets,
            }
        )


def test_full_catalog_retrieval_prioritizes_llm_search_plan_entities() -> None:
    catalog = IBAssetCatalogIndex(
        [
            IBTradableAsset(
                "GOOGL",
                "Alphabet Inc. - Class A Common Stock",
                "stock",
                "NASDAQ",
                "COMMON",
                "Communications",
                "Internet",
                "Web Portals/ISP",
            ),
            IBTradableAsset(
                "MSFT",
                "Microsoft Corporation - Common Stock",
                "stock",
                "NASDAQ",
                "COMMON",
                "Technology",
                "Software",
                "Applications Software",
            ),
            IBTradableAsset(
                "AMZN",
                "Amazon.com, Inc. - Common Stock",
                "stock",
                "NASDAQ",
                "COMMON",
                "Communications",
                "Internet",
                "E-Commerce/Products",
            ),
            IBTradableAsset(
                "CLOU",
                "Global X Cloud Computing ETF",
                "etf",
                "NASDAQ",
                "ETF",
            ),
            IBTradableAsset(
                "NKE",
                "Nike, Inc. Common Stock",
                "stock",
                "NYSE",
                "COMMON",
                "Consumer, Cyclical",
                "Apparel",
                "Athletic Footwear",
            ),
        ]
    )
    market = SourceMarket(
        market_id="market",
        event_id="event",
        event_title="Alphabet cloud revenue",
        question="Will Alphabet cloud revenue beat expectations?",
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["earnings", "tech"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )
    plan = CatalogSearchPlan(
        direct_entities=["Alphabet"],
        related_entities=["Microsoft", "Amazon"],
        ticker_hints=["GOOGL"],
        industries=["Cloud software"],
        etf_themes=["Cloud computing"],
        economic_keywords=["enterprise cloud"],
        rationale="Search direct cloud companies, competitors, and focused cloud ETFs.",
    )

    candidates = retrieve_catalog_candidates(market, plan, catalog)

    assert candidates[0].symbol == "GOOGL"
    assert {asset.symbol for asset in candidates[:4]} == {"GOOGL", "AMZN", "MSFT", "CLOU"}
    assert "NKE" not in {asset.symbol for asset in candidates}


def test_speech_word_markets_are_rejected_without_calling_llm() -> None:
    class NoCallOllama:
        async def structured(self, **kwargs):
            raise AssertionError("Speech-word markets must not reach the LLM")

    question = 'Will Powell say "inflation" 40 times during the press conference?'
    assert is_speech_word_market(question)
    assert not is_speech_word_market("Will Alphabet (GOOGL) beat quarterly earnings?")
    market = SourceMarket(
        market_id="speech-market",
        event_id="speech-event",
        event_title=question,
        question=question,
        created_at=START,
        end_at=START + timedelta(days=10),
        tags=["fed"],
        raw_market={},
        yes_token_id="yes-token",
        condition_id="condition-id",
        final_outcome=None,
    )

    decisions = asyncio.run(classify_markets(NoCallOllama(), [market]))

    assert len(decisions) == 1
    assert decisions[0].relevant_to_financial_markets is False
    assert "Speech-word" in decisions[0].reason


def test_security_master_builds_hash_and_resolves_ticker_variants() -> None:
    nasdaq_text = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. Common Stock|Q|N|N|100|N|N
AMZN|Amazon.com, Inc. Common Stock|Q|N|N|100|N|N
CMG|Chipotle Mexican Grill, Inc. Common Stock|Q|N|N|100|N|N
TESTZ|Test Security|Q|Y|N|100|N|N
File Creation Time: 0608202621:31|||||||
"""
    other_text = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
MMM|3M Company Common Stock|N|MMM|N|100|N|MMM
BRK.B|Berkshire Hathaway Inc. Class B Common Stock|N|BRK.B|N|100|N|BRK.B
AFL|Aflac Incorporated Common Stock|N|AFL|N|100|N|AFL
BA|Boeing Company (The) Common Stock|N|BA|N|100|N|BA
File Creation Time: 0608202621:31|||||||
"""
    entries = entries_from_text(nasdaq_text, other_text)
    master = SecurityMaster(entries)

    assert len(entries) == 7
    assert master.resolve("AAPL").entry.yfinance_symbol == "AAPL"
    assert master.resolve("3M").entry.yfinance_symbol == "MMM"
    assert master.resolve("3M (MMM)").entry.yfinance_symbol == "MMM"
    assert master.resolve("AFLAC (AFL").entry.yfinance_symbol == "AFL"
    assert master.resolve("AMZN.O").entry.yfinance_symbol == "AMZN"
    assert master.resolve(
        "AMAZON.COM, INC. (",
        asset_names=["Amazon.com, Inc."],
    ).entry.yfinance_symbol == "AMZN"
    assert master.resolve(
        "CHIPOTLE MEXICAN (CM",
        asset_names=["Chipotle Mexican Grill, Inc."],
    ).entry.yfinance_symbol == "CMG"
    assert master.resolve("BRK.B").entry.yfinance_symbol == "BRK-B"
    assert master.resolve("BAESY", asset_names=["Boeing Company"]).entry is None
    assert master.resolve("504249:1").rejection_reason == "invalid_symbol_format"
    assert master.resolve("BTC/USD").rejection_reason == "not_in_us_security_master"


def test_yfinance_symbol_normalizes_us_share_classes() -> None:
    assert yfinance_symbol("BRK.B") == "BRK-B"
    assert yfinance_symbol("BAC^A") == "BAC-PA"


def test_rejected_asset_log_contains_only_unresolved_symbols(tmp_path) -> None:
    path = tmp_path / "logs" / "rejected_asset_symbols.csv"
    write_rejected_asset_log(
        path,
        [
            {
                "original_symbol": "AAPL",
                "resolved_symbol": "AAPL",
                "rejection_reason": None,
            },
            {
                "original_symbol": "504249:1",
                "resolved_symbol": None,
                "rejection_reason": "not_in_us_security_master",
            },
        ],
    )

    assert path.read_text(encoding="utf-8").splitlines() == [
        "original_symbol,rejection_reason",
        "504249:1,not_in_us_security_master",
    ]


def test_empty_price_window_is_retried_saved_and_does_not_fail(monkeypatch) -> None:
    async def run() -> tuple[dict[str, int], list[dict[str, object]], int]:
        saved: list[dict[str, object]] = []

        class EmptyPriceClient:
            def __init__(self) -> None:
                self.calls = 0

            async def bars(self, *args: object, **kwargs: object) -> list[PriceBar]:
                self.calls += 1
                return []

        async def missing(*args: object, **kwargs: object):
            return [(kwargs["start"], kwargs["end"])]

        async def save(*args: object, **kwargs: object) -> None:
            saved.append(kwargs)

        client = EmptyPriceClient()
        engine = SimpleNamespace(
            config=SimpleNamespace(price_download_concurrency=4),
            prices=client,
        )
        monkeypatch.setattr(prices_stage, "missing_price_windows", missing)
        monkeypatch.setattr(prices_stage, "save_price_bars", save)
        result = await download_and_save_prices(
            engine,
            object(),
            [("EGG", "1d", START, START + timedelta(days=10))],
        )
        return result, saved, client.calls

    result, saved, calls = asyncio.run(run())

    assert calls == 3
    assert result == {
        "pending_request_count": 1,
        "downloaded_window_count": 1,
        "no_data_window_count": 1,
        "retryable_failure_count": 0,
    }
    assert len(saved) == 1
    assert saved[0]["symbol"] == "EGG"
    assert saved[0]["bars"] == []


def test_yahoo_request_bounds_preserve_exact_exclusive_window() -> None:
    start = datetime(2025, 2, 12, 21, 43, 39, 900510, tzinfo=timezone.utc)
    end = datetime(2025, 4, 10, 12, 0, 0, 68895, tzinfo=timezone.utc)

    request_start, request_end = yahoo_request_bounds(start, end)

    assert request_start == datetime(2025, 2, 12, 21, 43, 39, tzinfo=timezone.utc)
    assert request_end == datetime(2025, 4, 10, 12, 0, 1, tzinfo=timezone.utc)


def test_yfinance_receives_exact_aware_datetimes_without_extra_day(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_download(symbol: str, **kwargs: object):
        captured["symbol"] = symbol
        captured.update(kwargs)
        return __import__("pandas").DataFrame()

    monkeypatch.setattr("yfinance.download", fake_download)
    start = datetime(2025, 2, 12, 21, 43, 39, 900510, tzinfo=timezone.utc)
    end = datetime(2025, 4, 10, 12, 0, tzinfo=timezone.utc)

    assert _download_prices("EGG", start, end, "1h") == []
    assert captured["start"] == datetime(2025, 2, 12, 21, 43, 39, tzinfo=timezone.utc)
    assert captured["end"] == datetime(2025, 4, 10, 12, 0, tzinfo=timezone.utc)
    assert captured["ignore_tz"] is False


def test_concurrent_gdelt_searches_are_spaced_and_serialized_globally() -> None:
    async def run() -> list[datetime]:
        database = FakeGdeltScheduleDatabase()
        http_client = FakeGdeltHttpClient([200, 200, 200], response_delay=0.005)
        client = GdeltNewsClient(
            minimum_search_interval_seconds=0.02,
            search_limiter=PostgresGdeltSearchLimiter(
                0.02,
                connection_factory=database.connect,
            ),
            http_client=http_client,
        )
        await asyncio.gather(gdelt_search(client), gdelt_search(client), gdelt_search(client))
        await client.close()
        return http_client.calls

    starts = asyncio.run(run())
    assert starts[1] - starts[0] >= timedelta(seconds=0.02)
    assert starts[2] - starts[1] >= timedelta(seconds=0.02)


def test_separate_gdelt_client_instances_cannot_violate_global_limiter() -> None:
    async def run() -> list[datetime]:
        database = FakeGdeltScheduleDatabase()
        first_http = FakeGdeltHttpClient([200])
        second_http = FakeGdeltHttpClient([200])
        first = GdeltNewsClient(
            minimum_search_interval_seconds=0.02,
            search_limiter=PostgresGdeltSearchLimiter(
                0.02,
                connection_factory=database.connect,
            ),
            http_client=first_http,
        )
        second = GdeltNewsClient(
            minimum_search_interval_seconds=0.02,
            search_limiter=PostgresGdeltSearchLimiter(
                0.02,
                connection_factory=database.connect,
            ),
            http_client=second_http,
        )
        await asyncio.gather(gdelt_search(first), gdelt_search(second))
        await asyncio.gather(first.close(), second.close())
        return sorted(first_http.calls + second_http.calls)

    starts = asyncio.run(run())
    assert starts[1] - starts[0] >= timedelta(seconds=0.02)


def test_gdelt_429_has_diagnostics_and_is_not_retried() -> None:
    async def run() -> tuple[GdeltRateLimitError, int]:
        database = FakeGdeltScheduleDatabase()
        database.last_request_started_at = START - timedelta(days=1)
        database.last_request_completed_at = START - timedelta(days=1)
        http_client = FakeGdeltHttpClient([429, 200])
        client = GdeltNewsClient(
            minimum_search_interval_seconds=5.5,
            search_limiter=PostgresGdeltSearchLimiter(
                5.5,
                connection_factory=database.connect,
            ),
            http_client=http_client,
        )
        with pytest.raises(GdeltRateLimitError) as captured:
            await gdelt_search(client)
        await client.close()
        return captured.value, len(http_client.calls)

    error, call_count = asyncio.run(run())
    assert call_count == 1
    assert "HTTP status=429" in str(error)
    assert "Please limit requests to one every 5 seconds." in str(error)
    assert "request_timestamp=" in str(error)
    assert "previous_known_gdelt_request_timestamp=" in str(error)
    assert "configured_minimum_interval_seconds=5.5" in str(error)


def test_stage_order_matches_required_pipeline() -> None:
    assert STAGES == [
        "event_filter",
        "probabilities",
        "asset_worlds",
        "prices",
        "ml_observations",
        "simulation",
        "reports",
    ]
    assert [function.__module__.rsplit(".", 1)[-1] for function in STAGE_FUNCTIONS] == STAGES


def test_stop_after_stage_marks_run_paused_and_stops_execution(tmp_path, monkeypatch) -> None:
    async def run() -> tuple[list[str], list[tuple[str, str | None]]]:
        executed: list[str] = []
        updates: list[tuple[str, str | None]] = []
        engine = HistoricalBacktestEngine.__new__(HistoricalBacktestEngine)
        engine.config = BacktestConfig(output_root=tmp_path)
        engine.run_id = uuid4()
        engine.stop_after_stage = "probabilities"
        engine.hourly_boundary = START
        engine.run_dir = tmp_path / str(engine.run_id)
        engine.current_work_key = None
        engine.ollama = SimpleNamespace(model_name="test")

        class FakeConnection:
            async def close(self) -> None:
                return None

        async def close() -> None:
            return None

        async def fake_connect() -> FakeConnection:
            return FakeConnection()

        async def no_op(*args: object, **kwargs: object) -> None:
            return None

        async def latest_batch_sizes(*args: object, **kwargs: object) -> dict[str, int]:
            return {}

        async def update_run(
            conn: object,
            run_id: object,
            *,
            status: str,
            stage: str | None = None,
            error: str | None = None,
        ) -> None:
            updates.append((status, stage))

        def stage_function(name: str):
            async def execute(engine: object, conn: object) -> None:
                executed.append(name)

            return execute

        engine.close = close
        monkeypatch.setattr(engine_module, "connect", fake_connect)
        monkeypatch.setattr(engine_module, "initialize_historical_schema", no_op)
        monkeypatch.setattr(engine_module, "create_historical_run", no_op)
        monkeypatch.setattr(engine_module, "latest_batch_sizes", latest_batch_sizes)
        monkeypatch.setattr(engine_module, "update_run", update_run)
        monkeypatch.setattr(
            engine_module,
            "STAGE_FUNCTIONS",
            [stage_function(name) for name in STAGES],
        )

        await engine.run()
        return executed, updates

    executed, updates = asyncio.run(run())
    assert executed == STAGES[: STAGES.index("probabilities") + 1]
    assert updates[-1] == ("paused", "probabilities")
    assert ("complete", "complete") not in updates


def test_resume_loads_saved_config_and_executes_stages_in_order(tmp_path, monkeypatch) -> None:
    async def run() -> tuple[list[str], BacktestConfig, list[str]]:
        executed: list[str] = []
        lifecycle: list[str] = []
        stored_config = BacktestConfig(
            output_root=tmp_path / "stored",
            selected_event_ids=("saved-event",),
        )
        engine = HistoricalBacktestEngine.__new__(HistoricalBacktestEngine)
        engine.config = BacktestConfig(output_root=tmp_path / "placeholder")
        engine.run_id = uuid4()
        engine.stop_after_stage = None
        engine.hourly_boundary = START - timedelta(days=1)
        engine.run_dir = engine.config.run_dir(engine.run_id)
        engine.current_work_key = None

        class FakeConnection:
            async def close(self) -> None:
                lifecycle.append("database_closed")

        async def close() -> None:
            lifecycle.append("clients_closed")

        async def fake_connect() -> FakeConnection:
            return FakeConnection()

        async def no_op(*args: object, **kwargs: object) -> None:
            return None

        async def historical_run(conn: object, run_id: object) -> dict[str, object]:
            lifecycle.append("loaded_run")
            return {
                "config": stored_config.to_json(),
                "hourly_boundary": START,
                "output_dir": str(stored_config.run_dir(engine.run_id)),
            }

        def stage_function(name: str):
            async def execute(engine: object, conn: object) -> None:
                executed.append(name)

            return execute

        engine.close = close
        monkeypatch.setattr(engine_module, "connect", fake_connect)
        monkeypatch.setattr(engine_module, "initialize_historical_schema", no_op)
        monkeypatch.setattr(engine_module, "historical_run", historical_run)
        monkeypatch.setattr(engine_module, "update_run", no_op)
        monkeypatch.setattr(
            engine_module,
            "STAGE_FUNCTIONS",
            [stage_function(name) for name in STAGES],
        )

        await engine.run(resume=True)
        return executed, engine.config, lifecycle

    executed, restored_config, lifecycle = asyncio.run(run())
    assert executed == STAGES
    assert restored_config.selected_event_ids == ("saved-event",)
    assert lifecycle == ["loaded_run", "database_closed", "clients_closed"]


def test_repository_modules_and_compatibility_facade_export_same_functions() -> None:
    from database.backtesting import historical_repository
    from database.backtesting.repositories import (
        prior_ml_observations as package_prior_ml_observations,
    )
    from database.backtesting.repositories.machine_learning import (
        prior_ml_observations as direct_prior_ml_observations,
    )

    assert historical_repository.prior_ml_observations is direct_prior_ml_observations
    assert package_prior_ml_observations is direct_prior_ml_observations


def test_pipeline_cannot_finish_when_candidates_skip_simulation() -> None:
    with pytest.raises(RuntimeError, match="not every asset candidate reached simulation"):
        validate_pipeline_integrity(
            pass_count=3,
            asset_candidate_count=10,
            completed_simulation_count=0,
        )


def test_zero_trades_is_allowed_only_after_every_candidate_reaches_simulation() -> None:
    validate_pipeline_integrity(
        pass_count=3,
        asset_candidate_count=10,
        completed_simulation_count=10,
    )


def test_smoke_test_event_selection_is_saved_in_config() -> None:
    config = BacktestConfig(
        maximum_events=2,
        selected_event_ids=("event-1", "event-2"),
    )
    restored = BacktestConfig.from_json(config.to_json())

    assert restored.maximum_events == 2
    assert restored.selected_event_ids == ("event-1", "event-2")


def test_candidate_events_require_between_five_and_sixty_days() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.query = ""
            self.arguments: tuple[object, ...] = ()

        async def fetch(self, query: str, *arguments: object) -> list[object]:
            self.query = query
            self.arguments = arguments
            return []

    connection = FakeConnection()
    asyncio.run(
        candidate_events(
            connection,
            start=datetime(2026, 1, 1, tzinfo=timezone.utc),
            end=datetime(2026, 2, 1, tzinfo=timezone.utc),
            minimum_days_remaining=5.0,
            maximum_days_remaining=60.0,
            included_tags=["finance"],
            excluded_tags=["crypto"],
        )
    )

    assert "end_at > created_at + ($3 * INTERVAL '1 day')" in connection.query
    assert "end_at <= created_at + ($4 * INTERVAL '1 day')" in connection.query
    assert connection.arguments[2:4] == (5.0, 60.0)


def test_ib_style_commission_uses_minimum_and_cap() -> None:
    assert ib_style_commission(10, 1_000) == pytest.approx(1.0)
    assert ib_style_commission(10_000, 100) == pytest.approx(1.0)
    assert ib_style_commission(1_000, 100_000) == pytest.approx(5.0)


def test_stop_distance_uses_previous_fourteen_completed_true_ranges() -> None:
    bars = [
        PriceBar(
            timestamp=START + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000.0,
        )
        for index in range(15)
    ]
    assert average_true_range(bars, 14) == pytest.approx(2.0)
    assert average_true_range(bars[:14], 14) is None


def test_true_range_includes_gap_from_previous_close() -> None:
    bars = [
        PriceBar(
            timestamp=START + timedelta(hours=index),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000.0,
        )
        for index in range(14)
    ]
    bars.append(
        PriceBar(
            timestamp=START + timedelta(hours=14),
            open=110.0,
            high=111.0,
            low=109.0,
            close=110.0,
            volume=1_000.0,
        )
    )
    assert average_true_range(bars, 14) == pytest.approx((13 * 2 + 11) / 14)


def test_rate_of_change_requires_positive_completed_bar_momentum() -> None:
    rising = [
        PriceBar(
            timestamp=START + timedelta(hours=index),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100 + index,
            volume=1_000,
        )
        for index in range(15)
    ]
    assert rate_of_change(rising, 14) == pytest.approx(0.14)
    assert rate_of_change(rising[:14], 14) is None


def test_corrected_parameter_grid_and_semantic_corporate_archetype() -> None:
    config = BacktestConfig()
    assert config.momentum_lookback_grid == (5, 7, 12, 14, 18, 21)
    assert config.trailing_range_multiplier_grid == (1.5, 2.0, 2.5, 3.0)
    assert config.trailing_range_bars == 14
    assert config.trailing_range_multiplier == 3.0
    question = "Will Alphabet (GOOGL) beat quarterly earnings estimates?"
    assert event_archetype(["earnings"], question=question, symbol="GOOGL") == (
        "company_quarterly_earnings_beat"
    )
    assert event_archetype(["earnings"], question=question, symbol="NFLX") is None
    assert event_archetype(["finance"], question="Will something happen?", symbol="SPY") is None
    assert benchmark_symbol("TNA", quote_type="ETF", sector=None) == "IWM"


def test_walk_forward_momentum_selection_uses_only_prior_completed_results() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.query = ""
            self.arguments: tuple[object, ...] = ()

        async def fetchrow(self, query: str, *arguments: object):
            self.query = query
            self.arguments = arguments
            return {
                "range_period": 18,
                "range_multiplier": 2.5,
                "sample_count": 42,
                "average_net_profit": 3.25,
            }

    connection = FakeConnection()
    selected = asyncio.run(
        select_walk_forward_momentum_parameters(
            connection,
            run_id=uuid4(),
            before=START,
            resolution="1h",
            minimum_samples=30,
            fallback_period=14,
            fallback_multiplier=3.0,
        )
    )

    assert selected["range_period"] == 18
    assert selected["range_multiplier"] == pytest.approx(2.5)
    assert selected["selection_method"] == "walk_forward_best_prior_net_expectancy"
    assert "trigger_at < $3" in connection.query
    assert connection.arguments[2] == START


def test_historical_company_symbol_alias_does_not_resolve_to_reused_ticker() -> None:
    master = SecurityMaster(
        [
            SecurityMasterEntry("FB", "FB", "Some Current ETF", "NASDAQ", True, "test"),
            SecurityMasterEntry(
                "META",
                "META",
                "Meta Platforms Inc. Class A Common Stock",
                "NASDAQ",
                False,
                "test",
            ),
        ]
    )

    resolution = master.resolve("FB", asset_names=["Facebook"])

    assert resolution.entry is not None
    assert resolution.entry.yfinance_symbol == "META"
    assert resolution.match_method == "historical_symbol_alias"


def test_json_text_converts_non_finite_numbers_to_null() -> None:
    encoded = json_text(
        {
            "positive_infinity": float("inf"),
            "negative_infinity": float("-inf"),
            "not_a_number": float("nan"),
            "nested": [1.0, float("inf")],
        }
    )

    assert json.loads(encoded) == {
        "positive_infinity": None,
        "negative_infinity": None,
        "not_a_number": None,
        "nested": [1.0, None],
    }
    assert "Infinity" not in encoded
    assert "NaN" not in encoded


def test_candidate_price_windows_are_merged_without_downloading_full_backtest() -> None:
    requests = [
        ("TEST", "1d", START, START + timedelta(days=10)),
        ("TEST", "1d", START + timedelta(days=5), START + timedelta(days=20)),
        ("TEST", "1h", START + timedelta(days=1), START + timedelta(days=2)),
    ]
    assert _merge_requests(requests) == [
        ("TEST", "1d", START, START + timedelta(days=20)),
        ("TEST", "1h", START + timedelta(days=1), START + timedelta(days=2)),
    ]


def test_db_first_coverage_requests_only_missing_price_intervals() -> None:
    assert _uncovered_intervals(
        START,
        START + timedelta(days=10),
        [
            (START, START + timedelta(days=3)),
            (START + timedelta(days=5), START + timedelta(days=8)),
        ],
    ) == [
        (START + timedelta(days=3), START + timedelta(days=5)),
        (START + timedelta(days=8), START + timedelta(days=10)),
    ]


def test_entry_uses_strictly_next_available_bar() -> None:
    current = price_bar(0)
    following = price_bar(1)
    assert next_bar_after([current, following], current.timestamp) == following


def test_machine_learning_trade_locks_predicted_target_then_exits_on_reversal(tmp_path) -> None:
    async def run():
        config = BacktestConfig(
            output_root=tmp_path,
            trailing_range_bars=14,
            trailing_range_multiplier=3,
        )
        fake_engine = SimpleNamespace(
            config=config,
            run_id=uuid4(),
            run_dir=tmp_path,
            strategy=EventDrivenLongStrategy(
                range_period=14,
                range_multiplier=3,
            ),
        )
        bars = [
            PriceBar(
                timestamp=START + timedelta(hours=index),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1_000,
            )
            for index in range(15)
        ]
        bars.append(
            PriceBar(
                timestamp=START + timedelta(hours=15),
                open=100,
                high=104,
                low=99,
                close=103,
                volume=1_000,
            )
        )
        bars.append(
            PriceBar(
                timestamp=START + timedelta(hours=16),
                open=104,
                high=105,
                low=102,
                close=103,
                volume=1_000,
            )
        )
        market = SourceMarket(
            market_id="market-target",
            event_id="event-target",
            event_title="Event",
            question="Question?",
            created_at=START,
            end_at=START + timedelta(days=10),
            tags=["finance"],
            raw_market={},
            yes_token_id="yes-token",
            condition_id="condition-id",
            final_outcome="Yes",
        )
        trade, _ = await simulate_one_trade(
            fake_engine,
            market=market,
            asset=Asset("TEST", "Test Asset", "stock", "Direct exposure"),
            pass_number=1,
            trigger_at=START + timedelta(hours=14),
            portfolio="machine_learning",
            strategy_branch="machine_learning",
            direction="long",
            resolution="1h",
            bars=bars,
            probabilities=[ProbabilityPoint(START + timedelta(hours=14), 0.60)],
            predicted_target_price=103.0,
            evaluation_event_end=market.end_at,
        )
        return trade

    trade = asyncio.run(run())
    assert trade is not None
    assert trade.exit_reason == "ml_predicted_target_lock"
    assert trade.exit_price == pytest.approx(103.0)


def test_machine_learning_trade_uses_atr_trailing_stop(tmp_path) -> None:
    async def run():
        fake_engine = SimpleNamespace(
            config=BacktestConfig(output_root=tmp_path),
            run_id=uuid4(),
            run_dir=tmp_path,
            strategy=EventDrivenLongStrategy(range_period=14, range_multiplier=3),
        )
        bars = [
            PriceBar(
                timestamp=START + timedelta(hours=index),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1_000,
            )
            for index in range(16)
        ]
        bars.append(
            PriceBar(
                timestamp=START + timedelta(hours=16),
                open=100,
                high=101,
                low=90,
                close=95,
                volume=1_000,
            )
        )
        market = SourceMarket(
            market_id="market-ml-stop",
            event_id="event-ml-stop",
            event_title="Event",
            question="Question?",
            created_at=START,
            end_at=START + timedelta(days=10),
            tags=["finance"],
            raw_market={},
            yes_token_id="yes-token",
            condition_id="condition-id",
            final_outcome="Yes",
        )
        trade, _ = await simulate_one_trade(
            fake_engine,
            market=market,
            asset=Asset("TEST", "Test Asset", "stock", "Direct exposure"),
            pass_number=1,
            trigger_at=START + timedelta(hours=14),
            portfolio="machine_learning",
            strategy_branch="machine_learning",
            direction="long",
            resolution="1h",
            bars=bars,
            probabilities=[ProbabilityPoint(START + timedelta(hours=14), 0.60)],
            predicted_target_price=200.0,
            evaluation_event_end=market.end_at,
        )
        return trade

    trade = asyncio.run(run())

    assert trade is not None
    assert trade.exit_reason == "trailing_stop"
    assert trade.exit_at == START + timedelta(hours=16)


def test_ml_remaining_gap_is_direction_aware() -> None:
    def snapshot(*, classifier_intercept: float, ridge_intercept: float) -> MLModelSnapshot:
        feature_names = [
            "asset_ytd_change",
            "sector_one_month_trend",
            "spy_two_week_trend",
            "asset_two_week_trend",
        ]
        return MLModelSnapshot(
            snapshot_id=uuid4(),
            run_id=uuid4(),
            symbol="TEST",
            event_archetype="macro_rates",
            training_cutoff=START,
            training_event_ids=[],
            training_sample_count=8,
            status="trained",
            feature_names=feature_names,
            feature_means={name: 0.0 for name in feature_names},
            feature_scales={name: 1.0 for name in feature_names},
            classifier_coefficients={name: 0.0 for name in feature_names},
            classifier_intercept=classifier_intercept,
            ridge_coefficients={name: 0.0 for name in feature_names},
            ridge_intercept=ridge_intercept,
            hyperparameters={},
            validation_metrics={},
        )

    features = {
        "asset_ytd_change": 0.0,
        "sector_one_month_trend": 0.0,
        "spy_two_week_trend": 0.0,
        "asset_two_week_trend": 0.0,
    }
    long_prediction = predict(
        snapshot(classifier_intercept=10.0, ridge_intercept=0.10),
        run_id=uuid4(),
        market_id="market",
        event_id="event",
        pass_number=1,
        symbol="TEST",
        features=features,
        event_open_price=100.0,
        realized_price_at_entry=105.0,
    )
    short_prediction = predict(
        snapshot(classifier_intercept=-10.0, ridge_intercept=-0.10),
        run_id=uuid4(),
        market_id="market",
        event_id="event",
        pass_number=1,
        symbol="TEST",
        features=features,
        event_open_price=100.0,
        realized_price_at_entry=105.0,
    )

    assert long_prediction is not None
    assert long_prediction.remaining_gap == pytest.approx(0.05)
    assert short_prediction is not None
    assert short_prediction.remaining_gap == pytest.approx(0.15)


def test_machine_learning_trade_exits_one_day_before_market_end(tmp_path) -> None:
    async def run():
        config = BacktestConfig(output_root=tmp_path)
        fake_engine = SimpleNamespace(
            config=config,
            run_id=uuid4(),
            run_dir=tmp_path,
            strategy=EventDrivenLongStrategy(range_period=14, range_multiplier=3),
        )
        bars = [
            PriceBar(
                timestamp=START + timedelta(days=index),
                open=100,
                high=101,
                low=99,
                close=100,
                volume=1_000,
            )
            for index in range(20)
        ]
        market = SourceMarket(
            market_id="market-deadline",
            event_id="event-deadline",
            event_title="Event",
            question="Question?",
            created_at=START,
            end_at=START + timedelta(days=20),
            tags=["finance"],
            raw_market={},
            yes_token_id="yes-token",
            condition_id="condition-id",
            final_outcome="Yes",
        )
        trade, _ = await simulate_one_trade(
            fake_engine,
            market=market,
            asset=Asset("TEST", "Test Asset", "stock", "Direct exposure"),
            pass_number=1,
            trigger_at=START + timedelta(days=14),
            portfolio="machine_learning",
            strategy_branch="machine_learning",
            direction="long",
            resolution="1d",
            bars=bars,
            probabilities=[ProbabilityPoint(START + timedelta(days=14), 0.60)],
            predicted_target_price=200.0,
            evaluation_event_end=market.end_at,
        )
        return trade

    trade = asyncio.run(run())
    assert trade is not None
    assert trade.exit_reason == "one_day_before_market_end"
    assert trade.exit_at == START + timedelta(days=19)


def test_trade_is_fractional_and_only_trailing_stop_closes_it() -> None:
    strategy = EventDrivenLongStrategy(
        trade_notional=1_000,
        range_period=14,
        range_multiplier=3,
    )
    previous = [price_bar(index, high=101.0, low=99.0) for index in range(15)]
    entry = price_bar(15, open_price=125.0, high=126.0, low=124.0)
    trade = strategy.open_trade(
        run_id=uuid4(),
        market_id="market-1",
        event_id="event-1",
        question="Question?",
        symbol="TEST",
        asset_name="Test Asset",
        pass_number=1,
        trigger_at=START,
        entry_bar=entry,
        previous_bars=previous,
        final_outcome="No",
    )

    assert trade is not None
    assert trade.quantity == pytest.approx(8.0)
    assert trade.final_outcome == "No"

    above_stop = price_bar(16, open_price=125.0, high=130.0, low=124.0)
    closed = strategy.update_trade(trade, above_stop, previous + [entry])
    assert closed is False
    assert trade.exit_at is None

    touches_stop = price_bar(
        17,
        open_price=120.0,
        high=121.0,
        low=trade.current_stop - 1.0,
    )
    expected_stop = trade.current_stop
    closed = strategy.update_trade(trade, touches_stop, previous + [entry, above_stop])
    assert closed is True
    assert trade.exit_price == pytest.approx(expected_stop)
    assert trade.exit_reason == "trailing_stop"


def test_gap_through_stop_fills_at_observed_open() -> None:
    strategy = EventDrivenLongStrategy(range_period=14, range_multiplier=3)
    previous = [price_bar(index, high=101.0, low=99.0) for index in range(15)]
    entry = price_bar(15, open_price=100.0, high=101.0, low=99.0)
    trade = strategy.open_trade(
        run_id=uuid4(),
        market_id="market-gap",
        event_id="event-gap",
        question="Question?",
        symbol="TEST",
        asset_name="Test Asset",
        pass_number=1,
        trigger_at=START,
        entry_bar=entry,
        previous_bars=previous,
        final_outcome="No",
    )
    assert trade is not None
    observed_open = trade.current_stop - 2.0
    gap_bar = price_bar(
        16,
        open_price=observed_open,
        high=trade.current_stop + 1.0,
        low=observed_open - 1.0,
    )
    assert strategy.update_trade(trade, gap_bar, previous + [entry]) is True
    assert trade.exit_price == pytest.approx(observed_open)


def test_short_trailing_stop_only_lowers_and_fills_exactly_at_stop() -> None:
    strategy = EventDrivenLongStrategy(range_period=14, range_multiplier=3)
    previous = [price_bar(index, high=101.0, low=99.0) for index in range(15)]
    entry = price_bar(15, open_price=100.0, high=101.0, low=99.0)
    trade = strategy.open_trade(
        run_id=uuid4(),
        market_id="market-short",
        event_id="event-short",
        question="Question?",
        symbol="TEST",
        asset_name="Test Asset",
        pass_number=1,
        trigger_at=START,
        entry_bar=entry,
        previous_bars=previous,
        final_outcome="No",
        direction="short",
    )
    assert trade is not None
    initial_stop = trade.current_stop

    lower = price_bar(16, open_price=98.0, high=99.0, low=90.0)
    assert strategy.update_trade(trade, lower, previous + [entry]) is False
    assert trade.current_stop < initial_stop

    expected_stop = trade.current_stop
    touches_stop = price_bar(17, open_price=expected_stop, high=expected_stop + 1.0, low=90.0)
    assert strategy.update_trade(trade, touches_stop, previous + [entry, lower]) is True
    assert trade.exit_price == pytest.approx(expected_stop)
    assert trade.exit_reason == "trailing_stop"


def ml_observation(index: int, classification: int) -> MLObservation:
    first_pass = START + timedelta(days=index)
    return MLObservation(
        observation_id=uuid4(),
        run_id=uuid4(),
        event_id=f"event-{index}",
        market_id=f"market-{index}",
        first_pass_number=1,
        first_pass_at=first_pass,
        label_available_at=first_pass + timedelta(days=1),
        symbol="TEST",
        event_archetype="macro_rates",
        resolution="1d",
        features={
            "asset_ytd_change": index / 100,
            "sector_one_month_trend": index / 200,
            "spy_two_week_trend": -index / 300,
            "asset_two_week_trend": classification * (index + 1) / 100,
        },
        research_data={},
        classification_target=classification,
        regression_target=classification * (index + 1) / 100,
        valid_for_training=True,
    )


def test_model_snapshot_persistence_returns_canonical_database_id() -> None:
    async def run():
        canonical_id = uuid4()

        class FakeConnection:
            def __init__(self) -> None:
                self.query = ""

            async def fetchval(self, query: str, *arguments: object):
                self.query = query
                return canonical_id

        snapshot = train_snapshot(
            run_id=uuid4(),
            symbol="TEST",
            event_archetype="macro_rates",
            training_cutoff=START + timedelta(days=20),
            observations=[],
            minimum_prior_observations=8,
        )
        connection = FakeConnection()
        saved_id = await save_model_snapshot(connection, snapshot)
        return saved_id, canonical_id, connection.query

    saved_id, canonical_id, query = asyncio.run(run())
    assert saved_id == canonical_id
    assert "RETURNING snapshot_id" in query
    assert "ON CONFLICT (run_id, symbol, event_archetype, training_cutoff)" in query


def test_machine_learning_trains_after_eight_completed_prior_observations() -> None:
    observations = [ml_observation(index, 1 if index % 2 else -1) for index in range(8)]
    insufficient = train_snapshot(
        run_id=uuid4(),
        symbol="TEST",
        event_archetype="macro_rates",
        training_cutoff=START + timedelta(days=20),
        observations=observations[:7],
        minimum_prior_observations=8,
    )
    trained = train_snapshot(
        run_id=uuid4(),
        symbol="TEST",
        event_archetype="macro_rates",
        training_cutoff=START + timedelta(days=20),
        observations=observations,
        minimum_prior_observations=8,
    )
    assert insufficient.status == "insufficient_history"
    assert trained.status == "trained"
    assert trained.classifier_coefficients
    assert trained.ridge_coefficients


def test_unresolved_event_observation_is_saved_but_not_trainable() -> None:
    daily = [
        PriceBar(
            timestamp=START + timedelta(days=index),
            open=100 + index,
            high=101 + index,
            low=99 + index,
            close=100.5 + index,
            volume=1_000,
        )
        for index in range(40)
    ]
    observation = build_observation(
        run_id=uuid4(),
        event_id="event-unresolved",
        market_id="market-unresolved",
        first_pass_number=1,
        first_pass_at=START + timedelta(days=20),
        event_created_at=START,
        event_end_at=START + timedelta(days=50),
        label_data_cutoff=START + timedelta(days=40),
        symbol="TEST",
        event_archetype="macro_rates",
        resolution="1d",
        asset_daily=daily,
        sector_daily=daily,
        spy_daily=daily,
        research_data={},
    )
    assert observation.classification_target is None
    assert observation.regression_target is None
    assert observation.valid_for_training is False
    assert observation.research_data["event_open_price"] == pytest.approx(100)


def test_ml_regression_target_is_post_threshold_peak_from_event_open() -> None:
    daily = [
        PriceBar(
            timestamp=START + timedelta(days=index),
            open=100,
            high=150 if index == 1 else 110,
            low=95,
            close=105,
            volume=1_000,
        )
        for index in range(5)
    ]
    observation = build_observation(
        run_id=uuid4(),
        event_id="event-post-threshold-peak",
        market_id="market-post-threshold-peak",
        first_pass_number=1,
        first_pass_at=START + timedelta(days=2),
        event_created_at=START,
        event_end_at=START + timedelta(days=5),
        label_data_cutoff=START + timedelta(days=6),
        symbol="TEST",
        event_archetype="macro_rates",
        resolution="1d",
        asset_daily=daily,
        sector_daily=daily,
        spy_daily=daily,
        research_data={},
    )

    assert observation.classification_target == 1
    assert observation.regression_target == pytest.approx(0.10)
    assert observation.research_data["post_threshold_path_rows"] == 3


def test_ytd_feature_uses_previous_year_final_completed_close() -> None:
    previous_year = PriceBar(
        timestamp=datetime(2025, 12, 31, 14, tzinfo=timezone.utc),
        open=99,
        high=101,
        low=98,
        close=100,
        volume=1_000,
    )
    current = PriceBar(
        timestamp=datetime(2026, 1, 2, 14, tzinfo=timezone.utc),
        open=109,
        high=111,
        low=108,
        close=110,
        volume=1_000,
    )
    observation = build_observation(
        run_id=uuid4(),
        event_id="event-ytd",
        market_id="market-ytd",
        first_pass_number=1,
        first_pass_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        event_created_at=current.timestamp,
        event_end_at=current.timestamp + timedelta(days=1),
        label_data_cutoff=current.timestamp + timedelta(days=2),
        symbol="TEST",
        event_archetype="macro_rates",
        resolution="1d",
        asset_daily=[previous_year, current],
        sector_daily=[previous_year, current],
        spy_daily=[previous_year, current],
        research_data={},
    )
    assert observation.features["asset_ytd_change"] == pytest.approx(0.10)


def test_daily_close_is_not_visible_until_the_session_is_complete() -> None:
    bar = PriceBar(
        timestamp=START,
        open=100,
        high=110,
        low=90,
        close=105,
        volume=1_000,
    )
    assert close_as_of([bar], START + timedelta(hours=6)) is None
    assert close_as_of([bar], START + timedelta(hours=7)) == pytest.approx(105)


def test_prior_observations_are_filtered_by_label_availability_not_first_pass() -> None:
    class FakeConnection:
        def __init__(self) -> None:
            self.query = ""

        async def fetch(self, query: str, *arguments: object) -> list[object]:
            self.query = query
            return []

    connection = FakeConnection()
    asyncio.run(
        prior_ml_observations(
            connection,
            run_id=uuid4(),
            symbol="TEST",
            event_archetype="macro_rates",
            before=START,
        )
    )
    assert "label_available_at < $4" in connection.query
    assert "first_pass_at < $4" not in connection.query


def test_each_bought_trade_can_generate_its_own_graph(tmp_path) -> None:
    strategy = EventDrivenLongStrategy(range_period=14, range_multiplier=3)
    previous = [price_bar(index, high=101.0, low=99.0) for index in range(15)]
    entry = price_bar(15, open_price=125.0, high=126.0, low=124.0)
    trade = strategy.open_trade(
        run_id=uuid4(),
        market_id="market-graph",
        event_id="event-graph",
        question="Question?",
        symbol="TEST",
        asset_name="Test Asset",
        pass_number=2,
        trigger_at=START,
        entry_bar=entry,
        previous_bars=previous,
        final_outcome="Yes",
    )
    assert trade is not None
    trade.final_mark_price = entry.close

    path = create_trade_graph(
        trade,
        bars=[entry],
        probabilities=[ProbabilityPoint(entry.timestamp, 0.60)],
        simulation_end=entry.timestamp + timedelta(hours=1),
        graph_dir=tmp_path,
    )
    assert path.exists()
    assert "pass_2" in path.name
