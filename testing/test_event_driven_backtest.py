from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest

from database.backtesting.historical_repository import prior_ml_observations
from database.backtesting.market_data import next_bar_after
from database.backtesting.news import (
    GdeltNewsClient,
    GdeltRateLimitError,
    PostgresGdeltSearchLimiter,
)
from database.backtesting.polymarket import hourly_as_of_points
from database.backtesting.repository import candidate_events
import main_backtesting.engine as engine_module
from main_backtesting.config import BacktestConfig
from main_backtesting.engine import (
    HistoricalBacktestEngine,
    detect_passes,
    validate_pipeline_integrity,
)
from main_backtesting.stages import STAGES, STAGE_FUNCTIONS
from main_backtesting.models import Asset, MLObservation, PriceBar, ProbabilityPoint, SourceMarket
from main_backtesting.reporting import create_trade_graph
from main_backtesting.stages.prices import _merge_requests
from main_backtesting.stages.simulation import polymarket_volume_ratio, simulate_one_trade
from strategies.event_driven_long import (
    EventDrivenLongStrategy,
    average_true_range,
    ib_style_commission,
    rate_of_change,
)
from strategies.event_driven_ml import build_observation, close_as_of, train_snapshot

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


def test_stop_distance_uses_previous_fourteen_completed_high_low_ranges() -> None:
    bars = [
        price_bar(index, high=100.0 + index, low=98.0 + index)
        for index in range(14)
    ]
    assert average_true_range(bars, 14) == pytest.approx(2.0)
    assert average_true_range(bars[:13], 14) is None


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


def test_polymarket_volume_ratio_uses_current_completed_hour_vs_history() -> None:
    points = [
        ProbabilityPoint(
            START + timedelta(hours=index),
            0.50 + index / 100,
            volume_usdc=100.0 if index < 6 else 250.0,
        )
        for index in range(7)
    ]
    assert polymarket_volume_ratio(
        points,
        trigger_at=points[-1].timestamp,
        lookback_hours=24,
        minimum_history_hours=6,
    ) == pytest.approx(2.5)


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


def test_entry_uses_strictly_next_available_bar() -> None:
    current = price_bar(0)
    following = price_bar(1)
    assert next_bar_after([current, following], current.timestamp) == following


def test_machine_learning_trade_exits_at_predicted_target(tmp_path) -> None:
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
    assert trade.exit_reason == "ml_predicted_target"
    assert trade.exit_price == pytest.approx(103.0)


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


def test_short_trailing_stop_only_lowers_and_fills_exactly_at_stop() -> None:
    strategy = EventDrivenLongStrategy(range_period=14, range_multiplier=3)
    previous = [price_bar(index, high=101.0, low=99.0) for index in range(14)]
    entry = price_bar(14, open_price=100.0, high=101.0, low=99.0)
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

    lower = price_bar(15, open_price=98.0, high=99.0, low=90.0)
    assert strategy.update_trade(trade, lower, previous + [entry]) is False
    assert trade.current_stop < initial_stop

    expected_stop = trade.current_stop
    touches_stop = price_bar(16, open_price=expected_stop, high=expected_stop + 1.0, low=90.0)
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


def test_machine_learning_requires_more_than_eight_completed_prior_observations() -> None:
    observations = [ml_observation(index, 1 if index % 2 else -1) for index in range(9)]
    insufficient = train_snapshot(
        run_id=uuid4(),
        symbol="TEST",
        event_archetype="macro_rates",
        training_cutoff=START + timedelta(days=20),
        observations=observations[:8],
        minimum_prior_observations=9,
    )
    trained = train_snapshot(
        run_id=uuid4(),
        symbol="TEST",
        event_archetype="macro_rates",
        training_cutoff=START + timedelta(days=20),
        observations=observations,
        minimum_prior_observations=9,
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
