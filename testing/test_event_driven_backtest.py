from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from database.backtesting.repository import candidate_events
from main_backtesting.engine import detect_passes
from main_backtesting.models import PriceBar, ProbabilityPoint
from main_backtesting.reporting import create_trade_graph
from strategies.event_driven_long import (
    EventDrivenLongStrategy,
    ib_style_commission,
)

START = datetime(2026, 1, 2, 14, tzinfo=timezone.utc)


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


def test_numbered_passes_record_falls_below() -> None:
    passes = detect_passes(
        "market-1",
        probability_points([0.40, 0.56, 0.70, 0.55, 0.54, 0.60, 0.52, 0.80]),
        0.55,
    )

    assert [item.pass_number for item in passes] == [1, 2, 3]
    assert passes[0].above_probability == pytest.approx(0.56)
    assert passes[0].fell_below_probability == pytest.approx(0.55)
    assert passes[1].fell_below_probability == pytest.approx(0.52)
    assert passes[2].fell_below_at is None


def test_market_starting_above_threshold_creates_first_pass() -> None:
    passes = detect_passes("market-1", probability_points([0.80, 0.81]), 0.55)
    assert len(passes) == 1
    assert passes[0].pass_number == 1
    assert passes[0].above_at == START


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

    assert "end_at > GREATEST(created_at, $1) + ($3 * INTERVAL '1 day')" in connection.query
    assert "end_at <= GREATEST(created_at, $1) + ($4 * INTERVAL '1 day')" in connection.query
    assert connection.arguments[2:4] == (5.0, 60.0)


def test_ib_style_commission_uses_minimum_and_cap() -> None:
    assert ib_style_commission(10, 1_000) == pytest.approx(1.0)
    assert ib_style_commission(10_000, 100) == pytest.approx(1.0)
    assert ib_style_commission(1_000, 100_000) == pytest.approx(5.0)


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
