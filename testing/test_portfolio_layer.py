from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from dataclasses import replace

import pytest

from main_backtesting.config import BacktestConfig
from main_backtesting.models import PriceBar
from portfolio.config import PortfolioConfig
from portfolio.models import DecisionStatus, TradeCandidate
from portfolio.mtm import build_bar_index, mark_to_market
from portfolio.portfolio import Portfolio
from portfolio.replay import build_timeline, replay_portfolio
from portfolio.serialization import candidate_from_dict, candidate_to_dict
from portfolio.sizing import floor_to_lot, risk_target_quantity


START = datetime(2024, 1, 2, 14, tzinfo=timezone.utc)


def _candidate(
    *,
    symbol: str = "TEST",
    market_id: str = "m1",
    event_id: str = "e1",
    entry_at: datetime | None = None,
    exit_at: datetime | None = None,
    entry_price: float = 50.0,
    stop: float = 46.0,
    direction: str = "long",
    consumes_capital: bool = True,
    sector_key: str = "XLK",
    theme_tags: tuple[str, ...] = (),
    country_tags: tuple[str, ...] = (),
    adv: float | None = None,
    volume_quality: dict | None = None,
) -> TradeCandidate:
    entry = entry_at or START
    exit_time = exit_at or entry + timedelta(days=5)
    stop_distance = abs(entry_price - stop)
    run_id = uuid4()
    return TradeCandidate(
        candidate_id=f"{run_id}:{market_id}:1:{symbol}:machine_learning",
        run_id=run_id,
        market_id=market_id,
        event_id=event_id,
        symbol=symbol,
        asset_name=symbol,
        pass_number=1,
        strategy_branch="machine_learning",
        portfolio_label="machine_learning",
        direction=direction,
        consumes_capital=consumes_capital,
        trigger_at=entry - timedelta(hours=1),
        entry_at=entry,
        resolution="1d",
        entry_price=entry_price,
        exit_at=exit_time,
        exit_price=entry_price - 1,
        exit_reason="trailing_stop",
        path_reference_quantity=20.0,
        initial_stop=stop,
        atr=1.0,
        stop_distance=stop_distance,
        range_period=14,
        range_multiplier=3.0,
        sector_key=sector_key,
        theme_tags=theme_tags,
        country_tags=country_tags,
        adv=adv,
        polymarket_volume_quality=volume_quality or {},
        bar_lookup_key=f"{symbol}:1d",
        bar_window_start=entry - timedelta(days=30),
        bar_window_end=exit_time + timedelta(days=1),
    )


def test_risk_target_sizing_example() -> None:
    config = PortfolioConfig(starting_capital=100_000)
    qty, notional, risk_dollars, effective = risk_target_quantity(
        config=config,
        equity=100_000,
        entry_price=50,
        stop_distance=4,
        drawdown=0.0,
    )
    assert qty == pytest.approx(250)
    assert notional == pytest.approx(12_500)
    assert risk_dollars == pytest.approx(1_000)
    assert effective == pytest.approx(0.01)


def test_event_cap_binds() -> None:
    config = PortfolioConfig(starting_capital=100_000)
    portfolio = Portfolio.empty(config)
    first = _candidate(
        event_id="evt",
        entry_at=START,
        exit_at=START + timedelta(days=10),
        entry_price=100,
        stop=90,
    )
    second = _candidate(
        event_id="evt",
        symbol="TEST2",
        market_id="m2",
        entry_at=START + timedelta(minutes=1),
        exit_at=START + timedelta(days=10),
        entry_price=100,
        stop=90,
    )
    portfolio.set_bars({first.bar_lookup_key: [], second.bar_lookup_key: []})
    replay_portfolio(portfolio, [first, second])
    decisions = {item.candidate_id: item for item in portfolio.decisions}
    assert decisions[first.candidate_id].status == DecisionStatus.APPROVED
    second_decision = decisions[second.candidate_id]
    assert second_decision.status == DecisionStatus.APPROVED_CAPPED
    assert second_decision.binding_constraint == "max_event_exposure"


def test_exits_before_entries_same_timestamp() -> None:
    ts = START
    exiting = _candidate(
        symbol="OLD",
        market_id="m-old",
        entry_at=ts - timedelta(days=3),
        exit_at=ts,
        entry_price=100,
        stop=95,
    )
    entering = _candidate(
        symbol="NEW",
        market_id="m-new",
        entry_at=ts,
        exit_at=ts + timedelta(days=2),
        entry_price=50,
        stop=46,
    )
    config = PortfolioConfig(starting_capital=5_000, min_position_notional=100)
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({exiting.bar_lookup_key: [], entering.bar_lookup_key: []})
    replay_portfolio(portfolio, [entering, exiting])
    assert portfolio.decisions[-1].status in {
        DecisionStatus.APPROVED,
        DecisionStatus.APPROVED_CAPPED,
    }


def test_duplicate_event_symbol_rejected() -> None:
    config = PortfolioConfig()
    portfolio = Portfolio.empty(config)
    candidate = _candidate()
    portfolio.set_bars({candidate.bar_lookup_key: []})
    portfolio.evaluate(candidate)
    second = _candidate(symbol="TEST", market_id="m2")
    decision = portfolio.evaluate(second)
    assert decision.reason == "duplicate_event_symbol"


def test_candidate_roundtrip_json() -> None:
    candidate = _candidate()
    restored = candidate_from_dict(candidate_to_dict(candidate))
    assert restored.candidate_id == candidate.candidate_id
    assert restored.entry_at == candidate.entry_at


def test_passthrough_profile_approves_all_candidates() -> None:
    config = replace(
        PortfolioConfig().with_passthrough_profile(),
        starting_capital=100_000,
        risk_per_trade_pct=0.01,
    )
    portfolio = Portfolio.empty(config)
    candidates = [
        _candidate(
            symbol=f"S{i}",
            market_id=f"m{i}",
            event_id=f"e{i}",
            entry_at=START + timedelta(days=i * 3),
            exit_at=START + timedelta(days=i * 3 + 1),
        )
        for i in range(3)
    ]
    portfolio.set_bars({candidate.bar_lookup_key: [] for candidate in candidates})
    replay_portfolio(portfolio, candidates)
    assert len(portfolio.booked_trades) == 3
    timestamps = {(trade.market_id, trade.entry_at, trade.exit_at) for trade in portfolio.booked_trades}
    assert len(timestamps) == 3


def test_kill_switch_blocks_entries() -> None:
    config = PortfolioConfig(starting_capital=100_000, kill_switch_drawdown_pct=0.05)
    portfolio = Portfolio.empty(config)
    portfolio.state.cash = 90_000
    portfolio.state.equity = 90_000
    portfolio.state.peak_equity = 100_000
    candidate = _candidate()
    portfolio.set_bars({candidate.bar_lookup_key: []})
    decision = portfolio.evaluate(candidate)
    assert decision.status == DecisionStatus.KILL_SWITCH_BLOCKED


def test_volume_gate_rejects_when_failing() -> None:
    config = PortfolioConfig()
    portfolio = Portfolio.empty(config)
    candidate = _candidate(
        volume_quality={"gate_applied": True, "allowed": False, "reason": "polymarket_volume_not_significant"}
    )
    portfolio.set_bars({candidate.bar_lookup_key: []})
    decision = portfolio.evaluate(candidate)
    assert decision.reason == "low_polymarket_volume_quality"


def test_resume_candidate_payload_roundtrip() -> None:
    candidate = _candidate()
    payload = {"pass1_candidates": [candidate_to_dict(candidate)]}
    restored = candidate_from_dict(payload["pass1_candidates"][0])
    assert restored.symbol == candidate.symbol
    assert restored.exit_at == candidate.exit_at


def test_timeline_sort_is_deterministic() -> None:
    first = _candidate(symbol="A", market_id="m1", entry_at=START, exit_at=START + timedelta(hours=1))
    second = _candidate(symbol="B", market_id="m2", entry_at=START, exit_at=START + timedelta(hours=1))
    events = build_timeline([second, first])
    assert events[0].kind == "exit"
    assert events[0].symbol == "A"


def test_legacy_config_defaults_preserve_portfolio_disabled() -> None:
    config = BacktestConfig()
    assert config.portfolio_enabled is False
