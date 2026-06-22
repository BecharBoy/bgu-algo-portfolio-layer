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


def _evaluate_single(config: PortfolioConfig, candidate: TradeCandidate):
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({candidate.bar_lookup_key: []})
    return portfolio, portfolio.evaluate(candidate)


def test_risk_target_sizing_through_evaluate() -> None:
    # Drive the full evaluate() path (not the raw helper). With the single-name
    # cap raised above the risk target, no cap binds and the trade books at the
    # full downside-to-stop size: E=100k, risk=1%, entry 50, stop 46 -> 250 sh.
    config = PortfolioConfig(starting_capital=100_000, max_position_notional_pct=0.20)
    _, decision = _evaluate_single(config, _candidate(entry_price=50, stop=46))
    assert decision.status == DecisionStatus.APPROVED
    assert decision.binding_constraint is None
    assert decision.quantity == pytest.approx(250)
    assert decision.notional == pytest.approx(12_500)
    assert decision.risk_dollars == pytest.approx(1_000)
    assert decision.effective_risk_pct == pytest.approx(0.01)


def test_position_cap_downsizes_to_approved_capped() -> None:
    # Default single-name cap (10% x 100k = 10,000) binds below the 12,500 risk
    # target -> approved_capped, sized strictly DOWN, never up.
    config = PortfolioConfig(starting_capital=100_000)
    _, decision = _evaluate_single(config, _candidate(entry_price=50, stop=46))
    assert decision.status == DecisionStatus.APPROVED_CAPPED
    assert decision.binding_constraint == "max_position_size"
    assert decision.quantity == pytest.approx(200)  # 10,000 / 50
    assert decision.notional == pytest.approx(10_000)
    assert decision.notional < decision.requested_notional


def test_below_min_notional_rejects_without_state_mutation() -> None:
    # Risk target itself is below min notional (no cap binds) -> hard reject with
    # the below_min_notional reason, and no cash/exposure mutation.
    config = PortfolioConfig(
        starting_capital=100_000, risk_per_trade_pct=1e-6, min_position_notional=100
    )
    portfolio, decision = _evaluate_single(config, _candidate(entry_price=50, stop=46))
    assert decision.status == DecisionStatus.REJECTED
    assert decision.reason == "below_min_notional"
    assert decision.quantity == 0
    assert portfolio.state.cash == pytest.approx(config.starting_capital)
    assert portfolio.state.open_position_count == 0
    assert portfolio.state.heat == pytest.approx(0.0)


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


def test_exits_before_entries_same_timestamp_frees_capacity() -> None:
    # The slot cap is 1. An exit and a new entry share timestamp `ts`. The entry
    # can only be approved if the same-timestamp exit is processed FIRST (spec
    # §11.2). Under the previous global ordering (all exits before all entries),
    # the old position opened *after* its exit was consumed and never freed the
    # slot, so the new entry would reject with max_open_positions.
    ts = START + timedelta(days=3)
    exiting = _candidate(
        symbol="OLD", event_id="e-old", market_id="m-old",
        entry_at=START, exit_at=ts, entry_price=100, stop=95,
    )
    entering = _candidate(
        symbol="NEW", event_id="e-new", market_id="m-new",
        entry_at=ts, exit_at=ts + timedelta(days=2), entry_price=50, stop=46,
    )
    config = PortfolioConfig(starting_capital=100_000, max_open_positions=1)
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({exiting.bar_lookup_key: [], entering.bar_lookup_key: []})
    replay_portfolio(portfolio, [entering, exiting])
    decisions = {item.candidate_id: item for item in portfolio.decisions}
    assert decisions[exiting.candidate_id].status in {
        DecisionStatus.APPROVED, DecisionStatus.APPROVED_CAPPED,
    }
    entry_decision = decisions[entering.candidate_id]
    assert entry_decision.status in {
        DecisionStatus.APPROVED, DecisionStatus.APPROVED_CAPPED,
    }
    assert entry_decision.reason != "max_open_positions"
    assert {trade.market_id for trade in portfolio.booked_trades} == {"m-old", "m-new"}


def test_sequential_trades_open_and_close_restore_baseline() -> None:
    # Two non-overlapping trades must BOTH open at their entries and BOTH close at
    # their exits; after the last exit, heat/exposures/positions return to
    # baseline (the core regression the global-ordering bug broke).
    config = PortfolioConfig(starting_capital=100_000)
    first = _candidate(
        symbol="AAA", event_id="e-a", market_id="mA",
        entry_at=START, exit_at=START + timedelta(days=2),
    )
    second = _candidate(
        symbol="BBB", event_id="e-b", market_id="mB",
        entry_at=START + timedelta(days=5), exit_at=START + timedelta(days=7),
    )
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({first.bar_lookup_key: [], second.bar_lookup_key: []})
    replay_portfolio(portfolio, [first, second])
    state = portfolio.state
    assert len(portfolio.booked_trades) == 2
    assert state.open_position_count == 0
    assert state.heat == pytest.approx(0.0)
    assert state.gross_exposure == pytest.approx(0.0)
    assert state.net_exposure == pytest.approx(0.0)
    assert all(abs(value) < 1e-6 for value in state.event_exposure.values())
    assert all(abs(value) < 1e-6 for value in state.sector_exposure.values())
    # Nothing left marked: with no open positions equity equals cash.
    assert state.equity == pytest.approx(state.cash)


def test_duplicate_exit_does_not_close_original_position() -> None:
    # A duplicate (event,symbol) candidate is rejected at entry but still emits an
    # exit event; that exit must NOT evict the original, still-open position.
    config = PortfolioConfig(starting_capital=100_000)
    original = _candidate(
        symbol="TEST", event_id="e1", market_id="mA",
        entry_at=START, exit_at=START + timedelta(days=10),
    )
    duplicate = _candidate(
        symbol="TEST", event_id="e1", market_id="mB",
        entry_at=START + timedelta(days=1), exit_at=START + timedelta(days=5),
    )
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({original.bar_lookup_key: [], duplicate.bar_lookup_key: []})
    key = "e1:TEST"
    assert portfolio.evaluate(original).status in {
        DecisionStatus.APPROVED, DecisionStatus.APPROVED_CAPPED,
    }
    assert key in portfolio.state.open_positions
    assert portfolio.evaluate(duplicate).reason == "duplicate_event_symbol"
    portfolio.close(duplicate)
    assert key in portfolio.state.open_positions
    assert portfolio.state.open_positions[key].candidate_id == original.candidate_id
    portfolio.close(original)
    assert key not in portfolio.state.open_positions


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


def test_volume_gate_rejects_when_failing_and_enforcement_enabled() -> None:
    # Enforcement is a Milestone 2 control behind an explicit flag.
    config = PortfolioConfig(enforce_polymarket_volume_gate=True)
    portfolio = Portfolio.empty(config)
    candidate = _candidate(
        volume_quality={"gate_applied": True, "allowed": False, "reason": "polymarket_volume_not_significant"}
    )
    portfolio.set_bars({candidate.bar_lookup_key: []})
    decision = portfolio.evaluate(candidate)
    assert decision.reason == "low_polymarket_volume_quality"


def test_volume_gate_not_enforced_by_default() -> None:
    # Option 2 Core default: a present-but-failing volume quality is logged but
    # must NOT reject the trade (enforcement deferred to Milestone 2).
    config = PortfolioConfig()
    assert config.enforce_polymarket_volume_gate is False
    portfolio = Portfolio.empty(config)
    candidate = _candidate(
        volume_quality={"gate_applied": True, "allowed": False, "reason": "polymarket_volume_not_significant"}
    )
    portfolio.set_bars({candidate.bar_lookup_key: []})
    decision = portfolio.evaluate(candidate)
    assert decision.reason != "low_polymarket_volume_quality"
    assert decision.status in {DecisionStatus.APPROVED, DecisionStatus.APPROVED_CAPPED}


def test_resume_candidate_payload_roundtrip() -> None:
    candidate = _candidate()
    payload = {"pass1_candidates": [candidate_to_dict(candidate)]}
    restored = candidate_from_dict(payload["pass1_candidates"][0])
    assert restored.symbol == candidate.symbol
    assert restored.exit_at == candidate.exit_at


def test_timeline_sort_is_deterministic() -> None:
    # An exit and an entry share timestamp `ts`: the timeline must order by
    # timestamp first, then exit-before-entry within that timestamp (spec §11.2),
    # and be order-independent of the input list.
    ts = START + timedelta(days=5)
    exiting = _candidate(symbol="A", market_id="m1", event_id="eA",
                         entry_at=START, exit_at=ts)
    entering = _candidate(symbol="B", market_id="m2", event_id="eB",
                          entry_at=ts, exit_at=ts + timedelta(days=1))
    events = build_timeline([entering, exiting])
    at_ts = [event for event in events if event.timestamp == ts]
    assert [event.kind for event in at_ts] == ["exit", "entry"]
    assert at_ts[0].symbol == "A"
    again = build_timeline([exiting, entering])
    assert [(e.timestamp, e.sort_rank, e.market_id) for e in events] == [
        (e.timestamp, e.sort_rank, e.market_id) for e in again
    ]


def test_legacy_config_defaults_preserve_portfolio_disabled() -> None:
    config = BacktestConfig()
    assert config.portfolio_enabled is False


def test_generate_portfolio_reports_writes_all_files(tmp_path) -> None:
    # End-to-end report generation must not crash (regression for the missing
    # compute_trade_alpha_rows import) and must emit every Milestone 1 file.
    import csv
    import json

    from portfolio.reporting import generate_portfolio_reports

    config = PortfolioConfig(starting_capital=100_000)
    candidate = _candidate(
        symbol="AAA", event_id="e-a", market_id="mA",
        entry_at=START, exit_at=START + timedelta(days=2),
    )
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({candidate.bar_lookup_key: []})
    replay_portfolio(portfolio, [candidate])

    report_dir = tmp_path / "reports"
    generate_portfolio_reports(portfolio, report_dir)

    expected_files = [
        "portfolio_decisions.csv",
        "portfolio_equity_curve.csv",
        "portfolio_exposure.csv",
        "portfolio_summary.json",
        "trade_alpha.csv",
        "baseline_vs_portfolio.csv",
        "PORTFOLIO_INTEGRATION_NOTES.md",
    ]
    for name in expected_files:
        assert (report_dir / name).exists(), f"missing report: {name}"

    summary = json.loads((report_dir / "portfolio_summary.json").read_text())
    assert summary["booked_trade_count"] == 1
    decision_rows = list(csv.DictReader((report_dir / "portfolio_decisions.csv").open()))
    assert len(decision_rows) == 1


def test_passthrough_invariant_opens_all_candidates_unchanged_timestamps() -> None:
    # Spec safety rail 2 (offline form of test 39): with the real passthrough
    # profile (caps disabled, unlimited capital), EVERY capital-consuming
    # candidate must open at its Pass-1 entry/exit timestamps with no reject and
    # no retiming. Includes overlapping trades, a shared theme/sector, a short,
    # and a duplicate (event,symbol) pair — none may be rejected here.
    config = PortfolioConfig().with_passthrough_profile()
    base = START
    candidates = [
        _candidate(symbol="S1", event_id="e1", market_id="m1",
                   entry_at=base, exit_at=base + timedelta(days=10),
                   theme_tags=("oil",)),
        _candidate(symbol="S2", event_id="e2", market_id="m2",
                   entry_at=base + timedelta(days=1), exit_at=base + timedelta(days=9),
                   theme_tags=("oil",)),
        _candidate(symbol="S1", event_id="e1", market_id="m3",  # duplicate (e1,S1)
                   entry_at=base + timedelta(days=2), exit_at=base + timedelta(days=8)),
        _candidate(symbol="S3", event_id="e3", market_id="m4", direction="short",
                   entry_at=base + timedelta(days=3), exit_at=base + timedelta(days=7)),
    ]
    portfolio = Portfolio.empty(config)
    portfolio.set_bars({candidate.bar_lookup_key: [] for candidate in candidates})
    replay_portfolio(portfolio, candidates)

    assert len(portfolio.booked_trades) == len(candidates)
    opened = {
        (trade.market_id, trade.symbol, trade.entry_at, trade.exit_at)
        for trade in portfolio.booked_trades
    }
    expected = {
        (candidate.market_id, candidate.symbol, candidate.entry_at, candidate.exit_at)
        for candidate in candidates
    }
    assert opened == expected
    statuses = {decision.status for decision in portfolio.decisions}
    assert DecisionStatus.REJECTED not in statuses
    assert DecisionStatus.KILL_SWITCH_BLOCKED not in statuses


def test_legacy_volume_gate_applied_false_only_portfolio_marks_it() -> None:
    # Legacy parity (safety rail 1 / spec §6): the volume-quality diagnostic
    # defaults to gate_applied=False (pre-portfolio behavior). Only the portfolio
    # path opts in to gate_applied=True; this keeps legacy-mode diagnostics
    # unchanged.
    # Importing the simulation stage pulls the DB stack (asyncpg); skip cleanly
    # when those pipeline deps are absent (e.g. a pure offline checkout).
    simulation = pytest.importorskip("main_backtesting.stages.simulation")
    from main_backtesting.models import ProbabilityPoint

    polymarket_volume_quality = simulation.polymarket_volume_quality

    points = [
        ProbabilityPoint(START, 0.54, volume_usdc=100.0),
        ProbabilityPoint(START + timedelta(hours=1), 0.60, volume_usdc=1_000.0),
    ]
    kwargs = dict(
        trigger_at=START + timedelta(hours=1),
        minimum_pre_entry_usdc=10.0,
        concentration_minimum_usdc=1_000.0,
        max_single_hour_share=0.95,
    )
    legacy = polymarket_volume_quality(points, **kwargs)
    assert legacy["gate_applied"] is False
    portfolio_mode = polymarket_volume_quality(points, gate_applied=True, **kwargs)
    assert portfolio_mode["gate_applied"] is True
