from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4

from main_backtesting.models import PriceBar, Trade
from portfolio.config import PortfolioConfig
from portfolio.constraints import (
    apply_caps,
    compute_caps,
    estimate_commission,
    normalize_candidate,
    volume_gate_reject,
)
from portfolio.models import (
    DecisionStatus,
    PortfolioDecision,
    PortfolioState,
    Position,
    TradeCandidate,
)
from portfolio.mtm import refresh_state_marks
from portfolio.sizing import floor_to_lot, risk_target_quantity
from strategies.event_driven_long import ib_style_commission


@dataclass
class Portfolio:
    config: PortfolioConfig
    state: PortfolioState
    decisions: list[PortfolioDecision] = field(default_factory=list)
    equity_snapshots: list[dict[str, object]] = field(default_factory=list)
    exposure_snapshots: list[dict[str, object]] = field(default_factory=list)
    booked_trades: list[Trade] = field(default_factory=list)
    bars_by_key: dict[str, list[PriceBar]] = field(default_factory=dict)

    @classmethod
    def empty(cls, config: PortfolioConfig) -> Portfolio:
        return cls(config=config, state=PortfolioState.empty(config.starting_capital))

    def set_bars(self, bars_by_key: dict[str, list[PriceBar]]) -> None:
        self.bars_by_key = bars_by_key

    def _refresh(self, timestamp: datetime) -> None:
        refresh_state_marks(
            self.state,
            timestamp=timestamp,
            bars_by_key=self.bars_by_key,
            kill_switch_drawdown_pct=self.config.kill_switch_drawdown_pct,
        )
        # When caps are disabled (passthrough / unlimited-capital mode) the cash
        # gate is intentionally off, so negative cash is expected rather than a
        # bug. NaN detection stays always-on regardless.
        self.state.validate_invariants(allow_negative_cash=not self.config.caps_enabled)

    def evaluate(
        self,
        candidate: TradeCandidate,
        *,
        timestamp: datetime | None = None,
    ) -> PortfolioDecision:
        evaluated_at = timestamp or candidate.entry_at
        self._refresh(evaluated_at)
        before = self._snapshot()

        normalized, invalid_reason = normalize_candidate(candidate, self.config)
        if invalid_reason:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason=invalid_reason,
                before=before,
                warnings=(),
                active_caps=(),
            )

        assert normalized is not None
        warnings = list(normalized.warnings)
        active_caps = list(normalized.active_caps)

        if self.state.kill_switch_active:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason="kill_switch_blocked",
                status=DecisionStatus.KILL_SWITCH_BLOCKED,
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
            )

        if candidate.direction == "short" and not self.config.allow_shorts:
            if self.config.short_route == "paper" and self.config.paper_trade_enabled:
                return self._paper_route(candidate, evaluated_at=evaluated_at, before=before, warnings=warnings, active_caps=active_caps)
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason="short_disabled",
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
            )

        position_key = f"{candidate.event_id}:{candidate.symbol}"
        if self.config.one_position_per_event_symbol and position_key in self.state.open_positions:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason="duplicate_event_symbol",
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
            )

        if self.state.open_position_count >= self.config.max_open_positions:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason="max_open_positions",
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
            )

        # Polymarket volume-gate ENFORCEMENT is a Milestone 2 control: off by
        # default in the Option 2 Core so a present-but-failing volume quality
        # does not reject otherwise-valid trades. The quality dict is still
        # carried on the candidate and logged regardless; only the reject is gated.
        volume_reason = (
            volume_gate_reject(normalized)
            if self.config.enforce_polymarket_volume_gate
            else None
        )
        if volume_reason:
            if (
                not normalized.skip_volume_gate
                and self.config.on_missing_volume == "paper_trade"
                and self.config.paper_trade_enabled
            ):
                return self._paper_route(
                    candidate,
                    evaluated_at=evaluated_at,
                    before=before,
                    warnings=warnings,
                    active_caps=active_caps,
                )
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason=volume_reason,
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
            )

        qty_risk, notional_risk, requested_risk_dollars, effective_pct = risk_target_quantity(
            config=self.config,
            equity=self.state.equity,
            entry_price=candidate.entry_price,
            stop_distance=candidate.stop_distance,
            drawdown=self.state.drawdown,
        )

        caps = compute_caps(
            config=self.config,
            state=self.state,
            normalized=normalized,
            equity=self.state.equity,
            entry_price=candidate.entry_price,
            stop_distance=candidate.stop_distance,
        )
        final_notional, binding = apply_caps(
            notional_risk=notional_risk,
            caps=caps,
            min_notional=self.config.min_position_notional,
        )
        quantity = floor_to_lot(final_notional / candidate.entry_price)
        final_notional = quantity * candidate.entry_price
        commission = estimate_commission(quantity, final_notional)

        required_cash = final_notional + commission
        if candidate.direction == "short":
            required_cash = final_notional + commission
        if quantity < 1 or final_notional < self.config.min_position_notional:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason=binding or "below_min_notional",
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
                binding=binding,
                requested_quantity=qty_risk,
                requested_notional=notional_risk,
                requested_risk_dollars=requested_risk_dollars,
                effective_pct=effective_pct,
            )
        if self.config.caps_enabled and required_cash > self.state.cash:
            return self._reject(
                candidate,
                evaluated_at=evaluated_at,
                reason="insufficient_cash",
                before=before,
                warnings=tuple(warnings),
                active_caps=tuple(active_caps),
                requested_quantity=qty_risk,
                requested_notional=notional_risk,
                requested_risk_dollars=requested_risk_dollars,
                effective_pct=effective_pct,
            )

        if (
            normalized.skip_liquidity_cap
            and self.config.on_missing_liquidity == "paper_trade"
            and self.config.paper_trade_enabled
        ):
            return self._paper_route(
                candidate,
                evaluated_at=evaluated_at,
                before=before,
                warnings=warnings,
                active_caps=active_caps,
            )

        status = DecisionStatus.APPROVED
        reason = "approved"
        if binding and final_notional < notional_risk - 1e-9:
            status = DecisionStatus.APPROVED_CAPPED
            reason = "approved_capped"

        decision = self._build_decision(
            candidate=candidate,
            evaluated_at=evaluated_at,
            status=status,
            reason=reason,
            binding=binding,
            before=before,
            quantity=quantity,
            notional=final_notional,
            requested_quantity=qty_risk,
            requested_notional=notional_risk,
            requested_risk_dollars=requested_risk_dollars,
            effective_pct=effective_pct,
            commission=commission,
            warnings=tuple(warnings),
            active_caps=tuple(active_caps),
        )
        self.open(candidate, decision)
        decision = self._with_after(decision, self._snapshot())
        self.decisions.append(decision)
        return decision

    def open(self, candidate: TradeCandidate, decision: PortfolioDecision) -> Position:
        commission = estimate_commission(decision.quantity, decision.notional)
        position = Position(
            position_id=uuid4(),
            candidate_id=candidate.candidate_id,
            event_id=candidate.event_id,
            symbol=candidate.symbol,
            direction=candidate.direction,
            quantity=decision.quantity,
            entry_price=candidate.entry_price,
            entry_at=candidate.entry_at,
            initial_stop=candidate.initial_stop,
            stop_distance=candidate.stop_distance,
            notional=decision.notional,
            sector_key=candidate.sector_key,
            theme_tags=candidate.theme_tags,
            country_tags=candidate.country_tags,
            risk_dollars=decision.risk_dollars,
            entry_commission=commission,
            resolution=candidate.resolution,
        )
        key = f"{candidate.event_id}:{candidate.symbol}"
        self.state.open_positions[key] = position
        self.state.cash -= decision.notional + commission
        self._apply_exposure(position, multiplier=1)
        self._refresh(candidate.entry_at)
        return position

    def close(self, candidate: TradeCandidate) -> None:
        key = f"{candidate.event_id}:{candidate.symbol}"
        position = self.state.open_positions.get(key)
        # Only close the position THIS candidate booked. A non-booked candidate
        # (e.g. a duplicate (event,symbol) rejected at entry) still emits an exit
        # event; without this identity check it would close the original holding.
        if position is None or position.candidate_id != candidate.candidate_id:
            return
        del self.state.open_positions[key]
        multiplier = 1.0 if position.direction == "long" else -1.0
        gross = (candidate.exit_price - position.entry_price) * position.quantity * multiplier
        exit_commission = ib_style_commission(position.quantity, position.quantity * candidate.exit_price)
        self.state.cash += position.notional + gross - exit_commission
        if position.direction == "short":
            self.state.cash += 0.0
        self._apply_exposure(position, multiplier=-1)
        self._refresh(candidate.exit_at)

    def build_trade(self, candidate: TradeCandidate, decision: PortfolioDecision) -> Trade:
        quantity = decision.quantity
        entry_commission = estimate_commission(quantity, decision.notional)
        exit_commission = ib_style_commission(quantity, quantity * candidate.exit_price)
        trade = Trade(
            trade_id=uuid4(),
            run_id=candidate.run_id,
            market_id=candidate.market_id,
            event_id=candidate.event_id,
            question=candidate.question,
            symbol=candidate.symbol,
            asset_name=candidate.asset_name,
            pass_number=candidate.pass_number,
            trigger_at=candidate.trigger_at,
            entry_at=candidate.entry_at,
            entry_price=candidate.entry_price,
            quantity=quantity,
            entry_commission=entry_commission,
            initial_stop=candidate.initial_stop,
            current_stop=candidate.initial_stop,
            highest_price=candidate.entry_price,
            lowest_price=candidate.entry_price,
            final_outcome=candidate.final_outcome,
            portfolio=candidate.portfolio_label,
            strategy_branch=candidate.strategy_branch,
            resolution=candidate.resolution,
            direction=candidate.direction,
            predicted_target_price=candidate.predicted_target_price,
            range_period=candidate.range_period,
            range_multiplier=candidate.range_multiplier,
            parameter_selection=candidate.momentum_parameter_selection or {},
            exit_at=candidate.exit_at,
            exit_price=candidate.exit_price,
            exit_commission=exit_commission,
            exit_reason=candidate.exit_reason,
        )
        self.booked_trades.append(trade)
        return trade

    def record_snapshot(self, timestamp: datetime) -> None:
        self._refresh(timestamp)
        self.equity_snapshots.append(
            {
                "timestamp": timestamp.isoformat(),
                "cash": self.state.cash,
                "equity": self.state.equity,
                "drawdown": self.state.drawdown,
            }
        )
        self.exposure_snapshots.append(
            {
                "timestamp": timestamp.isoformat(),
                "gross_exposure": self.state.gross_exposure,
                "net_exposure": self.state.net_exposure,
                "heat": self.state.heat,
                "open_position_count": self.state.open_position_count,
            }
        )

    def _apply_exposure(self, position: Position, *, multiplier: int) -> None:
        signed = position.notional if position.direction == "long" else -position.notional
        self.state.gross_exposure += multiplier * position.notional
        self.state.net_exposure += multiplier * signed
        self.state.heat += multiplier * position.risk_dollars
        self.state.event_exposure[position.event_id] = (
            self.state.event_exposure.get(position.event_id, 0.0) + multiplier * position.notional
        )
        self.state.sector_exposure[position.sector_key] = (
            self.state.sector_exposure.get(position.sector_key, 0.0) + multiplier * position.notional
        )
        for tag in position.theme_tags:
            self.state.theme_exposure[tag] = (
                self.state.theme_exposure.get(tag, 0.0) + multiplier * position.notional
            )
        for tag in position.country_tags:
            self.state.country_exposure[tag] = (
                self.state.country_exposure.get(tag, 0.0) + multiplier * position.notional
            )

    def _snapshot(self) -> dict[str, float | int]:
        return {
            "cash": self.state.cash,
            "equity": self.state.equity,
            "heat": self.state.heat,
            "gross_exposure": self.state.gross_exposure,
            "net_exposure": self.state.net_exposure,
            "open_position_count": self.state.open_position_count,
            "drawdown": self.state.drawdown,
        }

    def _reject(
        self,
        candidate: TradeCandidate,
        *,
        evaluated_at: datetime,
        reason: str,
        before: dict[str, float | int],
        warnings: tuple[str, ...],
        active_caps: tuple[str, ...],
        status: DecisionStatus = DecisionStatus.REJECTED,
        binding: str | None = None,
        requested_quantity: float = 0.0,
        requested_notional: float = 0.0,
        requested_risk_dollars: float = 0.0,
        effective_pct: float = 0.0,
    ) -> PortfolioDecision:
        decision = self._build_decision(
            candidate=candidate,
            evaluated_at=evaluated_at,
            status=status,
            reason=reason,
            binding=binding,
            before=before,
            quantity=0.0,
            notional=0.0,
            requested_quantity=requested_quantity,
            requested_notional=requested_notional,
            requested_risk_dollars=requested_risk_dollars,
            effective_pct=effective_pct,
            commission=0.0,
            warnings=warnings,
            active_caps=active_caps,
        )
        decision = self._with_after(decision, before)
        self.decisions.append(decision)
        return decision

    def _paper_route(
        self,
        candidate: TradeCandidate,
        *,
        evaluated_at: datetime,
        before: dict[str, float | int],
        warnings: list[str],
        active_caps: list[str],
    ) -> PortfolioDecision:
        decision = self._build_decision(
            candidate=candidate,
            evaluated_at=evaluated_at,
            status=DecisionStatus.PAPER_TRADE,
            reason="paper_trade",
            binding=None,
            before=before,
            quantity=0.0,
            notional=0.0,
            requested_quantity=0.0,
            requested_notional=0.0,
            requested_risk_dollars=0.0,
            effective_pct=0.0,
            commission=0.0,
            warnings=tuple(warnings),
            active_caps=tuple(active_caps),
        )
        decision = self._with_after(decision, before)
        self.decisions.append(decision)
        return decision

    def _build_decision(
        self,
        *,
        candidate: TradeCandidate,
        evaluated_at: datetime,
        status: DecisionStatus,
        reason: str,
        binding: str | None,
        before: dict[str, float | int],
        quantity: float,
        notional: float,
        requested_quantity: float,
        requested_notional: float,
        requested_risk_dollars: float,
        effective_pct: float,
        commission: float,
        warnings: tuple[str, ...],
        active_caps: tuple[str, ...],
    ) -> PortfolioDecision:
        equity = float(before["equity"])
        risk_dollars = quantity * candidate.stop_distance
        return PortfolioDecision(
            decision_id=PortfolioDecision.new_id(),
            candidate_id=candidate.candidate_id,
            run_id=candidate.run_id,
            evaluated_at=evaluated_at,
            status=status,
            reason=reason,
            binding_constraint=binding,
            direction=candidate.direction,
            requested_quantity=requested_quantity,
            requested_notional=requested_notional,
            requested_risk_dollars=requested_risk_dollars,
            quantity=quantity,
            notional=notional,
            risk_dollars=risk_dollars,
            risk_pct_of_equity=(risk_dollars / equity if equity > 0 else 0.0),
            effective_risk_pct=effective_pct,
            entry_commission_estimate=commission,
            market_id=candidate.market_id,
            event_id=candidate.event_id,
            symbol=candidate.symbol,
            strategy_branch=candidate.strategy_branch,
            portfolio_label=candidate.portfolio_label,
            cash_before=float(before["cash"]),
            cash_after=float(before["cash"]),
            equity_before=equity,
            equity_after=equity,
            heat_before=float(before["heat"]),
            heat_after=float(before["heat"]),
            gross_exposure_before=float(before["gross_exposure"]),
            gross_exposure_after=float(before["gross_exposure"]),
            net_exposure_before=float(before["net_exposure"]),
            net_exposure_after=float(before["net_exposure"]),
            open_position_count_before=int(before["open_position_count"]),
            open_position_count_after=int(before["open_position_count"]),
            drawdown_before=float(before["drawdown"]),
            kill_switch_active=self.state.kill_switch_active,
            active_caps=active_caps,
            warnings=warnings,
        )

    def _with_after(
        self,
        decision: PortfolioDecision,
        after: dict[str, float | int],
    ) -> PortfolioDecision:
        from dataclasses import replace

        return replace(
            decision,
            cash_after=float(after["cash"]),
            equity_after=float(after["equity"]),
            heat_after=float(after["heat"]),
            gross_exposure_after=float(after["gross_exposure"]),
            net_exposure_after=float(after["net_exposure"]),
            open_position_count_after=int(after["open_position_count"]),
        )
