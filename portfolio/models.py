from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

Direction = Literal["long", "short"]
Resolution = Literal["1h", "1d"]


class DecisionStatus(str, Enum):
    APPROVED = "approved"
    APPROVED_CAPPED = "approved_capped"
    REJECTED = "rejected"
    KILL_SWITCH_BLOCKED = "kill_switch_blocked"
    PAPER_TRADE = "paper_trade"
    REDUCE_ONLY = "reduce_only"
    HUMAN_REVIEW = "human_review"


CAP_ORDER = (
    "max_position_size",
    "max_event_exposure",
    "max_sector_exposure",
    "max_theme_exposure",
    "max_country_exposure",
    "max_gross_exposure",
    "max_net_exposure",
    "max_portfolio_heat",
    "liquidity_adv",
    "insufficient_cash",
)


@dataclass(frozen=True)
class TradeCandidate:
    candidate_id: str
    run_id: UUID
    market_id: str
    event_id: str
    symbol: str
    asset_name: str
    pass_number: int
    strategy_branch: str
    portfolio_label: str
    direction: Direction
    consumes_capital: bool
    trigger_at: datetime
    entry_at: datetime
    resolution: Resolution
    entry_price: float
    exit_at: datetime
    exit_price: float
    exit_reason: str
    path_reference_quantity: float
    initial_stop: float
    atr: float
    stop_distance: float
    range_period: int
    range_multiplier: float
    sector_key: str
    bar_lookup_key: str
    bar_window_start: datetime
    bar_window_end: datetime
    schema_version: int = 1
    event_archetype: str | None = None
    classification_probability: float | None = None
    predicted_peak_percent: float | None = None
    predicted_target_price: float | None = None
    remaining_gap: float | None = None
    directions_agree: bool | None = None
    price_momentum: float | None = None
    probability_at_entry: float | None = None
    theme_tags: tuple[str, ...] = ()
    country_tags: tuple[str, ...] = ()
    adv: float | None = None
    polymarket_volume_quality: dict[str, Any] = field(default_factory=dict)
    question: str = ""
    final_outcome: str | None = None
    momentum_parameter_selection: dict[str, Any] | None = None


@dataclass
class Position:
    position_id: UUID
    candidate_id: str
    event_id: str
    symbol: str
    direction: Direction
    quantity: float
    entry_price: float
    entry_at: datetime
    initial_stop: float
    stop_distance: float
    notional: float
    sector_key: str
    theme_tags: tuple[str, ...]
    country_tags: tuple[str, ...]
    risk_dollars: float
    entry_commission: float
    resolution: Resolution


@dataclass
class PortfolioState:
    starting_capital: float
    cash: float
    equity: float
    peak_equity: float
    drawdown: float
    kill_switch_active: bool
    open_positions: dict[str, Position] = field(default_factory=dict)
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    heat: float = 0.0
    event_exposure: dict[str, float] = field(default_factory=dict)
    sector_exposure: dict[str, float] = field(default_factory=dict)
    theme_exposure: dict[str, float] = field(default_factory=dict)
    country_exposure: dict[str, float] = field(default_factory=dict)
    paper_positions: list[Position] = field(default_factory=list)

    @property
    def open_position_count(self) -> int:
        return len(self.open_positions)

    @classmethod
    def empty(cls, starting_capital: float) -> PortfolioState:
        return cls(
            starting_capital=starting_capital,
            cash=starting_capital,
            equity=starting_capital,
            peak_equity=starting_capital,
            drawdown=0.0,
            kill_switch_active=False,
        )

    def validate_invariants(self) -> None:
        import math

        if any(
            math.isnan(value)
            for value in (self.cash, self.equity, self.gross_exposure, self.heat)
        ):
            raise RuntimeError("PortfolioState invariant violation: NaN accounting value")
        if self.cash < 0 and self.open_positions:
            raise RuntimeError("PortfolioState invariant violation: negative cash with open positions")


@dataclass(frozen=True)
class PortfolioDecision:
    decision_id: UUID
    candidate_id: str
    run_id: UUID
    evaluated_at: datetime
    status: DecisionStatus
    reason: str
    direction: Direction
    requested_quantity: float
    requested_notional: float
    requested_risk_dollars: float
    quantity: float
    notional: float
    risk_dollars: float
    risk_pct_of_equity: float
    effective_risk_pct: float
    entry_commission_estimate: float
    market_id: str
    event_id: str
    symbol: str
    strategy_branch: str
    portfolio_label: str
    cash_before: float
    cash_after: float
    equity_before: float
    equity_after: float
    heat_before: float
    heat_after: float
    gross_exposure_before: float
    gross_exposure_after: float
    net_exposure_before: float
    net_exposure_after: float
    open_position_count_before: int
    open_position_count_after: int
    drawdown_before: float
    kill_switch_active: bool
    active_caps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    binding_constraint: str | None = None

    @staticmethod
    def new_id() -> UUID:
        return uuid4()
