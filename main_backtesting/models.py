from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

Direction = Literal["long", "short"]
Resolution = Literal["1h", "1d"]


@dataclass(frozen=True)
class SourceEvent:
    event_id: str
    title: str
    created_at: datetime
    end_at: datetime
    tags: list[str]
    matched_tags: list[str]


@dataclass(frozen=True)
class SourceMarket:
    market_id: str
    event_id: str
    event_title: str
    question: str
    created_at: datetime
    end_at: datetime
    tags: list[str]
    raw_market: dict[str, Any]
    yes_token_id: str
    condition_id: str | None
    final_outcome: str | None


@dataclass(frozen=True)
class ProbabilityPoint:
    timestamp: datetime
    probability: float
    source_timestamp: datetime | None = None
    available_at: datetime | None = None
    volume_usdc: float | None = None


@dataclass
class ThresholdPass:
    market_id: str
    pass_number: int
    above_at: datetime
    above_probability: float
    fell_below_at: datetime | None = None
    fell_below_probability: float | None = None


@dataclass(frozen=True)
class Asset:
    symbol: str
    asset_name: str
    asset_class: str
    reason: str


@dataclass(frozen=True)
class IBTradableAsset:
    symbol: str
    asset_name: str
    asset_class: Literal["stock", "etf"]
    primary_exchange: str
    stock_type: str
    industry: str | None = None
    category: str | None = None
    subcategory: str | None = None

    def prompt_record(self) -> dict[str, str]:
        record = {
            "symbol": self.symbol,
            "asset_name": self.asset_name,
            "asset_class": self.asset_class,
            "primary_exchange": self.primary_exchange,
            "stock_type": self.stock_type,
        }
        for field_name in ("industry", "category", "subcategory"):
            value = getattr(self, field_name)
            if value:
                record[field_name] = value
        return record


@dataclass(frozen=True)
class NewsArticle:
    url: str
    title: str
    published_at: datetime
    domain: str | None
    text: str


@dataclass(frozen=True)
class SentimentResult:
    label: str
    score: float
    positive_count: int
    neutral_count: int
    negative_count: int
    details: list[dict[str, Any]]


@dataclass(frozen=True)
class PriceBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class MLObservation:
    observation_id: UUID
    run_id: UUID
    event_id: str
    market_id: str
    first_pass_number: int
    first_pass_at: datetime
    label_available_at: datetime
    symbol: str
    event_archetype: str
    resolution: Resolution
    features: dict[str, float | None]
    research_data: dict[str, Any]
    classification_target: int | None
    regression_target: float | None
    valid_for_training: bool
    exclusion_reason: str | None = None


@dataclass
class MLModelSnapshot:
    snapshot_id: UUID
    run_id: UUID
    symbol: str
    event_archetype: str
    training_cutoff: datetime
    training_event_ids: list[str]
    training_sample_count: int
    status: str
    feature_names: list[str]
    feature_means: dict[str, float]
    feature_scales: dict[str, float]
    classifier_coefficients: dict[str, float] | None
    classifier_intercept: float | None
    ridge_coefficients: dict[str, float] | None
    ridge_intercept: float | None
    hyperparameters: dict[str, Any]
    validation_metrics: dict[str, Any]


@dataclass
class MLPrediction:
    prediction_id: UUID
    run_id: UUID
    snapshot_id: UUID | None
    market_id: str
    event_id: str
    pass_number: int
    symbol: str
    direction: Direction
    classification_probability: float | None
    predicted_peak_percent: float
    predicted_target_price: float
    realized_move_at_entry: float
    remaining_gap: float
    directions_agree: bool
    target_reached: bool | None = None
    target_reached_at: datetime | None = None
    actual_max_favorable: float | None = None
    actual_max_adverse: float | None = None
    actual_direction: str | None = None
    classification_correct: bool | None = None
    regression_error: float | None = None


@dataclass
class Trade:
    trade_id: UUID
    run_id: UUID
    market_id: str
    event_id: str
    question: str
    symbol: str
    asset_name: str
    pass_number: int
    trigger_at: datetime
    entry_at: datetime
    entry_price: float
    quantity: float
    entry_commission: float
    initial_stop: float
    current_stop: float
    highest_price: float
    final_outcome: str | None
    portfolio: str = "polymarket_momentum"
    strategy_branch: str = "momentum"
    resolution: Resolution = "1h"
    direction: Direction = "long"
    lowest_price: float | None = None
    predicted_target_price: float | None = None
    predicted_target_reached: bool | None = None
    range_period: int | None = None
    range_multiplier: float | None = None
    parameter_selection: dict[str, Any] = field(default_factory=dict)
    exit_at: datetime | None = None
    exit_price: float | None = None
    exit_commission: float = 0.0
    exit_reason: str | None = None
    final_mark_price: float | None = None
    maximum_price: float | None = None
    minimum_price: float | None = None
    stop_history: list[dict[str, Any]] = field(default_factory=list)
    graph_path: str | None = None

    @property
    def gross_profit(self) -> float | None:
        price = self.exit_price if self.exit_price is not None else self.final_mark_price
        if price is None:
            return None
        multiplier = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.quantity * multiplier

    @property
    def net_profit(self) -> float | None:
        gross = self.gross_profit
        if gross is None:
            return None
        return gross - self.entry_commission - self.exit_commission

    @property
    def maximum_profit(self) -> float | None:
        price = self.maximum_price if self.direction == "long" else self.minimum_price
        if price is None:
            return None
        multiplier = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.quantity * multiplier

    @property
    def maximum_loss(self) -> float | None:
        price = self.minimum_price if self.direction == "long" else self.maximum_price
        if price is None:
            return None
        multiplier = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry_price) * self.quantity * multiplier
