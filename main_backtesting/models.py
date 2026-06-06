from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


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
    final_outcome: str | None


@dataclass(frozen=True)
class ProbabilityPoint:
    timestamp: datetime
    probability: float


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
        return (price - self.entry_price) * self.quantity

    @property
    def net_profit(self) -> float | None:
        gross = self.gross_profit
        if gross is None:
            return None
        return gross - self.entry_commission - self.exit_commission

    @property
    def maximum_profit(self) -> float | None:
        if self.maximum_price is None:
            return None
        return (self.maximum_price - self.entry_price) * self.quantity

    @property
    def maximum_loss(self) -> float | None:
        if self.minimum_price is None:
            return None
        return (self.minimum_price - self.entry_price) * self.quantity
