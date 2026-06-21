from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from portfolio.models import DecisionStatus, TradeCandidate
from portfolio.portfolio import Portfolio


@dataclass(frozen=True, order=True)
class TimelineEvent:
    sort_rank: int
    timestamp: datetime
    entry_at: datetime
    market_id: str
    symbol: str
    kind: str
    candidate: TradeCandidate


def build_timeline(candidates: Iterable[TradeCandidate]) -> list[TimelineEvent]:
    events: list[TimelineEvent] = []
    for candidate in candidates:
        if not candidate.consumes_capital:
            continue
        events.append(
            TimelineEvent(
                sort_rank=1,
                timestamp=candidate.entry_at,
                entry_at=candidate.entry_at,
                market_id=candidate.market_id,
                symbol=candidate.symbol,
                kind="entry",
                candidate=candidate,
            )
        )
        events.append(
            TimelineEvent(
                sort_rank=0,
                timestamp=candidate.exit_at,
                entry_at=candidate.entry_at,
                market_id=candidate.market_id,
                symbol=candidate.symbol,
                kind="exit",
                candidate=candidate,
            )
        )
    return sorted(events)


def replay_portfolio(portfolio: Portfolio, candidates: list[TradeCandidate]) -> Portfolio:
    for event in build_timeline(candidates):
        if event.kind == "exit":
            portfolio.close(event.candidate)
            portfolio.record_snapshot(event.timestamp)
            continue
        decision = portfolio.evaluate(event.candidate, timestamp=event.timestamp)
        if decision.status in {DecisionStatus.APPROVED, DecisionStatus.APPROVED_CAPPED}:
            portfolio.build_trade(event.candidate, decision)
        portfolio.record_snapshot(event.timestamp)
    return portfolio
