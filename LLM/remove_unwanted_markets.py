from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from LLM.ollama_client import OllamaClient
from main_backtesting.models import SourceEvent, SourceMarket


class EventRelevanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    relevant_to_financial_markets: bool
    reason: str = Field(min_length=10, max_length=500)


class BatchedEventDecision(EventRelevanceDecision):
    event_id: str


class BatchedEventDecisions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[BatchedEventDecision]


class BatchedMarketDecision(EventRelevanceDecision):
    market_id: str


class BatchedMarketDecisions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[BatchedMarketDecision]


SYSTEM_PROMPT = """
You are a strict event filter for a stock-market backtest.

Decide whether the supplied prediction-market event could plausibly affect publicly
traded stocks or equity ETFs through a concrete economic channel.

Accept macroeconomic releases, central banks, corporate events, regulation, major
geopolitical shocks, trade, energy supply, and events with direct company exposure.
Reject sports, entertainment, celebrity trivia, generic politics without an economic
channel, mechanical stock-price target markets, and crypto-only events.

Do not predict the event outcome. Return only the supplied JSON schema.
""".strip()


async def classify_event(
    ollama: OllamaClient,
    event: SourceEvent,
) -> EventRelevanceDecision:
    return await ollama.structured(
        system_prompt=SYSTEM_PROMPT,
        payload={
            "event_id": event.event_id,
            "title": event.title,
            "tags": event.tags,
            "created_at": event.created_at,
            "end_at": event.end_at,
        },
        response_model=EventRelevanceDecision,
        max_tokens=150,
    )


async def classify_events(
    ollama: OllamaClient,
    events: list[SourceEvent],
) -> list[BatchedEventDecision]:
    if not events:
        return []
    if len(events) == 1:
        decision = await classify_event(ollama, events[0])
        return [
            BatchedEventDecision(
                event_id=events[0].event_id,
                relevant_to_financial_markets=decision.relevant_to_financial_markets,
                reason=decision.reason,
            )
        ]
    try:
        response = await ollama.structured(
            system_prompt=SYSTEM_PROMPT
            + "\nReturn exactly one decision for every supplied event_id.",
            payload={
                "events": [
                    {
                        "event_id": event.event_id,
                        "title": event.title,
                        "tags": event.tags,
                        "created_at": event.created_at,
                        "end_at": event.end_at,
                    }
                    for event in events
                ]
            },
            response_model=BatchedEventDecisions,
            max_tokens=max(250, len(events) * 150),
        )
        # Deduplicate: keep first occurrence of each event_id
        seen: dict[str, BatchedEventDecision] = {}
        for item in response.decisions:
            if item.event_id not in seen:
                seen[item.event_id] = item
        expected = {event.event_id for event in events}
        if seen.keys() >= expected:
            return [seen[eid] for eid in expected]
        # Some missing — retry just the missing ones by splitting
        missing = [e for e in events if e.event_id not in seen]
        print(
            f"[event-filter] Ollama dropped {len(missing)} event(s) from batch of "
            f"{len(events)}, splitting to retry"
        )
        retried = await classify_events(ollama, missing)
        return list(seen.values()) + retried
    except (ValidationError, ValueError) as exc:
        mid = len(events) // 2
        print(
            f"[event-filter] Batch of {len(events)} failed ({type(exc).__name__}), "
            f"splitting into {mid}+{len(events) - mid}"
        )
        left = await classify_events(ollama, events[:mid])
        right = await classify_events(ollama, events[mid:])
        return left + right


async def _classify_single_market(
    ollama: OllamaClient,
    market: SourceMarket,
) -> BatchedMarketDecision:
    """Classify one market with the single-item schema to avoid batch issues."""
    decision = await ollama.structured(
        system_prompt=SYSTEM_PROMPT,
        payload={
            "market_id": market.market_id,
            "event_title": market.event_title,
            "market_question": market.question,
            "tags": market.tags,
            "created_at": market.created_at,
            "end_at": market.end_at,
        },
        response_model=EventRelevanceDecision,
        max_tokens=150,
    )
    return BatchedMarketDecision(
        market_id=market.market_id,
        relevant_to_financial_markets=decision.relevant_to_financial_markets,
        reason=decision.reason,
    )


async def classify_markets(
    ollama: OllamaClient,
    markets: list[SourceMarket],
) -> list[BatchedMarketDecision]:
    if not markets:
        return []
    if len(markets) == 1:
        return [await _classify_single_market(ollama, markets[0])]
    try:
        response = await ollama.structured(
            system_prompt=SYSTEM_PROMPT
            + "\nJudge each specific market question independently. Return exactly one "
            "decision for every supplied market_id.",
            payload={
                "markets": [
                    {
                        "market_id": market.market_id,
                        "event_title": market.event_title,
                        "market_question": market.question,
                        "tags": market.tags,
                        "created_at": market.created_at,
                        "end_at": market.end_at,
                    }
                    for market in markets
                ]
            },
            response_model=BatchedMarketDecisions,
            max_tokens=max(250, len(markets) * 150),
        )
        # Deduplicate: keep first occurrence of each market_id
        seen: dict[str, BatchedMarketDecision] = {}
        for item in response.decisions:
            if item.market_id not in seen:
                seen[item.market_id] = item
        expected = {market.market_id for market in markets}
        if seen.keys() >= expected:
            return [seen[mid] for mid in expected]
        # Some missing — retry just the missing ones by splitting
        missing = [m for m in markets if m.market_id not in seen]
        print(
            f"[market-filter] Ollama dropped {len(missing)} market(s) from batch of "
            f"{len(markets)}, splitting to retry"
        )
        retried = await classify_markets(ollama, missing)
        return list(seen.values()) + retried
    except (ValidationError, ValueError) as exc:
        mid = len(markets) // 2
        print(
            f"[market-filter] Batch of {len(markets)} failed ({type(exc).__name__}), "
            f"splitting into {mid}+{len(markets) - mid}"
        )
        left = await classify_markets(ollama, markets[:mid])
        right = await classify_markets(ollama, markets[mid:])
        return left + right

