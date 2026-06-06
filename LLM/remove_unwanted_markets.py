from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from LLM.ollama_client import OllamaClient
from main_backtesting.models import SourceEvent


class EventRelevanceDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    relevant_to_financial_markets: bool
    reason: str = Field(min_length=10, max_length=500)


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
        max_tokens=400,
    )

