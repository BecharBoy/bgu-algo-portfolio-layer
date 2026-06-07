from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from LLM.ollama_client import OllamaClient
from main_backtesting.models import Asset, NewsArticle, SentimentResult, SourceMarket


class OllamaSentiment(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    label: Literal["positive", "neutral", "negative", "insufficient"]
    score: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=5, max_length=500)


class BatchedOllamaSentiment(OllamaSentiment):
    request_id: str


class BatchedOllamaSentiments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentiments: list[BatchedOllamaSentiment]


SYSTEM_PROMPT = """
Judge whether the supplied historical articles are positive, neutral, or negative for
the supplied stock in the context of the prediction-market question. Use only the
articles supplied. This result is recorded for comparison and does not make trades.
Return only the supplied JSON schema.
""".strip()
BATCH_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + "\nReturn one independent sentiment for every request_id and echo request_id."
)


def batch_request_payload(
    requests: list[tuple[str, SourceMarket, Asset, list[NewsArticle]]],
) -> dict[str, object]:
    return {
        "requests": [
            {
                "request_id": request_id,
                "market_question": market.question,
                "asset": {
                    "symbol": asset.symbol,
                    "asset_name": asset.asset_name,
                    "reason_in_world": asset.reason,
                },
                "articles": [
                    {
                        "title": article.title,
                        "published_at": article.published_at,
                        "excerpt": article.text[:900],
                    }
                    for article in articles
                ],
            }
            for request_id, market, asset, articles in requests
            if articles
        ]
    }


async def analyze_with_ollama(
    ollama: OllamaClient,
    *,
    market: SourceMarket,
    asset: Asset,
    articles: list[NewsArticle],
) -> SentimentResult:
    if not articles:
        return SentimentResult("insufficient", 0.0, 0, 0, 0, [])
    result = await ollama.structured(
        system_prompt=SYSTEM_PROMPT,
        payload={
            "market_question": market.question,
            "asset": {
                "symbol": asset.symbol,
                "asset_name": asset.asset_name,
                "reason_in_world": asset.reason,
            },
            "articles": [
                {
                    "title": article.title,
                    "published_at": article.published_at,
                    "text": article.text[:4_000],
                }
                for article in articles
            ],
        },
        response_model=OllamaSentiment,
        max_tokens=500,
    )


async def analyze_batch_with_ollama(
    ollama: OllamaClient,
    requests: list[tuple[str, SourceMarket, Asset, list[NewsArticle]]],
) -> dict[str, SentimentResult]:
    populated = [item for item in requests if item[3]]
    results = {
        request_id: SentimentResult("insufficient", 0.0, 0, 0, 0, [])
        for request_id, _, _, articles in requests
        if not articles
    }
    if not populated:
        return results

    response = await ollama.structured(
        system_prompt=BATCH_SYSTEM_PROMPT,
        payload=batch_request_payload(populated),
        response_model=BatchedOllamaSentiments,
        max_tokens=max(600, len(populated) * 500),
    )
    expected = {request_id for request_id, _, _, _ in populated}
    actual = {item.request_id for item in response.sentiments}
    if len(actual) != len(response.sentiments) or actual != expected:
        raise ValueError(
            f"Sentiment batch identifiers mismatch: expected={expected} actual={actual}"
        )
    for item in response.sentiments:
        results[item.request_id] = SentimentResult(
            label=item.label,
            score=item.score,
            positive_count=int(item.label == "positive"),
            neutral_count=int(item.label == "neutral"),
            negative_count=int(item.label == "negative"),
            details=[item.model_dump()],
        )
    return results
