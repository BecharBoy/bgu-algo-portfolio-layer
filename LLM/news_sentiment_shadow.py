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


SYSTEM_PROMPT = """
Judge whether the supplied historical articles are positive, neutral, or negative for
the supplied stock in the context of the prediction-market question. Use only the
articles supplied. This result is recorded for comparison and does not make trades.
Return only the supplied JSON schema.
""".strip()


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
    details = [result.model_dump()]
    return SentimentResult(
        label=result.label,
        score=result.score,
        positive_count=int(result.label == "positive"),
        neutral_count=int(result.label == "neutral"),
        negative_count=int(result.label == "negative"),
        details=details,
    )

