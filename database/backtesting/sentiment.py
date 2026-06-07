from __future__ import annotations

import asyncio
from typing import Any

from main_backtesting.models import NewsArticle, SentimentResult


def _load_pipeline() -> Any:
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError(
            "FinBERT requires transformers and torch. Install them in the project "
            "environment before running the full backtest."
        ) from exc
    import torch

    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)


class FinbertSentimentAnalyzer:
    def __init__(self, batch_size: int = 32) -> None:
        self._pipeline: Any | None = None
        self.batch_size = batch_size

    async def analyze(self, articles: list[NewsArticle]) -> SentimentResult:
        if not articles:
            return SentimentResult("insufficient", 0.0, 0, 0, 0, [])
        if self._pipeline is None:
            self._pipeline = await asyncio.to_thread(_load_pipeline)

        texts = [f"{article.title}. {article.text}"[:12_000] for article in articles]
        results = await asyncio.to_thread(
            self._pipeline,
            texts,
            truncation=True,
            max_length=512,
            batch_size=self.batch_size,
        )
        details = [
            {
                "url": article.url,
                "published_at": article.published_at,
                "label": str(result["label"]).lower(),
                "score": float(result["score"]),
            }
            for article, result in zip(articles, results)
        ]
        counts = {
            label: sum(item["label"] == label for item in details)
            for label in ("positive", "neutral", "negative")
        }
        label = (
            "positive"
            if counts["positive"] > counts["negative"] and counts["positive"] >= 1
            else "negative_or_unclear"
        )
        score = sum(item["score"] for item in details if item["label"] == "positive")
        score = score / counts["positive"] if counts["positive"] else 0.0
        return SentimentResult(
            label=label,
            score=score,
            positive_count=counts["positive"],
            neutral_count=counts["neutral"],
            negative_count=counts["negative"],
            details=details,
        )
