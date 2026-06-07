from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BacktestConfig:
    start: datetime = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end: datetime = datetime(2026, 6, 2, tzinfo=timezone.utc)
    maximum_events: int = 0
    selected_event_ids: tuple[str, ...] = ()
    threshold: float = 0.55
    minimum_days_remaining: float = 5.0
    maximum_days_remaining: float = 60.0
    trade_notional: float = 1_000.0
    trailing_range_bars: int = 14
    trailing_range_multiplier: float = 3.0
    momentum_lookback_bars: int = 14
    polymarket_volume_lookback_hours: int = 24
    polymarket_volume_minimum_history_hours: int = 6
    polymarket_volume_confirmation_ratio: float = 1.0
    news_lookback: timedelta = timedelta(hours=72)
    max_articles: int = 9
    probability_chunk_days: int = 10
    minimum_ml_prior_observations: int = 9
    gdelt_concurrency: int = 8
    gdelt_minimum_request_interval_seconds: float = 5.5
    article_download_concurrency: int = 12
    price_download_concurrency: int = 4
    event_filter_batch_size: int = 1
    asset_world_batch_size: int = 1
    ollama_sentiment_batch_size: int = 1
    finbert_batch_size: int = 32
    historical_data_cutoff: datetime = datetime(2026, 6, 2, tzinfo=timezone.utc)
    event_filter_prompt_version: str = "historical-market-filter-v2"
    asset_world_prompt_version: str = "historical-pass-world-v1"
    ollama_sentiment_prompt_version: str = "historical-sentiment-v1"
    output_root: Path = REPO_ROOT / "main_backtesting" / "output" / "runs"
    included_tags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "ai",
                "big-tech",
                "business",
                "china",
                "commodities",
                "economy",
                "economic-policy",
                "earnings",
                "equities",
                "fda",
                "fed",
                "fed-rates",
                "finance",
                "foreign-policy",
                "gdp",
                "geopolitics",
                "inflation",
                "iran",
                "israel",
                "jobs",
                "kpis",
                "macro-indicators",
                "middle-east",
                "military-action",
                "nfp",
                "oil",
                "politics",
                "real-estate",
                "russia",
                "stocks",
                "strait-of-hormuz",
                "tech",
                "trade-war",
                "ukraine",
                "unemployment",
                "world",
            }
        )
    )
    excluded_tags: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {
                "bitcoin",
                "crypto",
                "crypto-prices",
                "daily",
                "daily-close",
                "finance-updown",
                "hit-price",
                "multi-strikes",
                "pyth-finance",
                "recurring",
                "sports",
                "stock-prices",
                "today",
                "up-or-down",
                "weekly",
            }
        )
    )

    def run_dir(self, run_id: object) -> Path:
        return self.output_root / str(run_id)

    def to_json(self) -> dict[str, Any]:
        values = asdict(self)
        for key, value in values.items():
            if isinstance(value, Path):
                values[key] = str(value)
            elif isinstance(value, frozenset):
                values[key] = sorted(value)
            elif isinstance(value, timedelta):
                values[key] = value.total_seconds()
            elif isinstance(value, datetime):
                values[key] = value.isoformat()
        return values

    @classmethod
    def from_json(cls, values: dict[str, Any]) -> BacktestConfig:
        parsed = dict(values)
        parsed["start"] = datetime.fromisoformat(parsed["start"])
        parsed["end"] = datetime.fromisoformat(parsed["end"])
        if "historical_data_cutoff" in parsed:
            parsed["historical_data_cutoff"] = datetime.fromisoformat(
                parsed["historical_data_cutoff"]
            )
        parsed["news_lookback"] = timedelta(seconds=float(parsed["news_lookback"]))
        parsed["output_root"] = Path(parsed["output_root"])
        parsed["included_tags"] = frozenset(parsed["included_tags"])
        parsed["excluded_tags"] = frozenset(parsed["excluded_tags"])
        parsed["selected_event_ids"] = tuple(parsed.get("selected_event_ids", ()))
        return cls(**parsed)


def hourly_availability_boundary(now: datetime | None = None) -> datetime:
    current = now or datetime.now(timezone.utc)
    return (current - timedelta(days=729)).replace(hour=0, minute=0, second=0, microsecond=0)
