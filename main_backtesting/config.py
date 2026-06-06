from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BacktestConfig:
    start: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end: datetime = datetime(2026, 2, 1, tzinfo=timezone.utc)
    threshold: float = 0.55
    minimum_days_remaining: float = 5.0
    maximum_days_remaining: float = 60.0
    trade_notional: float = 1_000.0
    trailing_range_bars: int = 14
    trailing_range_multiplier: float = 3.0
    news_lookback: timedelta = timedelta(hours=72)
    max_articles: int = 9
    probability_chunk_days: int = 10
    request_concurrency: int = 5
    output_dir: Path = REPO_ROOT / "main_backtesting" / "output"
    graph_dir: Path = REPO_ROOT / "main_backtesting" / "output" / "graphs"
    report_dir: Path = REPO_ROOT / "main_backtesting" / "output" / "reports"
    temp_dir: Path = REPO_ROOT / "main_backtesting" / "output" / "temp"
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
