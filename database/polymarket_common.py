from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

DEFAULT_DATA_START = "2022-01-01T00:00:00Z"
DEFAULT_DATA_END = "2026-05-01T23:59:59Z"
MIN_DURATION_DAYS = 5.0
MAX_DURATION_DAYS = 60.0
PRICE_FIDELITY_MINUTES = 1
PRICE_CHUNK_SECONDS = 10 * 86400
INSERT_BATCH_SIZE = 5_000

TARGET_TAG_SLUGS = [
    "equities",
    "earnings",
    "kpis",
    "economy",
    "macro-indicators",
    "business",
    "monthly",
    "hit-price",
    "finance-updown",
    "pyth-finance",
    "stocks",
    "geopolitics",
    "oil",
    "iran",
    "us-x-iran",
    "strait-of-hormuz",
    "ai",
    "big-tech",
    "tech",
    "privates",
]


@dataclass(frozen=True)
class EventRecord:
    event: dict[str, Any]
    matched_tags: set[str]


@dataclass(frozen=True)
class MarketRecord:
    event_id: str
    market_id: str
    condition_id: str | None
    yes_token_id: str | None
    no_token_id: str | None
    question: str | None
    outcomes: list[str]
    outcome_prices: list[float]
    history_start: datetime
    history_end: datetime
    raw_market: dict[str, Any]
    market_detail: dict[str, Any]


def parse_dt(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def json_array(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise TypeError(f"Expected JSON array, got {type(parsed).__name__}")
        return parsed
    raise TypeError(f"Expected list or JSON array string, got {type(value).__name__}")


def text_array(value: Any) -> list[str]:
    return [str(item) for item in json_array(value) if item is not None]


def float_array(value: Any) -> list[float]:
    return [float(item) for item in json_array(value) if item is not None and item != ""]


def optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "t", "yes", "y", "1"}:
            return True
        if normalized in {"false", "f", "no", "n", "0"}:
            return False
    raise TypeError(f"Expected boolean-like value, got {type(value).__name__}: {value!r}")


def duration_days(event: dict[str, Any]) -> float | None:
    created = parse_dt(event.get("createdAt"))
    end = parse_dt(event.get("endDate"))
    if created is None or end is None:
        return None
    return (end - created).total_seconds() / 86400.0


def event_tag_labels(event: dict[str, Any]) -> list[str]:
    tags = event.get("tags") or []
    if not isinstance(tags, list):
        raise TypeError("event.tags must be a list")
    labels: list[str] = []
    for tag in tags:
        if not isinstance(tag, dict):
            raise TypeError("event.tags items must be dicts")
        label = tag.get("slug") or tag.get("label") or tag.get("name")
        if label:
            labels.append(str(label))
    return labels


def yes_no_tokens(outcomes: list[str], token_ids: list[str]) -> tuple[str | None, str | None]:
    if len(outcomes) != len(token_ids):
        raise ValueError("outcomes and clobTokenIds length mismatch")
    token_by_outcome = {outcome.lower(): token_id for outcome, token_id in zip(outcomes, token_ids)}
    return token_by_outcome.get("yes"), token_by_outcome.get("no")


def infer_resolution(outcomes: list[str], outcome_prices: list[float], closed: bool) -> tuple[str | None, bool | None, str]:
    if not closed:
        return None, None, "open"
    if not outcomes or not outcome_prices:
        return None, None, "closed_unpriced"
    if len(outcomes) != len(outcome_prices):
        raise ValueError("outcomes and outcome_prices length mismatch")

    by_outcome = {outcome.lower(): price for outcome, price in zip(outcomes, outcome_prices)}
    if "yes" in by_outcome and by_outcome["yes"] >= 0.95:
        return "Yes", True, "resolved"
    if "no" in by_outcome and by_outcome["no"] >= 0.95:
        return "No", False, "resolved"

    best_index = max(range(len(outcome_prices)), key=outcome_prices.__getitem__)
    best_outcome = outcomes[best_index]
    best_price = outcome_prices[best_index]
    if best_price >= 0.95:
        return best_outcome, None, "resolved_non_binary"
    return best_outcome, None, "closed_ambiguous"


def jsonb(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def normalize_question(question: str | None) -> str:
    if not question:
        return ""
    normalized = question.lower()
    normalized = re.sub(r"[^a-z0-9\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def market_group_id(question: str | None) -> str:
    words = normalize_question(question).split()
    return "_".join(words[:4]) if words else "unknown_market_group"
