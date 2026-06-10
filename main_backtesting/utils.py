from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def input_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def chunks(values: list[T], size: int) -> Iterator[list[T]]:
    if size <= 0:
        raise ValueError("Batch size must be positive")
    for index in range(0, len(values), size):
        yield values[index : index + size]


def direct_subject_symbol(question: str) -> str | None:
    matches = re.findall(r"\(([A-Z][A-Z0-9.\-]{0,9})\)", question.upper())
    return matches[0] if matches else None


def event_archetype(
    tags: Iterable[str],
    *,
    question: str = "",
    symbol: str | None = None,
) -> str | None:
    normalized = {tag.lower().strip() for tag in tags}
    normalized_question = question.lower()
    if "earnings" in normalized or "earnings" in normalized_question:
        if "beat" in normalized_question and (
            "quarterly earnings" in normalized_question
            or "quarterly eps estimate" in normalized_question
        ):
            subject = direct_subject_symbol(question)
            if symbol is not None and subject != symbol.upper():
                return None
            return "company_quarterly_earnings_beat"
        if symbol is not None:
            return None
        return "company_earnings"
    rules = [
        ({"nfp", "nonfarm-payroll", "jobs", "unemployment"}, "macro_labor"),
        ({"fed", "fed-rates", "global-rates"}, "macro_rates"),
        ({"inflation", "cpi"}, "macro_inflation"),
        ({"gdp", "growth"}, "macro_growth"),
        ({"housing", "real-estate"}, "macro_housing"),
        ({"strait-of-hormuz", "oil", "shipping"}, "geopolitics_energy"),
        ({"ukraine", "russia"}, "geopolitics_europe"),
        ({"iran", "israel", "middle-east"}, "geopolitics_middle_east"),
        ({"fda"}, "company_fda"),
        ({"ipo", "ipos"}, "company_ipo"),
    ]
    for candidates, label in rules:
        if normalized & candidates:
            return label
    return None
