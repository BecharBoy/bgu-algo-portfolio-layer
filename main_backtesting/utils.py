from __future__ import annotations

import hashlib
import json
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


def event_archetype(tags: Iterable[str]) -> str:
    normalized = {tag.lower().strip() for tag in tags}
    rules = [
        ({"nfp", "nonfarm-payroll", "jobs", "unemployment"}, "macro_labor"),
        ({"fed", "fed-rates", "global-rates"}, "macro_rates"),
        ({"inflation", "cpi"}, "macro_inflation"),
        ({"gdp", "growth"}, "macro_growth"),
        ({"housing", "real-estate"}, "macro_housing"),
        ({"strait-of-hormuz", "oil", "shipping"}, "geopolitics_energy"),
        ({"ukraine", "russia"}, "geopolitics_europe"),
        ({"iran", "israel", "middle-east"}, "geopolitics_middle_east"),
        ({"earnings"}, "company_earnings"),
        ({"fda"}, "company_fda"),
        ({"ipo", "ipos"}, "company_ipo"),
    ]
    for candidates, label in rules:
        if normalized & candidates:
            return label
    return "uncategorized"
