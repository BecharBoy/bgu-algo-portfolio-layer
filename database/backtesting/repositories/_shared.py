from __future__ import annotations

import json
import math
from typing import Any

SCHEMA = "checking_relevant_events"


def json_safe_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe_value(item) for item in value]
    return value


def json_text(value: Any) -> str:
    return json.dumps(
        json_safe_value(value),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    )


def json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value
