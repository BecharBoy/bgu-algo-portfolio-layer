from __future__ import annotations

import json
from typing import Any

SCHEMA = "checking_relevant_events"


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def json_value(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value
