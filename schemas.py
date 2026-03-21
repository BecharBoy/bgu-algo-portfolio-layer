from typing import Any, Dict, List, TypedDict


class Signal(TypedDict, total=False):
    signal_id: str
    timestamp: str
    strategy: str
    symbol: str
    action: str
    price_reference: float
    reason: str
    metadata: Dict[str, Any]


class OrderIntent(TypedDict, total=False):
    order_id: str
    strategy: str
    symbol: str
    action: str
    quantity: int
    order_type: str
    tif: str
    metadata: Dict[str, Any]


def validate_signal(signal: Dict[str, Any]) -> bool:
    # TODO: Validate required signal fields and allowed action values.
    # TODO: Add schema versioning for forward compatibility.
    pass


def validate_order_intent(order: Dict[str, Any]) -> bool:
    # TODO: Validate required order fields before broker placement.
    pass


def deduplicate_signals(signals: List[Signal]) -> List[Signal]:
    # TODO: Deduplicate by (strategy, symbol, action, timestamp bucket) policy.
    pass
