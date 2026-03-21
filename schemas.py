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
    # Check for the existence of all strictly required keys
    required_keys = {"signal_id", "timestamp", "strategy", "symbol", "action", "price_reference"}
    if not required_keys.issubset(signal.keys()):
        return False

    # Validate action values
    if signal["action"] not in {"BUY", "SELL"}:
        return False

    # Validate price_reference is a positive numeric value
    if not isinstance(signal["price_reference"], (int, float)) or signal["price_reference"] <= 0:
        return False

    # Forward compatibility: ensure metadata is a dict if it exists
    if "metadata" in signal and not isinstance(signal["metadata"], dict):
        return False

    return True


def validate_order_intent(order: Dict[str, Any]) -> bool:
    # Check for the existence of all strictly required keys
    required_keys = {"order_id", "strategy", "symbol", "action", "quantity", "order_type", "tif"}
    if not required_keys.issubset(order.keys()):
        return False

    # Validate strategy source (matching the ones registered in Portfolio.py)
    allowed_strategies = {"mean_reversion", "cointegration_arb", "meanreversion", "cointegration"}
    if order["strategy"] not in allowed_strategies:
        return False

    # Validate action values
    if order["action"] not in {"BUY", "SELL"}:
        return False

    # Validate quantity is a strictly positive integer
    if type(order["quantity"]) is not int or order["quantity"] <= 0:
        return False

    return True


def deduplicate_signals(signals: List[Signal]) -> List[Signal]:
    # Deduplicate keeping the latest signal per (strategy, symbol, action) bucket
    unique_signals: Dict[str, Signal] = {}
    
    for sig in signals:
        # Discard structurally invalid signals immediately
        if not validate_signal(sig):
            continue
            
        # Create a unique composite key for the bucket
        bucket_key = f"{sig['strategy']}_{sig['symbol']}_{sig['action']}"
        
        # Overwriting the key guarantees we keep the most recently processed signal 
        # for this specific action and symbol from this strategy.
        unique_signals[bucket_key] = sig

    return list(unique_signals.values())
