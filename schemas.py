from typing import Any, Dict, List, TypedDict
import logging


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
    allowed_strategies = {"MeanReversionMomentum", "StatArbStrategy"}
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


def aggregate_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cross-strategy conflict resolver.

    Called BEFORE deduplicate_signals in Portfolio.run_cycle().
    Groups all signals by symbol. If two strategies disagree on
    direction for the same symbol (one BUY, one SELL), both are
    dropped — we do not trade a symbol where strategies conflict.
    If both agree on direction, keep the first signal seen
    (arbitrary but stable tiebreaker when strategies agree).

    This prevents the system from placing both a BUY and a SELL
    for the same symbol in the same cycle, which would waste
    transaction costs and net to zero exposure.
    """
    # Group by symbol
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for sig in signals:
        sym = sig.get("symbol")
        if not sym:
            continue
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(sig)

    result: List[Dict[str, Any]] = []

    for symbol, sym_signals in by_symbol.items():
        actions = {s["action"] for s in sym_signals if "action" in s}

        # Conflict: strategies disagree on direction — cancel both
        if len(actions) > 1:
            strategies_in_conflict = [s.get("strategy", "unknown") for s in sym_signals]
            logging.warning(
                f"aggregate_signals: conflict on {symbol} "
                f"from {strategies_in_conflict} — cancelling both."
            )
            continue

        # No conflict: keep the first signal seen (stable tiebreaker)
        result.append(sym_signals[0])

    return result
