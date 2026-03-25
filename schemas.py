from typing import Any, Dict, List, TypedDict, Optional
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
    is_exit: bool


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
    required_keys = {"signal_id", "timestamp", "strategy", "symbol", "action", "price_reference"}
    if not required_keys.issubset(signal.keys()):
        return False
    if signal["action"] not in {"BUY", "SELL"}:
        return False
    if not isinstance(signal["price_reference"], (int, float)) or signal["price_reference"] <= 0:
        return False
    if "metadata" in signal and not isinstance(signal["metadata"], dict):
        return False
    return True


def validate_order_intent(order: Dict[str, Any]) -> bool:
    required_keys = {"order_id", "strategy", "symbol", "action", "quantity", "order_type", "tif"}
    if not required_keys.issubset(order.keys()):
        return False
    allowed_strategies = {"MeanReversionMomentum", "StatArbStrategy", "CointegrationArb"}
    if order["strategy"] not in allowed_strategies:
        return False
    if order["action"] not in {"BUY", "SELL"}:
        return False
    if type(order["quantity"]) is not int or order["quantity"] <= 0:
        return False
    return True


def deduplicate_signals(signals: List[Signal]) -> List[Signal]:
    unique_signals: Dict[str, Signal] = {}
    for sig in signals:
        if not validate_signal(sig):
            continue
        bucket_key = f"{sig['strategy']}_{sig['symbol']}_{sig['action']}"
        unique_signals[bucket_key] = sig
    return list(unique_signals.values())


def aggregate_signals(signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cross-strategy conflict resolver with pair-integrity protection.

    KEY CHANGE: exit signals (is_exit=True) are ALWAYS passed through and
    never participate in conflict resolution. Only new entry signals from
    different strategies conflict if they disagree on direction.

    Step 1 — separate exits from entries.
    Step 2 — resolve per-symbol direction conflicts among entries only.
    Step 3 — enforce pair integrity on entries: if one leg was dropped,
              drop the partner leg too.
    Step 4 — recombine exits + surviving entries.
    """
    exits:   List[Dict[str, Any]] = []
    entries: List[Dict[str, Any]] = []

    for sig in signals:
        if sig.get("is_exit"):
            exits.append(sig)
        else:
            entries.append(sig)

    # ── Step 2: per-symbol cross-strategy conflict resolution (entries only) ─
    by_symbol: Dict[str, List[Dict[str, Any]]] = {}
    for sig in entries:
        sym = sig.get("symbol")
        if not sym:
            continue
        by_symbol.setdefault(sym, []).append(sig)

    result:          List[Dict[str, Any]] = []
    dropped_symbols: set                  = set()

    for symbol, sym_signals in by_symbol.items():
        actions = {s["action"] for s in sym_signals if "action" in s}
        if len(actions) > 1:
            strategies_in_conflict = [s.get("strategy", "unknown") for s in sym_signals]
            logging.warning(
                f"aggregate_signals: conflict on {symbol} "
                f"from {strategies_in_conflict} — cancelling both."
            )
            dropped_symbols.add(symbol)
            continue
        result.append(sym_signals[0])

    # ── Step 3: pair integrity among entries ─────────────────────────────────
    orphaned_pair_ids: set = set()
    for sig in entries:
        if sig.get("symbol") in dropped_symbols:
            pid = (sig.get("metadata") or {}).get("pair_id")
            if pid:
                orphaned_pair_ids.add(pid)

    if orphaned_pair_ids:
        before = len(result)
        result = [
            s for s in result
            if (s.get("metadata") or {}).get("pair_id") not in orphaned_pair_ids
        ]
        logging.warning(
            f"aggregate_signals: dropped {before - len(result)} orphaned entry leg(s) "
            f"for pair_ids={orphaned_pair_ids} to preserve pair integrity."
        )

    # ── Step 4: exits always pass through ────────────────────────────────────
    return exits + result
