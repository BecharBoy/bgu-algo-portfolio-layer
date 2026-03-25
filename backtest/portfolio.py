from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# IBKR Pro Fixed commission schedule (US equities)
# ---------------------------------------------------------------------------
_IBKR_RATE    = 0.005
_IBKR_MIN     = 1.00
_IBKR_MAX_PCT = 0.01


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_entry_price: float
    entry_date: str
    entry_commission: float = 0.0
    strategy: str = ""                          # owning strategy name
    metadata: dict = field(default_factory=dict) # e.g. pair_id, hedge_ratio


class Portfolio:
    """
    Isolated per-strategy portfolio.

    Each strategy gets its own Portfolio instance seeded with half the total
    capital. Positions are stamped with the owning strategy name so that
    generate_signals() for strategy A can never see strategy B's positions.

    Sizing rule: 15% of available CASH per new entry (not MTM equity).
    """

    def __init__(
        self,
        initial_cash: float,
        strategy_name: str = "",
        slippage: float = 0.0005,
    ) -> None:
        self.initial_cash  = float(initial_cash)
        self.cash          = float(initial_cash)
        self.strategy_name = strategy_name
        self.slippage      = float(slippage)
        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict[str, Any]] = []
        self.rejection_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Price / commission helpers
    # ------------------------------------------------------------------

    def _effective_price(self, action: str, price: float) -> float:
        return price * (1.0 + self.slippage) if action.upper() == "BUY" else price * (1.0 - self.slippage)

    def _commission_paid(self, quantity: int, price: float) -> float:
        notional = abs(quantity * price)
        raw      = abs(quantity) * _IBKR_RATE
        return min(max(raw, _IBKR_MIN), notional * _IBKR_MAX_PCT)

    # ------------------------------------------------------------------
    # Sizing  (15% of available cash for new entries)
    # ------------------------------------------------------------------

    def size_signal(
        self,
        signal: dict[str, Any],
        execution_price: float,
        allocation: float = 0.15,
    ) -> int:
        """
        Returns the share quantity for a signal.

        For exit signals: returns the full open quantity (or 0 if no position).
        For entry signals: sizes to `allocation` fraction of available cash.
        """
        symbol  = signal["symbol"]
        action  = signal["action"].upper()
        current = self.positions.get(symbol)

        # --- Exit path ---
        if signal.get("is_exit"):
            if current is None:
                return 0
            return abs(current.quantity)

        # --- Close-by-direction path (same strategy, opposite direction) ---
        if current is not None:
            if current.quantity > 0 and action == "SELL":
                return abs(current.quantity)
            if current.quantity < 0 and action == "BUY":
                return abs(current.quantity)
            # Same direction: no pyramiding
            return 0

        # --- New entry: size from available cash ---
        if self.cash <= 0:
            return 0
        fill_price = self._effective_price(action, execution_price)
        position_value = self.cash * allocation
        return max(int(position_value / fill_price), 0)

    # ------------------------------------------------------------------
    # Pre-flight: can we afford ALL legs of a pair atomically?
    # ------------------------------------------------------------------

    def can_afford_pair(
        self,
        legs: list[dict[str, Any]],
        prices: dict[str, float],
        allocation: float = 0.15,
    ) -> bool:
        """
        Dry-run: simulate cash deduction for every NEW ENTRY leg sequentially.
        Returns False if any leg would push simulated_cash below zero.
        Exits and close signals are not counted (they add cash).
        """
        simulated_cash = self.cash
        for leg in legs:
            if leg.get("is_exit"):
                continue
            symbol = leg["symbol"]
            action = leg["action"].upper()
            current = self.positions.get(symbol)
            # If it's a close (opposite direction on open position) skip — adds cash
            if current is not None:
                if (current.quantity > 0 and action == "SELL") or (current.quantity < 0 and action == "BUY"):
                    continue
                # Same direction — would be rejected anyway
                continue
            price = prices.get(symbol)
            if not price:
                return False
            fill_price = self._effective_price(action, price)
            qty        = max(int(simulated_cash * allocation / fill_price), 0)
            if qty <= 0:
                return False
            cost = qty * fill_price + self._commission_paid(qty, fill_price)
            simulated_cash -= cost
            if simulated_cash < 0:
                return False
        return True

    # ------------------------------------------------------------------
    # Signal execution router
    # ------------------------------------------------------------------

    def execute_signal(
        self,
        signal: dict[str, Any],
        execution_price: float,
        date: str,
        quantity: int,
    ) -> bool:
        action = signal["action"].upper()
        symbol = signal["symbol"]

        if quantity <= 0:
            self._log_rejection(signal, date, "quantity_zero")
            return False

        current = self.positions.get(symbol)

        if action == "BUY" and current is not None and current.quantity < 0:
            return self._close_short(signal, execution_price, date, quantity)
        if action == "SELL" and current is not None and current.quantity > 0:
            return self._close_long(signal, execution_price, date, quantity)
        if action == "BUY":
            return self._open_long(signal, execution_price, date, quantity)
        if action == "SELL":
            return self._open_short(signal, execution_price, date, quantity)

        self._log_rejection(signal, date, f"unsupported_action:{action}")
        return False

    # ------------------------------------------------------------------
    # Execution internals
    # ------------------------------------------------------------------

    def _open_long(self, signal: dict, price: float, date: str, qty: int) -> bool:
        symbol     = signal["symbol"]
        fill_price = self._effective_price("BUY", price)
        commission = self._commission_paid(qty, fill_price)
        cash_needed = qty * fill_price + commission

        if self.cash < cash_needed:
            self._log_rejection(signal, date, "insufficient_cash")
            return False

        self.cash -= cash_needed
        current    = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=qty,
                avg_entry_price=fill_price, entry_date=date,
                entry_commission=commission,
                strategy=signal.get("strategy", self.strategy_name),
                metadata=dict(signal.get("metadata") or {}),
            )
        else:
            total_qty  = current.quantity + qty
            current.avg_entry_price  = (current.quantity * current.avg_entry_price + qty * fill_price) / total_qty
            current.quantity         = total_qty
            current.entry_commission += commission

        self._log_trade(signal=signal, date=date, qty=qty, side="BUY",
                        price=fill_price, commission=commission,
                        realized_pnl=None, hold_days=None)
        return True

    def _open_short(self, signal: dict, price: float, date: str, qty: int) -> bool:
        symbol     = signal["symbol"]
        fill_price = self._effective_price("SELL", price)
        commission = self._commission_paid(qty, fill_price)
        proceeds   = qty * fill_price - commission

        self.cash += proceeds
        current    = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = Position(
                symbol=symbol, quantity=-qty,
                avg_entry_price=fill_price, entry_date=date,
                entry_commission=commission,
                strategy=signal.get("strategy", self.strategy_name),
                metadata=dict(signal.get("metadata") or {}),
            )
        else:
            total_qty  = abs(current.quantity) + qty
            current.avg_entry_price  = (abs(current.quantity) * current.avg_entry_price + qty * fill_price) / total_qty
            current.quantity         = -total_qty
            current.entry_commission += commission

        self._log_trade(signal=signal, date=date, qty=qty, side="SELL",
                        price=fill_price, commission=commission,
                        realized_pnl=None, hold_days=None)
        return True

    def _close_long(self, signal: dict, price: float, date: str, qty: int) -> bool:
        symbol  = signal["symbol"]
        current = self.positions.get(symbol)
        if current is None or current.quantity <= 0:
            self._log_rejection(signal, date, "no_long_position")
            return False

        shares          = min(qty, current.quantity)
        fill_price      = self._effective_price("SELL", price)
        commission      = self._commission_paid(shares, fill_price)
        prop_entry_comm = current.entry_commission * (shares / current.quantity)
        realized_pnl    = (fill_price - current.avg_entry_price) * shares - commission - prop_entry_comm

        self.cash               += shares * fill_price - commission
        current.quantity        -= shares
        current.entry_commission -= prop_entry_comm
        hold_days = self._holding_days(current.entry_date, date)
        if current.quantity == 0:
            del self.positions[symbol]

        self._log_trade(signal=signal, date=date, qty=shares, side="SELL",
                        price=fill_price, commission=commission,
                        realized_pnl=realized_pnl, hold_days=hold_days)
        return True

    def _close_short(self, signal: dict, price: float, date: str, qty: int) -> bool:
        symbol  = signal["symbol"]
        current = self.positions.get(symbol)
        if current is None or current.quantity >= 0:
            self._log_rejection(signal, date, "no_short_position")
            return False

        open_qty        = abs(current.quantity)
        shares          = min(qty, open_qty)
        fill_price      = self._effective_price("BUY", price)
        commission      = self._commission_paid(shares, fill_price)
        prop_entry_comm = current.entry_commission * (shares / open_qty)
        realized_pnl    = (current.avg_entry_price - fill_price) * shares - commission - prop_entry_comm

        self.cash               -= shares * fill_price + commission
        current.quantity        += shares
        current.entry_commission -= prop_entry_comm
        hold_days = self._holding_days(current.entry_date, date)
        if current.quantity == 0:
            del self.positions[symbol]

        self._log_trade(signal=signal, date=date, qty=shares, side="BUY",
                        price=fill_price, commission=commission,
                        realized_pnl=realized_pnl, hold_days=hold_days)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _holding_days(self, entry_date: str, exit_date: str) -> int:
        from datetime import datetime
        return max((datetime.fromisoformat(str(exit_date)) - datetime.fromisoformat(str(entry_date))).days, 0)

    def _log_trade(self, *, signal, date, qty, side, price, commission, realized_pnl, hold_days):
        self.trade_log.append({
            "date":         date,
            "symbol":       signal["symbol"],
            "strategy":     signal.get("strategy"),
            "action":       side,
            "quantity":     qty,
            "price":        price,
            "commission":   commission,
            "slippage":     self.slippage,
            "realized_pnl": realized_pnl,
            "hold_days":    hold_days,
            "reason":       signal.get("reason"),
            "metadata":     signal.get("metadata"),
            "signal_id":    signal.get("signal_id"),
            "status":       "filled",
        })

    def _log_rejection(self, signal, date, reason):
        self.rejection_log.append({
            "date":      date,
            "symbol":    signal.get("symbol"),
            "strategy":  signal.get("strategy"),
            "action":    signal.get("action"),
            "reason":    reason,
            "signal_id": signal.get("signal_id"),
            "status":    "rejected",
        })

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get_equity(self, current_prices: dict[str, float]) -> float:
        equity = self.cash
        for symbol, pos in self.positions.items():
            equity += pos.quantity * current_prices.get(symbol, pos.avg_entry_price)
        return equity

    def get_positions(self) -> dict[str, Position]:
        """Returns the strategy's own Position objects keyed by symbol."""
        return dict(self.positions)

    def get_trade_log(self) -> list[dict[str, Any]]:
        return list(self.trade_log)

    def get_rejection_log(self) -> list[dict[str, Any]]:
        return list(self.rejection_log)
