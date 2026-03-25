from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# IBKR Pro Fixed commission schedule (US equities)
#   - $0.005 per share
#   - minimum $1.00 per order
#   - capped at 1.0% of notional
# ---------------------------------------------------------------------------
_IBKR_RATE    = 0.005   # $ per share
_IBKR_MIN     = 1.00    # $ per order
_IBKR_MAX_PCT = 0.01    # 1% of notional cap


@dataclass
class Position:
    symbol: str
    quantity: int
    avg_entry_price: float
    entry_date: str
    entry_commission: float = 0.0


class Portfolio:
    def __init__(
        self,
        initial_cash: float,
        commission: float = _IBKR_RATE,
        slippage: float = 0.0005,
        max_total_exposure: float = 0.95,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission = float(commission)
        self.slippage = float(slippage)
        self.max_total_exposure = float(max_total_exposure)
        self.positions: dict[str, Position] = {}
        self.trade_log: list[dict[str, Any]] = []
        self.rejection_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Price helpers
    # ------------------------------------------------------------------

    def _effective_price(self, action: str, execution_price: float) -> float:
        if action.upper() == "BUY":
            return execution_price * (1.0 + self.slippage)
        return execution_price * (1.0 - self.slippage)

    def _commission_paid(self, quantity: int, price: float) -> float:
        """
        IBKR Pro Fixed schedule:
            raw   = qty * $0.005/share
            floor = $1.00 per order
            cap   = 1% of notional
        """
        notional = abs(quantity * price)
        raw      = abs(quantity) * _IBKR_RATE
        return min(max(raw, _IBKR_MIN), notional * _IBKR_MAX_PCT)

    def _gross_exposure(self, current_prices: dict[str, float]) -> float:
        exposure = 0.0
        for symbol, position in self.positions.items():
            price = current_prices.get(symbol, position.avg_entry_price)
            exposure += abs(position.quantity) * price
        return exposure

    # ------------------------------------------------------------------
    # Signal sizing
    # ------------------------------------------------------------------

    def size_signal(
        self,
        signal: dict[str, Any],
        execution_price: float,
        portfolio_equity: float,
        default_allocation: float,
    ) -> int:
        current_position = self.positions.get(signal["symbol"])
        action = signal["action"].upper()

        # --- Closing an existing position ---
        if current_position is not None:
            if current_position.quantity > 0 and action == "SELL":
                return abs(current_position.quantity)   # close long
            if current_position.quantity < 0 and action == "BUY":
                return abs(current_position.quantity)   # close short
            # Same direction as existing position -- no pyramiding
            if (current_position.quantity > 0 and action == "BUY") or (
                current_position.quantity < 0 and action == "SELL"
            ):
                return 0

        # --- Guard: exit signals must never open a new position ---
        # If is_exit=True but no position exists, the leg was already
        # closed (or the entry was rejected). Skip cleanly.
        if signal.get("is_exit"):
            return 0

        # --- New position sizing ---
        allocation = float(signal.get("weight_allocation", default_allocation))
        if allocation <= 0.0:
            return 0

        # Never size a new position when equity is zero or negative
        if portfolio_equity <= 0.0:
            return 0

        capped_allocation = min(allocation, self.max_total_exposure)
        position_value    = portfolio_equity * capped_allocation
        quantity          = int(position_value / execution_price)
        return max(quantity, 0)

    # ------------------------------------------------------------------
    # Signal execution -- router
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

        current_position = self.positions.get(symbol)

        if action == "BUY" and current_position is not None and current_position.quantity < 0:
            return self._close_short(signal, execution_price, date, quantity)
        if action == "SELL" and current_position is not None and current_position.quantity > 0:
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

    def _open_long(self, signal: dict[str, Any], execution_price: float, date: str, quantity: int) -> bool:
        symbol     = signal["symbol"]
        fill_price = self._effective_price("BUY", execution_price)
        commission = self._commission_paid(quantity, fill_price)
        cash_needed = quantity * fill_price + commission

        if self.cash < cash_needed:
            self._log_rejection(signal, date, "insufficient_cash")
            return False

        self.cash -= cash_needed
        current    = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_entry_price=fill_price,
                entry_date=date,
                entry_commission=commission,
            )
        else:
            total_qty  = current.quantity + quantity
            total_cost = (current.quantity * current.avg_entry_price) + (quantity * fill_price)
            current.avg_entry_price  = total_cost / total_qty
            current.quantity         = total_qty
            current.entry_commission += commission

        self._log_trade(signal=signal, date=date, quantity=quantity, side="BUY",
                        price=fill_price, commission_paid=commission,
                        realized_pnl=None, hold_days=None)
        return True

    def _open_short(self, signal: dict[str, Any], execution_price: float, date: str, quantity: int) -> bool:
        symbol     = signal["symbol"]
        fill_price = self._effective_price("SELL", execution_price)
        commission = self._commission_paid(quantity, fill_price)
        proceeds   = quantity * fill_price - commission

        self.cash += proceeds
        current    = self.positions.get(symbol)
        if current is None:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=-quantity,
                avg_entry_price=fill_price,
                entry_date=date,
                entry_commission=commission,
            )
        else:
            total_qty  = abs(current.quantity) + quantity
            total_cost = (abs(current.quantity) * current.avg_entry_price) + (quantity * fill_price)
            current.avg_entry_price  = total_cost / total_qty
            current.quantity         = -total_qty
            current.entry_commission += commission

        self._log_trade(signal=signal, date=date, quantity=quantity, side="SELL",
                        price=fill_price, commission_paid=commission,
                        realized_pnl=None, hold_days=None)
        return True

    def _close_long(self, signal: dict[str, Any], execution_price: float, date: str, quantity: int) -> bool:
        symbol  = signal["symbol"]
        current = self.positions.get(symbol)
        if current is None or current.quantity <= 0:
            self._log_rejection(signal, date, "no_long_position")
            return False

        shares_to_close          = min(quantity, current.quantity)
        fill_price               = self._effective_price("SELL", execution_price)
        commission               = self._commission_paid(shares_to_close, fill_price)
        prop_entry_comm          = current.entry_commission * (shares_to_close / current.quantity)
        gross_pnl                = (fill_price - current.avg_entry_price) * shares_to_close
        realized_pnl             = gross_pnl - commission - prop_entry_comm

        self.cash               += shares_to_close * fill_price - commission
        current.quantity        -= shares_to_close
        current.entry_commission -= prop_entry_comm

        hold_days = self._holding_days(current.entry_date, date)
        if current.quantity == 0:
            del self.positions[symbol]

        self._log_trade(signal=signal, date=date, quantity=shares_to_close, side="SELL",
                        price=fill_price, commission_paid=commission,
                        realized_pnl=realized_pnl, hold_days=hold_days)
        return True

    def _close_short(self, signal: dict[str, Any], execution_price: float, date: str, quantity: int) -> bool:
        symbol  = signal["symbol"]
        current = self.positions.get(symbol)
        if current is None or current.quantity >= 0:
            self._log_rejection(signal, date, "no_short_position")
            return False

        open_qty                 = abs(current.quantity)
        shares_to_close          = min(quantity, open_qty)
        fill_price               = self._effective_price("BUY", execution_price)
        commission               = self._commission_paid(shares_to_close, fill_price)
        prop_entry_comm          = current.entry_commission * (shares_to_close / open_qty)
        gross_pnl                = (current.avg_entry_price - fill_price) * shares_to_close
        realized_pnl             = gross_pnl - commission - prop_entry_comm

        self.cash               -= shares_to_close * fill_price + commission
        current.quantity        += shares_to_close
        current.entry_commission -= prop_entry_comm

        hold_days = self._holding_days(current.entry_date, date)
        if current.quantity == 0:
            del self.positions[symbol]

        self._log_trade(signal=signal, date=date, quantity=shares_to_close, side="BUY",
                        price=fill_price, commission_paid=commission,
                        realized_pnl=realized_pnl, hold_days=hold_days)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _holding_days(self, entry_date: str, exit_date: str) -> int:
        from datetime import datetime
        entry = datetime.fromisoformat(str(entry_date))
        exit_ = datetime.fromisoformat(str(exit_date))
        return max((exit_ - entry).days, 0)

    def _log_trade(
        self,
        *,
        signal: dict[str, Any],
        date: str,
        quantity: int,
        side: str,
        price: float,
        commission_paid: float,
        realized_pnl: float | None,
        hold_days: int | None,
    ) -> None:
        self.trade_log.append({
            "date":         date,
            "symbol":       signal["symbol"],
            "strategy":     signal.get("strategy"),
            "action":       side,
            "quantity":     quantity,
            "price":        price,
            "commission":   commission_paid,
            "slippage":     self.slippage,
            "realized_pnl": realized_pnl,
            "hold_days":    hold_days,
            "reason":       signal.get("reason"),
            "metadata":     signal.get("metadata"),
            "signal_id":    signal.get("signal_id"),
            "status":       "filled",
        })

    def _log_rejection(self, signal: dict[str, Any], date: str, reason: str) -> None:
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
        for symbol, position in self.positions.items():
            current_price = current_prices.get(symbol, position.avg_entry_price)
            equity += position.quantity * current_price
        return equity

    def get_positions(self) -> dict[str, Position]:
        """
        Returns the live Position objects (not dicts).
        StatArbStrategy accesses .quantity directly via hasattr check.
        """
        return dict(self.positions)

    def get_trade_log(self) -> list[dict[str, Any]]:
        return list(self.trade_log)

    def get_rejection_log(self) -> list[dict[str, Any]]:
        return list(self.rejection_log)


def datetime_from_date_str(value: str):
    from datetime import datetime
    return datetime.fromisoformat(str(value))
