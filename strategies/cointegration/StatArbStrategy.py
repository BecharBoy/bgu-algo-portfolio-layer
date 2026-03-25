from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from Strategy import BaseStrategy

# Import the compiled C++ extension.
# Run: cd strategies/cointegration && python setup.py build_ext --inplace
try:
    import cointegration_engine
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


class StatArbStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "CointegrationArb",
        capital_allocation: float = 0.3,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        lookback_window: int = 60,
        num_threads: int = 6,
        min_correlation: float = 0.85,
    ) -> None:
        super().__init__(name, capital_allocation)
        self.zscore_entry    = zscore_entry
        self.zscore_exit     = zscore_exit
        self.lookback_window = lookback_window
        self.num_threads     = num_threads
        self.min_correlation = min_correlation

    async def generate_signals(
        self,
        market_data: dict[str, pd.DataFrame],
        current_positions: dict[str, Any],   # dict[str, Position] from portfolio
    ) -> list[dict]:
        if not _CPP_AVAILABLE:
            raise RuntimeError(
                "cointegration_engine C++ extension not found. "
                "Run: cd strategies/cointegration && python setup.py build_ext --inplace"
            )
        if len(market_data) < 2:
            return []

        # --- Build price matrix for C++ scanner ---
        tickers = list(market_data.keys())
        min_len = min(len(df) for df in market_data.values())
        if min_len < self.lookback_window:
            return []

        price_matrix = [
            market_data[t]["Close"].values[-self.lookback_window:].tolist()
            for t in tickers
        ]

        # --- Call C++ engine (releases GIL internally) ---
        cointegrated_pairs = cointegration_engine.run_cpp_scan(
            tickers=tickers,
            price_matrix=price_matrix,
            num_threads=self.num_threads,
            min_correlation=self.min_correlation,
        )

        if not cointegrated_pairs:
            return []

        # --- Build pos_map: symbol -> net quantity (int) ---
        # get_positions() now returns dict[str, Position] with .quantity attribute
        pos_map: dict[str, int] = {}
        for sym, pos in current_positions.items():
            qty = pos.quantity if hasattr(pos, "quantity") else pos.get("net_quantity", 0)
            pos_map[sym] = qty

        signals: list[dict] = []

        for pair in cointegrated_pairs:
            stock_x = pair.stock_x
            stock_y = pair.stock_y
            beta    = pair.hedge_ratio

            if stock_x not in market_data or stock_y not in market_data:
                continue

            prices_x = market_data[stock_x]["Close"].values[-self.lookback_window:]
            prices_y = market_data[stock_y]["Close"].values[-self.lookback_window:]

            spread      = prices_y - beta * prices_x
            spread_mean = np.mean(spread)
            spread_std  = np.std(spread, ddof=1)
            if spread_std == 0.0:
                continue

            zscore = (spread[-1] - spread_mean) / spread_std

            # A pair is considered open when EITHER leg has a non-zero position
            qty_x = pos_map.get(stock_x, 0)
            qty_y = pos_map.get(stock_y, 0)
            pair_is_open = (qty_x != 0) or (qty_y != 0)

            if pair_is_open:
                # Exit when spread has reverted toward the mean
                if abs(zscore) < self.zscore_exit:
                    pair_id = f"{stock_x}_{stock_y}_exit"
                    for symbol, qty in ((stock_x, qty_x), (stock_y, qty_y)):
                        if qty == 0:
                            # Leg was never opened (entry rejected) -- skip
                            continue
                        close_action = "SELL" if qty > 0 else "BUY"
                        signals.append(self._build_leg(
                            symbol=symbol,
                            action=close_action,
                            price=float(market_data[symbol]["Close"].iloc[-1]),
                            reason="pair_exit_reversion",
                            metadata={
                                "zscore":       zscore,
                                "hedge_ratio":  beta,
                                "pair_id":      pair_id,
                                "pair_partner": stock_y if symbol == stock_x else stock_x,
                            },
                            is_exit=True,   # <-- tells portfolio NOT to open a new position
                        ))
                # Spread still wide -- hold
                continue

            # --- New entry ---
            pair_id = str(uuid.uuid4())

            if zscore > self.zscore_entry:
                # Y overpriced vs X  -->  SELL Y (short), BUY X (long)
                signals += self._make_entry(
                    stock_y, "SELL", stock_x, "BUY",
                    market_data, beta, zscore, pair_id, "stat_arb_spread_high",
                )
            elif zscore < -self.zscore_entry:
                # Y underpriced vs X  -->  BUY Y (long), SELL X (short)
                signals += self._make_entry(
                    stock_y, "BUY", stock_x, "SELL",
                    market_data, beta, zscore, pair_id, "stat_arb_spread_low",
                )

        return signals

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_entry(
        self,
        sym_y: str, action_y: str,
        sym_x: str, action_x: str,
        market_data: dict[str, pd.DataFrame],
        beta: float, zscore: float, pair_id: str, reason: str,
    ) -> list[dict]:
        return [
            self._build_leg(
                symbol=sym_y,
                action=action_y,
                price=float(market_data[sym_y]["Close"].iloc[-1]),
                reason=reason,
                metadata={
                    "zscore":       zscore,
                    "hedge_ratio":  beta,
                    "pair_leg":     "short" if action_y == "SELL" else "long",
                    "pair_id":      pair_id,
                    "pair_partner": sym_x,
                },
                is_exit=False,
            ),
            self._build_leg(
                symbol=sym_x,
                action=action_x,
                price=float(market_data[sym_x]["Close"].iloc[-1]),
                reason=reason,
                metadata={
                    "zscore":       zscore,
                    "hedge_ratio":  beta,
                    "pair_leg":     "long" if action_x == "BUY" else "short",
                    "pair_id":      pair_id,
                    "pair_partner": sym_y,
                },
                is_exit=False,
            ),
        ]

    def _build_leg(
        self,
        symbol: str,
        action: str,
        price: float,
        reason: str,
        metadata: dict,
        is_exit: bool = False,
    ) -> dict:
        return {
            "signal_id":         str(uuid.uuid4()),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "symbol":            symbol,
            "action":            action,
            "strategy":          self.name,
            "weight_allocation": self.capital_allocation,
            "price_reference":   price,
            "reason":            reason,
            "metadata":          metadata,
            "is_exit":           is_exit,   # consumed by portfolio.size_signal()
        }
