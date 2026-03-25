# strategies/cointegration/StatArbStrategy.py

import pandas as pd
import numpy as np
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any
from Strategy import BaseStrategy
from Wrapper import Wrapper


class StatArbStrategy(BaseStrategy):
    def __init__(
        self,
        name: str = "CointegrationArb",
        capital_allocation: float = 0.5,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        lookback_window: int = 60,
    ):
        super().__init__(name, capital_allocation)
        self.wrapper = Wrapper(num_threads=6, min_correlation=0.85)
        self.zscore_entry = zscore_entry
        self.zscore_exit = zscore_exit
        self.lookback_window = lookback_window

    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, Any],
    ) -> List[Dict]:

        if len(market_data) < 2:
            return []

        cointegrated_pairs = await self.wrapper.run_cointegration_scan(market_data)

        if not cointegrated_pairs:
            return []

        signals: List[Dict] = []

        for pair in cointegrated_pairs:
            stock_x = pair["stock_x"]
            stock_y = pair["stock_y"]
            beta    = pair["hedge_ratio"]
            alpha   = pair.get("alpha", 0.0)

            if stock_x not in market_data or stock_y not in market_data:
                continue

            prices_x = market_data[stock_x]["Close"].values
            prices_y = market_data[stock_y]["Close"].values

            min_len = min(len(prices_x), len(prices_y))
            if min_len < self.lookback_window:
                continue

            prices_x = prices_x[-self.lookback_window:]
            prices_y = prices_y[-self.lookback_window:]

            spread = prices_y - beta * prices_x - alpha

            spread_mean = np.mean(spread)
            spread_std  = np.std(spread, ddof=1)

            if spread_std == 0.0:
                continue

            zscore = (spread[-1] - spread_mean) / spread_std

            pair_is_open = self._is_pair_already_open(pair, current_positions)

            if pair_is_open:
                if abs(zscore) < self.zscore_exit:
                    # Exit BOTH legs — close whatever side each symbol is on.
                    # We determine close direction by checking current_positions.
                    # If a symbol is long (net_quantity > 0) we SELL to close; if short we BUY.
                    pair_id = f"{stock_x}_{stock_y}_exit"
                    for symbol in (stock_x, stock_y):
                        pos = current_positions.get(symbol)
                        if pos is None:
                            continue
                        net_qty = pos.get("net_quantity", 0)
                        close_action = "SELL" if net_qty > 0 else "BUY"
                        signals.append(self._build_signal_leg(
                            symbol=symbol,
                            action=close_action,
                            price=market_data[symbol]["Close"].iloc[-1],
                            reason="pair_exit_reversion",
                            metadata={
                                "zscore": zscore,
                                "hedge_ratio": beta,
                                "pair_id": pair_id,
                                "pair_partner": stock_y if symbol == stock_x else stock_x,
                            },
                        ))
                continue

            # Shared pair_id tags both legs so aggregation can protect them atomically.
            pair_id = str(uuid.uuid4())

            if zscore > self.zscore_entry:
                # Spread too high: Y overpriced vs X → SELL Y, BUY X
                signals.extend([
                    self._build_signal_leg(
                        symbol=stock_y,
                        action="SELL",
                        price=market_data[stock_y]["Close"].iloc[-1],
                        reason="stat_arb_spread_high",
                        metadata={
                            "zscore": zscore,
                            "hedge_ratio": beta,
                            "pair_leg": "short",
                            "pair_id": pair_id,
                            "pair_partner": stock_x,
                        }
                    ),
                    self._build_signal_leg(
                        symbol=stock_x,
                        action="BUY",
                        price=market_data[stock_x]["Close"].iloc[-1],
                        reason="stat_arb_spread_high",
                        metadata={
                            "zscore": zscore,
                            "hedge_ratio": beta,
                            "pair_leg": "long",
                            "pair_id": pair_id,
                            "pair_partner": stock_y,
                        }
                    ),
                ])

            elif zscore < -self.zscore_entry:
                # Spread too low: Y underpriced vs X → BUY Y, SELL X
                signals.extend([
                    self._build_signal_leg(
                        symbol=stock_y,
                        action="BUY",
                        price=market_data[stock_y]["Close"].iloc[-1],
                        reason="stat_arb_spread_low",
                        metadata={
                            "zscore": zscore,
                            "hedge_ratio": beta,
                            "pair_leg": "long",
                            "pair_id": pair_id,
                            "pair_partner": stock_x,
                        }
                    ),
                    self._build_signal_leg(
                        symbol=stock_x,
                        action="SELL",
                        price=market_data[stock_x]["Close"].iloc[-1],
                        reason="stat_arb_spread_low",
                        metadata={
                            "zscore": zscore,
                            "hedge_ratio": beta,
                            "pair_leg": "short",
                            "pair_id": pair_id,
                            "pair_partner": stock_y,
                        }
                    ),
                ])

        return signals

    def _build_signal_leg(
        self,
        symbol: str,
        action: str,
        price: float,
        reason: str,
        metadata: Dict[str, Any],
    ) -> Dict:
        return {
            "signal_id":       str(uuid.uuid4()),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "symbol":          symbol,
            "action":          action,
            "strategy":        self.name,
            "price_reference": price,
            "reason":          reason,
            "metadata":        metadata,
        }

    def _is_pair_already_open(
        self,
        pair: Dict[str, Any],
        current_positions: Dict[str, Any],
    ) -> bool:
        return pair["stock_x"] in current_positions or pair["stock_y"] in current_positions
