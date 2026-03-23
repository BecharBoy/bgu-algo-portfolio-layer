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
        weight_allocation: float = 0.5,
        zscore_entry: float = 2.0,
        zscore_exit: float = 0.5,
        lookback_window: int = 60,
    ):
        super().__init__(name, weight_allocation)
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
                    signals.append(self._build_pair_signal(
                        stock_x, stock_y, "SELL", "SELL",
                        market_data, beta, zscore, "pair_exit_reversion"
                    ))
                continue

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
            "signal_id":         str(uuid.uuid4()),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "symbol":            symbol,
            "action":            action,
            "strategy":          self.name,
            "price_reference":   price,
            "reason":            reason,
            "weight_allocation": self.weight_allocation,
            "metadata":          metadata,
        }

    def _build_pair_signal(
        self,
        stock_x: str,
        stock_y: str,
        action_x: str,
        action_y: str,
        market_data: Dict,
        beta: float,
        zscore: float,
        reason: str,
    ) -> Dict:
        return self._build_signal_leg(
            symbol=stock_y,
            action=action_y,
            price=market_data[stock_y]["Close"].iloc[-1],
            reason=reason,
            metadata={"zscore": zscore, "hedge_ratio": beta, "pair_partner": stock_x},
        )

    def _is_pair_already_open(
        self,
        pair: Dict[str, Any],
        current_positions: Dict[str, Any],
    ) -> bool:
        return pair["stock_x"] in current_positions or pair["stock_y"] in current_positions
