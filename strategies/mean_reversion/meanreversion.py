import pandas as pd
import talib as ta
from typing import List, Dict, Any
from Strategy import BaseStrategy
import uuid
from datetime import datetime, timezone

class MeanReversionMomentum(BaseStrategy):
    def __init__(self, name: str = "MeanReversionMomentum", weight_allocation: float = 0.5):
        super().__init__(name, weight_allocation)
        self.sma_window = 30
        self.rsi_period = 14
        self.atr_period = 14
        # TODO: Add explicit entry and exit threshold parameters.
        # TODO: Add anti-lookahead policy (generate on close, execute next session).

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO: Validate required columns: Open, High, Low, Close, Volume.
        # TODO: Consider caching indicator windows for larger universes.
        data = df.copy()

        data["SMA"] = data["Close"].rolling(window=self.sma_window).mean()
        data["Upper_BB"] = data["SMA"] + 2 * data["Close"].rolling(self.sma_window).std()
        data["Lower_BB"] = data["SMA"] - 2 * data["Close"].rolling(self.sma_window).std()

        data["RSI"] = ta.RSI(data["Close"].values, timeperiod=self.rsi_period)
        data["ATR"] = ta.ATR(
            data["High"].values,
            data["Low"].values,
            data["Close"].values,
            timeperiod=self.atr_period,
        )

        macd, macd_signal, _ = ta.MACD(
            data["Close"].values,
            fastperiod=24,
            slowperiod=52,
            signalperiod=18,
        )
        data["MACD_Line"] = macd
        data["MACD_Signal"] = macd_signal

        return data

    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, Any],
    ) -> List[Dict]:
        # TODO: Add per-ticker exception isolation to keep the batch resilient.
        # TODO: Return one shared signal schema with timestamp and signal_id.
        signals: List[Dict[str, Any]] = []

        for ticker, df in market_data.items():
            if df.empty or len(df) < 52:
                continue

            df_with_ind = self._calculate_indicators(df)
            current_price = df_with_ind["Close"].iloc[-1]

            if ticker in current_positions:
                position_data = current_positions[ticker]
                if self._get_sell_signal(df_with_ind, position_data, current_price):
                    signals.append(
                        self._build_signal(
                            symbol=ticker,
                            action="SELL",
                            current_price=current_price,
                            reason="exit_rule_triggered",
                        )
                    )
            else:
                if self._get_buy_signal(df_with_ind, current_price):
                    signals.append(
                        self._build_signal(
                            symbol=ticker,
                            action="BUY",
                            current_price=current_price,
                            reason="entry_rule_triggered",
                        )
                    )

        return signals

    def _get_buy_signal(self, df: pd.DataFrame, current_price: float) -> bool:
        # TODO: Replace placeholder crossover logic with full mean reversion criteria.
        # TODO: Use RSI, ATR, Bollinger, and trend filter consistently.
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        if last_row["MACD_Line"] >= last_row["MACD_Signal"] and prev_row["MACD_Line"] <= prev_row["MACD_Signal"]:
            return True

        return False

    def _get_sell_signal(self, df: pd.DataFrame, pos_data: Dict, current_price: float) -> bool:
        last = df.iloc[-1]
        if last["RSI"] > 70:
            return True
        if current_price > last["Upper_BB"]:
            return True
        prev = df.iloc[-2]
        if last["MACD_Line"] <= last["MACD_Signal"] and prev["MACD_Line"] >= prev["MACD_Signal"]:
            return True
        return False

    def _build_signal(self, symbol, action, current_price, reason) -> Dict:
        return {
            "signal_id":       str(uuid.uuid4()),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "symbol":          symbol,
            "action":          action,
            "strategy":        self.name,
            "price_reference": current_price,
            "reason":          reason,
            "weight_allocation": self.weight_allocation,
        }

    def _validate_indicator_row(self, row: pd.Series) -> bool:
        # TODO: Validate NaNs and malformed final rows before decision logic.
        pass
