import pandas as pd
import talib as ta
from typing import List, Dict, Any
from Strategy import BaseStrategy
import uuid
from datetime import datetime, timezone


class MeanReversionMomentum(BaseStrategy):
    def __init__(self, name: str = "MeanReversionMomentum", capital_allocation: float = 0.5):
        super().__init__(name, capital_allocation)
        self.sma_window = 30
        self.rsi_period = 14
        self.atr_period = 14

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
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

    def _validate_indicator_row(self, row: pd.Series) -> bool:
        required = ["SMA", "Upper_BB", "Lower_BB", "RSI", "ATR", "MACD_Line", "MACD_Signal"]
        for col in required:
            if col not in row.index:
                return False
            if pd.isna(row[col]):
                return False
        return True

    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, Any],
    ) -> List[Dict]:
        signals: List[Dict[str, Any]] = []

        for ticker, df in market_data.items():
            if df.empty or len(df) < 52:
                continue

            try:
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
            except Exception as e:
                # One bad ticker must never kill the whole batch
                import logging
                logging.warning(f"MeanReversionMomentum: skipping {ticker} — {e}")
                continue

        return signals

    def _get_buy_signal(self, df: pd.DataFrame, current_price: float) -> bool:
        """
        Entry logic — two-condition gate (not three).

        Condition 1 (REQUIRED): Price is below the lower Bollinger Band.
          This is the primary mean-reversion trigger. The stock is
          statistically cheap relative to its 30-day history.

        Condition 2 (CONFIRMATION — pick ONE of two):
          A) RSI < 35 — momentum confirms oversold state, OR
          B) MACD bullish crossover on the last bar — momentum turning up.

        Requiring all three simultaneously is too restrictive:
        a MACD crossover rarely lands on the exact same bar as a
        BB breach. The two-condition gate fires ~3-5x more often
        while keeping signal quality high.
        """
        if len(df) < 2:
            return False

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if not self._validate_indicator_row(last):
            return False
        if not self._validate_indicator_row(prev):
            return False

        # Primary gate: price must be below lower Bollinger Band
        price_oversold = current_price < last["Lower_BB"]
        if not price_oversold:
            return False

        # Confirmation gate: RSI oversold OR MACD just crossed up
        rsi_oversold = last["RSI"] < 35

        macd_cross_up = (
            last["MACD_Line"] >= last["MACD_Signal"] and
            prev["MACD_Line"] <  prev["MACD_Signal"]
        )

        return rsi_oversold or macd_cross_up

    def _get_sell_signal(self, df: pd.DataFrame, pos_data: Dict, current_price: float) -> bool:
        if len(df) < 2:
            return False

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if not self._validate_indicator_row(last):
            return False

        # RSI overbought
        if last["RSI"] > 70:
            return True

        # Price broke above upper Bollinger Band
        if current_price > last["Upper_BB"]:
            return True

        # MACD bearish crossover
        if (last["MACD_Line"] <= last["MACD_Signal"] and
                prev["MACD_Line"] >= prev["MACD_Signal"]):
            return True

        return False

    def _build_signal(self, symbol, action, current_price, reason) -> Dict:
        return {
            "signal_id":         str(uuid.uuid4()),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "symbol":            symbol,
            "action":            action,
            "strategy":          self.name,
            "price_reference":   current_price,
            "reason":            reason,
        }
