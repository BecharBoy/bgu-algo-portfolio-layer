import pandas as pd
import talib as ta
from typing import List, Dict, Any
from Strategy import BaseStrategy
import uuid
from datetime import datetime, timezone


class MeanReversionMomentum(BaseStrategy):
    def __init__(self, name: str = "MeanReversionMomentum", capital_allocation: float = 0.15):
        super().__init__(name, capital_allocation)
        self.sma_window = 30
        self.rsi_period = 14
        self.atr_period = 14

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data["SMA"]      = data["Close"].rolling(window=self.sma_window).mean()
        data["Upper_BB"] = data["SMA"] + 2 * data["Close"].rolling(self.sma_window).std()
        data["Lower_BB"] = data["SMA"] - 2 * data["Close"].rolling(self.sma_window).std()
        data["RSI"]      = ta.RSI(data["Close"].values, timeperiod=self.rsi_period)
        data["ATR"]      = ta.ATR(data["High"].values, data["Low"].values,
                                   data["Close"].values, timeperiod=self.atr_period)
        macd, macd_signal, _ = ta.MACD(data["Close"].values, fastperiod=24,
                                        slowperiod=52, signalperiod=18)
        data["MACD_Line"]   = macd
        data["MACD_Signal"] = macd_signal
        return data

    def _validate_indicator_row(self, row: pd.Series) -> bool:
        required = ["SMA", "Upper_BB", "Lower_BB", "RSI", "ATR", "MACD_Line", "MACD_Signal"]
        return all(col in row.index and not pd.isna(row[col]) for col in required)

    async def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        current_positions: Dict[str, Any],   # dict[str, Position] — ONLY this strategy's positions
    ) -> List[Dict]:
        """
        current_positions contains only positions owned by MeanReversionMomentum.
        The engine passes portfolio.get_positions() filtered per strategy, so MR
        will never see CointegrationArb positions and cannot accidentally close them.
        """
        signals: List[Dict[str, Any]] = []

        for ticker, df in market_data.items():
            if df.empty or len(df) < 52:
                continue
            try:
                df_ind = self._calculate_indicators(df)
                current_price = df_ind["Close"].iloc[-1]

                if ticker in current_positions:
                    # Only exit if this strategy owns the position
                    pos = current_positions[ticker]
                    pos_strategy = pos.strategy if hasattr(pos, "strategy") else pos.get("strategy", self.name)
                    if pos_strategy != self.name:
                        continue   # not ours — never touch it
                    if self._get_sell_signal(df_ind, pos, current_price):
                        signals.append(self._build_signal(
                            symbol=ticker, action="SELL",
                            current_price=current_price,
                            reason="exit_rule_triggered",
                            is_exit=True,
                        ))
                else:
                    if self._get_buy_signal(df_ind, current_price):
                        signals.append(self._build_signal(
                            symbol=ticker, action="BUY",
                            current_price=current_price,
                            reason="entry_rule_triggered",
                            is_exit=False,
                        ))
            except Exception as e:
                import logging
                logging.warning(f"MeanReversionMomentum: skipping {ticker} — {e}")
                continue

        return signals

    def _get_buy_signal(self, df: pd.DataFrame, current_price: float) -> bool:
        if len(df) < 2:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if not self._validate_indicator_row(last) or not self._validate_indicator_row(prev):
            return False
        if current_price >= last["Lower_BB"]:
            return False
        rsi_oversold  = last["RSI"] < 35
        macd_cross_up = (last["MACD_Line"] >= last["MACD_Signal"] and
                         prev["MACD_Line"] <  prev["MACD_Signal"])
        return rsi_oversold or macd_cross_up

    def _get_sell_signal(self, df: pd.DataFrame, pos_data: Any, current_price: float) -> bool:
        if len(df) < 2:
            return False
        last = df.iloc[-1]
        prev = df.iloc[-2]
        if not self._validate_indicator_row(last):
            return False
        if last["RSI"] > 70:
            return True
        if current_price > last["Upper_BB"]:
            return True
        if (last["MACD_Line"] <= last["MACD_Signal"] and
                prev["MACD_Line"] >= prev["MACD_Signal"]):
            return True
        return False

    def _build_signal(self, symbol, action, current_price, reason, is_exit=False) -> Dict:
        return {
            "signal_id":         str(uuid.uuid4()),
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "symbol":            symbol,
            "action":            action,
            "strategy":          self.name,
            "weight_allocation": self.capital_allocation,
            "price_reference":   current_price,
            "reason":            reason,
            "is_exit":           is_exit,
        }
