import pandas as pd
import talib as ta
from typing import List, Dict, Any
from Strategy import BaseStrategy


class MeanReversionMomentum(BaseStrategy):
    def __init__(self, name: str = "MeanReversionMomentum", weight_allocation: float = 0.5):
        super().__init__(name, weight_allocation)
        # שומרים כאן פרמטרים של האלגוריתם בלבד, לא שומרים DataFrames או מילונים של אינדיקטורים
        self.sma_window = 30
        self.rsi_period = 14
        self.atr_period = 14

    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        פעולה וקטורית על ה-DataFrame שסופק מבחוץ.
        מניח שה-DataFrame מכיל עמודות: Open, High, Low, Close, Volume
        """
        # עותק כדי לא לזהם את הנתונים המקוריים שמועברים אולי לאסטרטגיות אחרות
        data = df.copy()

        data['SMA'] = data['Close'].rolling(window=self.sma_window).mean()
        data['Upper_BB'] = data['SMA'] + 2 * data['Close'].rolling(self.sma_window).std()
        data['Lower_BB'] = data['SMA'] - 2 * data['Close'].rolling(self.sma_window).std()

        data['RSI'] = ta.RSI(data['Close'].values, timeperiod=self.rsi_period)
        data['ATR'] = ta.ATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=self.atr_period)

        macd, macdsignal, macdhist = ta.MACD(data['Close'].values, fastperiod=24, slowperiod=52, signalperiod=18)
        data['MACD_Line'] = macd
        data['MACD_Signal'] = macdsignal

        return data

    async def generate_signals(self, market_data: Dict[str, pd.DataFrame], current_positions: Dict[str, Any]) -> List[
        Dict]:
        """
        הפונקציה המרכזית שנקראת על ידי ה-Portfolio.
        market_data: מילון של טיקרים וה-DataFrame ההיסטורי שלהם.
        """
        signals = []

        for ticker, df in market_data.items():
            if df.empty or len(df) < 52:  # מוודא שיש מספיק נתונים לחישוב MACD איטי
                continue

            df_with_ind = self._calculate_indicators(df)
            current_price = df_with_ind['Close'].iloc[-1]

            # בדיקה האם יש פוזיציה פתוחה בנכס הזה
            if ticker in current_positions:
                pos_data = current_positions[ticker]
                if self._get_sell_signal(df_with_ind, pos_data, current_price):
                    signals.append({
                        'symbol': ticker,
                        'action': 'SELL',
                        'strategy': self.name,
                        'price_reference': current_price
                    })
            else:
                if self._get_buy_signal(df_with_ind, current_price):
                    signals.append({
                        'symbol': ticker,
                        'action': 'BUY',
                        'strategy': self.name,
                        'price_reference': current_price
                    })

        return signals

    def _get_buy_signal(self, df: pd.DataFrame, current_price: float) -> bool:
        """לוגיקת כניסה לעסקה לפי שורת הנתונים האחרונה"""
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2]

        # ... כאן תכנס לוגיקת ה-Bullish/Bearish, MACD, Bollinger וה-ATR שהייתה בקוד הישן שלך ...
        # דוגמה חלקית:
        if last_row['MACD_Line'] >= last_row['MACD_Signal'] and prev_row['MACD_Line'] <= prev_row['MACD_Signal']:
            return True

        return False

    def _get_sell_signal(self, df: pd.DataFrame, pos_data: dict, current_price: float) -> bool:
        """לוגיקת יציאה מעסקה, כולל בדיקת Stop Loss / Take Profit ומומנטום דועך"""
        # ... לוגיקת המכירה ...
        pass