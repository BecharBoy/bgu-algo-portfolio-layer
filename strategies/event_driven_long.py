from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from main_backtesting.models import PriceBar, ThresholdPass, Trade


class ThresholdPassTracker:
    def __init__(self, market_id: str, threshold: float = 0.55) -> None:
        self.market_id = market_id
        self.threshold = threshold
        self.is_above = False
        self.passes: list[ThresholdPass] = []

    def observe(self, timestamp: datetime, probability: float) -> ThresholdPass | None:
        if probability > self.threshold and not self.is_above:
            self.is_above = True
            threshold_pass = ThresholdPass(
                market_id=self.market_id,
                pass_number=len(self.passes) + 1,
                above_at=timestamp,
                above_probability=probability,
            )
            self.passes.append(threshold_pass)
            return threshold_pass

        if probability <= self.threshold and self.is_above:
            self.is_above = False
            current_pass = self.passes[-1]
            current_pass.fell_below_at = timestamp
            current_pass.fell_below_probability = probability
        return None


def ib_style_commission(quantity: float, order_value: float) -> float:
    return min(max(quantity * 0.005, 1.0), order_value * 0.01)


def true_range(current: PriceBar, previous_close: float) -> float:
    return max(
        current.high - current.low,
        abs(current.high - previous_close),
        abs(current.low - previous_close),
    )


def average_true_range(previous_bars: list[PriceBar], period: int = 14) -> float | None:
    if len(previous_bars) < period + 1:
        return None
    selected = previous_bars[-(period + 1) :]
    ranges = [
        true_range(selected[index], selected[index - 1].close)
        for index in range(1, len(selected))
    ]
    return sum(ranges) / len(ranges)


class EventDrivenLongStrategy:
    def __init__(
        self,
        *,
        trade_notional: float = 1_000.0,
        range_period: int = 14,
        range_multiplier: float = 3.0,
    ) -> None:
        self.trade_notional = trade_notional
        self.range_period = range_period
        self.range_multiplier = range_multiplier

    def open_trade(
        self,
        *,
        run_id: UUID,
        market_id: str,
        event_id: str,
        question: str,
        symbol: str,
        asset_name: str,
        pass_number: int,
        trigger_at: datetime,
        entry_bar: PriceBar,
        previous_bars: list[PriceBar],
        final_outcome: str | None,
    ) -> Trade | None:
        atr = average_true_range(previous_bars, self.range_period)
        if atr is None or atr <= 0 or entry_bar.open <= 0:
            return None

        quantity = self.trade_notional / entry_bar.open
        commission = ib_style_commission(quantity, self.trade_notional)
        stop = entry_bar.open - self.range_multiplier * atr
        return Trade(
            trade_id=uuid4(),
            run_id=run_id,
            market_id=market_id,
            event_id=event_id,
            question=question,
            symbol=symbol,
            asset_name=asset_name,
            pass_number=pass_number,
            trigger_at=trigger_at,
            entry_at=entry_bar.timestamp,
            entry_price=entry_bar.open,
            quantity=quantity,
            entry_commission=commission,
            initial_stop=stop,
            current_stop=stop,
            highest_price=entry_bar.open,
            final_outcome=final_outcome,
            maximum_price=entry_bar.open,
            minimum_price=entry_bar.open,
            stop_history=[{"timestamp": entry_bar.timestamp, "stop": stop}],
        )

    def update_trade(self, trade: Trade, bar: PriceBar, previous_bars: list[PriceBar]) -> bool:
        trade.maximum_price = max(trade.maximum_price or bar.high, bar.high)
        trade.minimum_price = min(trade.minimum_price or bar.low, bar.low)

        # The user selected exact-stop fills, even when the hourly bar gaps below.
        if bar.low <= trade.current_stop:
            trade.exit_at = bar.timestamp
            trade.exit_price = trade.current_stop
            trade.exit_commission = ib_style_commission(
                trade.quantity,
                trade.quantity * trade.exit_price,
            )
            trade.exit_reason = "trailing_stop"
            return True

        atr = average_true_range(previous_bars, self.range_period)
        if atr is None or atr <= 0:
            return False
        trade.highest_price = max(trade.highest_price, bar.high)
        raised_stop = trade.highest_price - self.range_multiplier * atr
        trade.current_stop = max(trade.current_stop, raised_stop)
        trade.stop_history.append({"timestamp": bar.timestamp, "stop": trade.current_stop})
        return False

