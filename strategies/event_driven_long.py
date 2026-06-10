from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from main_backtesting.models import Direction, PriceBar, Resolution, ThresholdPass, Trade


class ThresholdPassTracker:
    def __init__(self, market_id: str, threshold: float = 0.55) -> None:
        self.market_id = market_id
        self.threshold = threshold
        self.is_above = False
        self.passes: list[ThresholdPass] = []

    def observe(self, timestamp: datetime, probability: float) -> ThresholdPass | None:
        if probability > self.threshold and not self.is_above and not self.passes:
            self.is_above = True
            threshold_pass = ThresholdPass(
                market_id=self.market_id,
                pass_number=1,
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


def rate_of_change(bars: list[PriceBar], lookback: int = 14) -> float | None:
    if lookback < 1:
        raise ValueError("Momentum lookback must be positive")
    if len(bars) <= lookback:
        return None
    start = bars[-lookback - 1].close
    if start <= 0:
        return None
    return bars[-1].close / start - 1.0


def ib_style_commission(quantity: float, order_value: float) -> float:
    return min(max(quantity * 0.005, 1.0), order_value * 0.01)


def average_true_range(previous_bars: list[PriceBar], period: int = 14) -> float | None:
    """Average true range over completed bars, including overnight gaps."""
    if len(previous_bars) <= period:
        return None
    selected = previous_bars[-period:]
    prior = previous_bars[-period - 1 : -1]
    ranges = [
        max(
            bar.high - bar.low,
            abs(bar.high - previous.close),
            abs(bar.low - previous.close),
        )
        for previous, bar in zip(prior, selected)
    ]
    return sum(ranges) / len(ranges)


class EventDrivenStrategy:
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
        direction: Direction = "long",
        portfolio: str = "polymarket_momentum",
        strategy_branch: str = "momentum",
        resolution: Resolution = "1h",
        predicted_target_price: float | None = None,
    ) -> Trade | None:
        atr = average_true_range(previous_bars, self.range_period)
        if atr is None or atr <= 0 or entry_bar.open <= 0:
            return None

        quantity = self.trade_notional / entry_bar.open
        commission = ib_style_commission(quantity, self.trade_notional)
        stop = (
            entry_bar.open - self.range_multiplier * atr
            if direction == "long"
            else entry_bar.open + self.range_multiplier * atr
        )
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
            lowest_price=entry_bar.open,
            final_outcome=final_outcome,
            portfolio=portfolio,
            strategy_branch=strategy_branch,
            resolution=resolution,
            direction=direction,
            predicted_target_price=predicted_target_price,
            range_period=self.range_period,
            range_multiplier=self.range_multiplier,
            maximum_price=entry_bar.open,
            minimum_price=entry_bar.open,
            stop_history=[{"timestamp": entry_bar.timestamp, "stop": stop}],
        )

    def update_trade(self, trade: Trade, bar: PriceBar, previous_bars: list[PriceBar]) -> bool:
        trade.maximum_price = max(trade.maximum_price or bar.high, bar.high)
        trade.minimum_price = min(trade.minimum_price or bar.low, bar.low)

        gap_through = (
            bar.open <= trade.current_stop
            if trade.direction == "long"
            else bar.open >= trade.current_stop
        )
        touched = gap_through or (
            bar.low <= trade.current_stop
            if trade.direction == "long"
            else bar.high >= trade.current_stop
        )
        if touched:
            trade.exit_at = bar.timestamp
            trade.exit_price = bar.open if gap_through else trade.current_stop
            trade.exit_commission = ib_style_commission(
                trade.quantity,
                trade.quantity * trade.exit_price,
            )
            trade.exit_reason = "trailing_stop"
            return True

        completed = previous_bars + [bar]
        atr = average_true_range(completed, self.range_period)
        if atr is None or atr <= 0:
            return False
        rolling = completed[-self.range_period :]
        trade.highest_price = max(item.high for item in rolling)
        trade.lowest_price = min(item.low for item in rolling)
        if trade.direction == "long":
            candidate = trade.highest_price - self.range_multiplier * atr
            trade.current_stop = max(trade.current_stop, candidate)
        else:
            candidate = trade.lowest_price + self.range_multiplier * atr
            trade.current_stop = min(trade.current_stop, candidate)
        trade.stop_history.append({"timestamp": bar.timestamp, "stop": trade.current_stop})
        return False

    @staticmethod
    def close_trade(trade: Trade, *, timestamp: datetime, price: float, reason: str) -> None:
        trade.exit_at = timestamp
        trade.exit_price = price
        trade.exit_commission = ib_style_commission(
            trade.quantity,
            trade.quantity * price,
        )
        trade.exit_reason = reason


EventDrivenLongStrategy = EventDrivenStrategy
