from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

import schemas
from Strategy import BaseStrategy
from backtest.portfolio import Portfolio


class BacktestEngine:
    def __init__(
        self,
        strategies: BaseStrategy | Sequence[BaseStrategy],
        portfolio: Portfolio,
        data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        lookback: int = 252,
        min_bars: int = 52,
    ) -> None:
        if isinstance(strategies, BaseStrategy):
            self.strategies = [strategies]
        else:
            self.strategies = list(strategies)
        self.portfolio = portfolio
        self.data = data
        self.start_date = pd.Timestamp(start_date).normalize()
        self.end_date = pd.Timestamp(end_date).normalize()
        self.lookback = lookback
        self.min_bars = min_bars
        self._strategy_allocations = {
            strategy.name: getattr(strategy, "capital_allocation", 0.0)
            for strategy in self.strategies
        }

    def _get_trading_dates(self) -> list[pd.Timestamp]:
        all_dates: set[pd.Timestamp] = set()
        for frame in self.data.values():
            in_range = frame[(frame["date"] >= self.start_date) & (frame["date"] <= self.end_date)]
            all_dates.update(in_range["date"].tolist())
        return sorted(all_dates)

    def _build_window(self, as_of_date: pd.Timestamp) -> dict[str, pd.DataFrame]:
        window: dict[str, pd.DataFrame] = {}
        for ticker, frame in self.data.items():
            history = frame[frame["date"] < as_of_date].tail(self.lookback).copy()
            if len(history) >= self.min_bars:
                history.reset_index(drop=True, inplace=True)
                window[ticker] = history
        return window

    def _bar_for_date(self, frame: pd.DataFrame, trading_date: pd.Timestamp) -> pd.Series | None:
        matching = frame[frame["date"] == trading_date]
        if matching.empty:
            return None
        return matching.iloc[-1]

    async def run(self) -> pd.DataFrame:
        if not self.strategies:
            raise RuntimeError("BacktestEngine requires at least one strategy")

        equity_curve: list[dict] = []
        trading_dates = self._get_trading_dates()

        for trading_date in trading_dates:
            window = self._build_window(trading_date)
            if not window:
                continue

            raw_signals: list[dict] = []
            current_positions = self.portfolio.get_positions()
            for strategy in self.strategies:
                strategy_signals = await strategy.generate_signals(window, current_positions)
                raw_signals.extend(strategy_signals)

            clean_signals = schemas.deduplicate_signals(schemas.aggregate_signals(raw_signals))

            equity_before_execution = self.portfolio.get_equity(
                {
                    ticker: float(bar["Close"])
                    for ticker, frame in self.data.items()
                    if (bar := self._bar_for_date(frame, trading_date)) is not None
                }
            )

            for signal in clean_signals:
                symbol = signal["symbol"]
                frame = self.data.get(symbol)
                if frame is None:
                    continue

                bar = self._bar_for_date(frame, trading_date)
                if bar is None:
                    continue

                execution_price = float(bar["Open"])
                default_allocation = self._strategy_allocations.get(signal["strategy"], 0.0)
                quantity = self.portfolio.size_signal(
                    signal=signal,
                    execution_price=execution_price,
                    portfolio_equity=equity_before_execution,
                    default_allocation=default_allocation,
                )
                self.portfolio.execute_signal(
                    signal=signal,
                    execution_price=execution_price,
                    date=trading_date.date().isoformat(),
                    quantity=quantity,
                )

            close_prices = {
                ticker: float(bar["Close"])
                for ticker, frame in self.data.items()
                if (bar := self._bar_for_date(frame, trading_date)) is not None
            }
            equity_curve.append(
                {
                    "date": trading_date.date().isoformat(),
                    "equity": self.portfolio.get_equity(close_prices),
                    "cash": self.portfolio.cash,
                    "num_positions": len(self.portfolio.get_positions()),
                }
            )

        return pd.DataFrame(equity_curve)
