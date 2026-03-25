from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import pandas as pd

import schemas
from Strategy import BaseStrategy
from backtest.portfolio import Portfolio

log = logging.getLogger(__name__)


class BacktestEngine:
    """
    Multi-strategy backtest engine with per-strategy portfolio isolation.

    Each strategy operates on its own Portfolio instance (seeded at init).
    Pair signals (CointegrationArb) are executed atomically: if the pre-flight
    cash check fails for the pair, both legs are dropped.
    """

    def __init__(
        self,
        strategies: BaseStrategy | Sequence[BaseStrategy],
        portfolios: dict[str, Portfolio],   # strategy.name -> Portfolio
        data: dict[str, pd.DataFrame],
        start_date: str,
        end_date: str,
        lookback: int = 252,
        min_bars: int = 52,
    ) -> None:
        self.strategies  = [strategies] if isinstance(strategies, BaseStrategy) else list(strategies)
        self.portfolios  = portfolios
        self.data        = data
        self.start_date  = pd.Timestamp(start_date).normalize()
        self.end_date    = pd.Timestamp(end_date).normalize()
        self.lookback    = lookback
        self.min_bars    = min_bars

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        return matching.iloc[-1] if not matching.empty else None

    def _close_prices(self, trading_date: pd.Timestamp) -> dict[str, float]:
        return {
            ticker: float(bar["Close"])
            for ticker, frame in self.data.items()
            if (bar := self._bar_for_date(frame, trading_date)) is not None
        }

    def _open_prices(self, trading_date: pd.Timestamp) -> dict[str, float]:
        return {
            ticker: float(bar["Open"])
            for ticker, frame in self.data.items()
            if (bar := self._bar_for_date(frame, trading_date)) is not None
        }

    # ------------------------------------------------------------------
    # Pair grouping and atomic pre-flight
    # ------------------------------------------------------------------

    @staticmethod
    def _group_by_pair(signals: list[dict]) -> tuple[list[list[dict]], list[dict]]:
        """
        Splits signals into:
        - pair_groups: list of [leg_a, leg_b] lists (CointegArb pairs)
        - singles: everything else (MR signals and orphan single-leg signals)
        """
        pair_buckets: dict[str, list[dict]] = defaultdict(list)
        singles: list[dict] = []

        for sig in signals:
            pid = (sig.get("metadata") or {}).get("pair_id")
            if pid and sig.get("strategy") == "CointegrationArb":
                pair_buckets[pid].append(sig)
            else:
                singles.append(sig)

        pair_groups = list(pair_buckets.values())
        return pair_groups, singles

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    async def run(self) -> pd.DataFrame:
        if not self.strategies:
            raise RuntimeError("BacktestEngine requires at least one strategy")

        equity_curve: list[dict] = []
        trading_dates = self._get_trading_dates()

        for trading_date in trading_dates:
            window = self._build_window(trading_date)
            if not window:
                continue

            open_px  = self._open_prices(trading_date)
            close_px = self._close_prices(trading_date)
            date_str = trading_date.date().isoformat()

            # ── Signal generation: each strategy sees ONLY its own positions ──
            raw_signals: list[dict] = []
            for strategy in self.strategies:
                portfolio = self.portfolios.get(strategy.name)
                if portfolio is None:
                    log.error(f"No portfolio for strategy '{strategy.name}' — skipping.")
                    continue
                own_positions = portfolio.get_positions()
                strategy_signals = await strategy.generate_signals(window, own_positions)
                raw_signals.extend(strategy_signals)

            # ── Conflict resolution + deduplication ───────────────────────────
            clean_signals = schemas.deduplicate_signals(schemas.aggregate_signals(raw_signals))

            # ── Group by pair for atomic pre-flight ───────────────────────────
            pair_groups, singles = self._group_by_pair(clean_signals)

            # ── Execute pair groups (atomic) ──────────────────────────────────
            for legs in pair_groups:
                portfolio = self.portfolios.get("CointegrationArb")
                if portfolio is None:
                    continue
                # Pre-flight: can we afford ALL new-entry legs?
                entry_legs = [l for l in legs if not l.get("is_exit")]
                if entry_legs and not portfolio.can_afford_pair(entry_legs, open_px):
                    log.warning(
                        f"[ENGINE] Pair pre-flight failed for "
                        f"{[l['symbol'] for l in entry_legs]} — dropping pair."
                    )
                    continue
                # Execute legs in order
                for leg in legs:
                    symbol = leg["symbol"]
                    if symbol not in open_px:
                        continue
                    qty = portfolio.size_signal(leg, open_px[symbol])
                    portfolio.execute_signal(leg, open_px[symbol], date_str, qty)

            # ── Execute single (MR) signals ───────────────────────────────────
            for signal in singles:
                strategy_name = signal.get("strategy", "")
                portfolio = self.portfolios.get(strategy_name)
                if portfolio is None:
                    continue
                symbol = signal["symbol"]
                if symbol not in open_px:
                    continue
                qty = portfolio.size_signal(signal, open_px[symbol])
                portfolio.execute_signal(signal, open_px[symbol], date_str, qty)

            # ── Record combined equity curve ──────────────────────────────────
            total_equity = sum(p.get_equity(close_px) for p in self.portfolios.values())
            total_cash   = sum(p.cash for p in self.portfolios.values())
            total_pos    = sum(len(p.get_positions()) for p in self.portfolios.values())
            equity_curve.append({
                "date":          date_str,
                "equity":        total_equity,
                "cash":          total_cash,
                "num_positions": total_pos,
            })

        return pd.DataFrame(equity_curve)
