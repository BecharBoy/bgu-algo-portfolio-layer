import asyncio
import logging
import pandas as pd
from math import floor
from typing import List, Dict
from config import Settings
from DB import DB
from IB import IB_Connect
from Strategy import BaseStrategy
from Data_Feed import DataFeed
import schemas


class Portfolio:

    def __init__(self, settings: Settings, db: DB, ib: IB_Connect, datafeed: DataFeed):
        self.settings = settings
        self.db = db
        self.ib = ib
        self.datafeed = datafeed
        self.strategies: List[BaseStrategy] = []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self.strategies.append(strategy)

    # Replace _apply_risk_management in Portfolio.py

def _apply_risk_management(self, signals: List[Dict]) -> List[Dict]:
    # --- 1. Conflict resolution: opposing signals on same ticker → cancel both ---
    action_map: Dict[str, str] = {}
    conflicts: set = set()

    for sig in signals:
        sym = sig["symbol"]
        if sym in action_map:
            if action_map[sym] != sig["action"]:
                conflicts.add(sym)
        else:
            action_map[sym] = sig["action"]

    signals = [s for s in signals if s["symbol"] not in conflicts]
    if conflicts:
        logging.warning(
            f"risk_management: cancelled conflicting signals for: {conflicts}"
        )

    # --- 2. Max position count cap ---
    max_positions = self.settings.max_concurrent_positions  # add to Settings
    if len(signals) > max_positions:
        signals = signals[:max_positions]
        logging.warning(
            f"risk_management: capped signals to {max_positions}"
        )

    # --- 3. Per-signal weight cap: never > 20% NLV in one ticker ---
    for sig in signals:
        if sig.get("weight_allocation", 0) > 0.20:
            sig["weight_allocation"] = 0.20
            logging.warning(
                f"risk_management: capped weight_allocation for {sig['symbol']} to 0.20"
            )

    return signals


    def enforce_signal_timing(self, signals: List[Dict]) -> List[Dict]:
        # Signals generated on today's close execute next session only.
        # Currently enforced by job scheduling — revisit if intraday execution added.
        return signals

    async def _fetch_bars_for_all_tickers(self, lookback_days: int = 60) -> Dict[str, pd.DataFrame]:
        all_bars: Dict[str, pd.DataFrame] = {}
        for ticker in self.settings.universe:
            rows = await self.db.get_recent_bars(ticker, lookback_days=lookback_days)
            if rows:
                all_bars[ticker] = pd.DataFrame(rows).sort_values("date")
        return all_bars

    async def run_cycle(self) -> None:
        if not self.strategies:
            raise RuntimeError("No strategies registered — call add_strategy() first")

        current_prices = await self.datafeed.fetch_current_prices()
        open_positions_raw = await self.db.get_open_positions_from_db()
        open_positions = {p["symbol"]: p for p in open_positions_raw}
        account = await self.ib.get_account_summary()
        nlv = account["nlv"]
        cash = account["cash"]

        all_bars = await self._fetch_bars_for_all_tickers()

        raw_signals = []
        for strategy in self.strategies:
            signals = await strategy.generate_signals(all_bars, open_positions)
            raw_signals.extend(signals)

        if not raw_signals:
            logging.info("run_cycle: no signals generated this session.")
            return

        clean_signals = schemas.deduplicate_signals(raw_signals)
        clean_signals = self._apply_risk_management(clean_signals)

        orders = []
        for signal in clean_signals:
            ticker = signal["symbol"]
            price = current_prices.get(ticker)
            if not price:
                continue
            quantity = floor((nlv * signal["weight_allocation"]) / price)
            if quantity <= 0:
                continue
            orders.append({
                "order_id":   f"{signal['signal_id']}_order",
                "signal_id":  signal["signal_id"],
                "strategy":   signal["strategy"],
                "symbol":     ticker,
                "action":     signal["action"],
                "quantity":   quantity,
                "order_type": "MKT",
                "tif":        "DAY",
                "created_at": signal["timestamp"],
            })

        await self.db.save_signals(clean_signals)
        await self.db.save_orders(orders)

        for order in orders:
            await self.ib.place_market_order(
                action=order["action"],
                symbol=order["symbol"],
                quantity=order["quantity"],
            )

        fills = await self.ib.wait_for_order_updates(timeout_seconds=15)
        for fill in fills:
            await self.db.log_trade_execution(fill)

        await self.db.update_account_snapshot(nlv=nlv, cash=cash)

    async def dry_run_cycle(self) -> Dict:
        if not self.strategies:
            raise RuntimeError("No strategies registered")

        current_prices = await self.datafeed.fetch_current_prices()
        open_positions = await self.db.get_open_positions_from_db()
        open_positions_raw = await self.db.get_open_positions_from_db()
        open_positions = {p["symbol"]: p for p in open_positions_raw}
        account = await self.ib.get_account_summary()
        nlv = account["nlv"]

        all_bars = await self._fetch_bars_for_all_tickers()

        raw_signals = []
        for strategy in self.strategies:
            signals = await strategy.generate_signals(all_bars, open_positions)
            raw_signals.extend(signals)

        if not raw_signals:
            return {"signals": [], "orders": [], "note": "no signals generated"}

        clean_signals = schemas.deduplicate_signals(raw_signals)
        clean_signals = self._apply_risk_management(clean_signals)

        orders = []
        for signal in clean_signals:
            ticker = signal["symbol"]
            price = current_prices.get(ticker)
            if not price:
                continue
            quantity = floor((nlv * signal["weight_allocation"]) / price)
            if quantity <= 0:
                continue
            orders.append({
                "order_id":   f"{signal['signal_id']}_order",
                "signal_id":  signal["signal_id"],
                "strategy":   signal["strategy"],
                "symbol":     ticker,
                "action":     signal["action"],
                "quantity":   quantity,
                "order_type": "MKT",
                "tif":        "DAY",
                "created_at": signal["timestamp"],
            })

        return {"signals": clean_signals, "orders": orders}
