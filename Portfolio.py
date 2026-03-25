import asyncio
import logging
import uuid
import pandas as pd
from math import floor
from typing import List, Dict
from config import Settings
from DB import DB
from IB import IB_Connect
from Strategy import BaseStrategy
from Data_Feed import DataFeed
import schemas
import datetime
from schemas import Signal
logger = logging.getLogger(__name__)

# Fraction of each strategy's capital allocation used per trade.
# Each order notional = nlv * strategy.capital_allocation * TRADE_WEIGHT
# This ensures no single trade ever consumes more than 15% of a strategy's budget.
TRADE_WEIGHT = 0.15


class Portfolio:

    def __init__(self, settings: Settings, db: DB, ib: IB_Connect, datafeed: DataFeed):
        self.settings = settings
        self.db = db
        self.ib = ib
        self.datafeed = datafeed
        self.strategies: List[BaseStrategy] = []
        # Populated by data_quality_job before run_cycle; if None, fetch fresh.
        self._cached_bars: Dict[str, pd.DataFrame] | None = None
        # Mapping from strategy.name -> capital_allocation; populated by add_strategy().
        self._strategy_capital_allocations: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self.strategies.append(strategy)
        self._strategy_capital_allocations[strategy.name] = strategy.capital_allocation

    def _apply_risk_management(self, signals: List[Signal]) -> List[Signal]:
        # --- 1. Conflict resolution: opposing signals on same ticker cancel both ---
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
            logger.warning(f"[RISK] Cancelled conflicting signals for: {conflicts}")

        # --- 2. Max position count cap ---
        max_positions = self.settings.max_concurrent_positions
        if len(signals) > max_positions:
            signals = signals[:max_positions]
            logger.warning(f"[RISK] Capped signals to {max_positions}")

        return signals

    def enforce_signal_timing(self, signals: List[Dict]) -> List[Dict]:
        # Signals generated on today's close execute next session only.
        # Enforced by job scheduling — revisit if intraday execution is added.
        return signals

    async def _fetch_bars_for_all_tickers(
        self, lookback_days: int = 60
    ) -> Dict[str, pd.DataFrame]:
        all_bars: Dict[str, pd.DataFrame] = {}
        for ticker in self.settings.universe:
            rows = await self.db.get_recent_bars(ticker, lookback_days=lookback_days)
            if rows:
                all_bars[ticker] = pd.DataFrame(rows).sort_values("date")
        return all_bars

    async def _reconcile_positions(self) -> None:
        """
        End-of-cycle safety net.
        Compares live IB positions against DB positions and logs any mismatch
        as CRITICAL. Does not auto-fix — forces manual review via TWS.
        """
        ib_positions  = await self.ib.get_positions()
        db_positions_raw = await self.db.get_open_positions_from_db()
        db_positions  = {p["symbol"]: p["net_quantity"] for p in db_positions_raw}

        all_symbols = set(ib_positions.keys()) | set(db_positions.keys())
        has_mismatch = False

        for symbol in all_symbols:
            ib_qty = ib_positions.get(symbol, {}).get("quantity", 0)
            db_qty = db_positions.get(symbol, 0)
            if ib_qty != db_qty:
                has_mismatch = True
                logger.critical(
                    f"[RECONCILE] MISMATCH {symbol}: "
                    f"IB={ib_qty} vs DB={db_qty} — check TWS and fills table."
                )

        if not has_mismatch:
            logger.info("[RECONCILE] IB and DB positions are in sync.")

    async def run_cycle(self) -> None:
        if not self.strategies:
            raise RuntimeError("No strategies registered — call add_strategy() first")

        current_prices    = await self.datafeed.fetch_current_prices()
        open_positions_raw = await self.db.get_open_positions_from_db()
        open_positions    = {p["symbol"]: p for p in open_positions_raw}
        account           = await self.ib.get_account_summary()
        nlv               = account["nlv"]
        cash              = account["cash"]

        # Use pre-validated bars from data_quality_job if available,
        # otherwise fetch fresh (e.g. dry_run_cycle or standalone call).
        if self._cached_bars is not None:
            all_bars = self._cached_bars
            logger.info("[RUN] Using pre-validated bars from data_quality_job.")
        else:
            all_bars = await self._fetch_bars_for_all_tickers()
            logger.info("[RUN] No cached bars found — fetched fresh from DB.")

        # ── Signal generation ─────────────────────────────────────────────────
        raw_signals: List[Dict] = []
        for strategy in self.strategies:
            signals = await strategy.generate_signals(all_bars, open_positions)
            raw_signals.extend(signals)

        if not raw_signals:
            logger.info("[STRATEGY] No signals generated this session.")
            await self.db.update_account_snapshot(nlv=nlv, cash=cash)
            return

        aggregated_signals = schemas.aggregate_signals(raw_signals)
        clean_signals = schemas.deduplicate_signals(aggregated_signals)
        clean_signals = self._apply_risk_management(clean_signals)
        logger.info(f"[STRATEGY] {len(clean_signals)} clean signal(s) after risk management.")

        # ── Order sizing ──────────────────────────────────────────────────────
        pending_orders = []
        for signal in clean_signals:
            ticker = signal["symbol"]
            price  = current_prices.get(ticker)
            if not price:
                logger.error(f"[ORDER] No reference price for {ticker} — skipping signal.")
                continue

            strategy_name = signal["strategy"]
            strategy_capital_allocation = self._strategy_capital_allocations.get(strategy_name)
            if strategy_capital_allocation is None:
                logger.error(f"[ORDER] No capital_allocation for strategy '{strategy_name}' — skipping signal.")
                continue

            quantity = floor((nlv * strategy_capital_allocation * TRADE_WEIGHT) / price)
            if quantity <= 0:
                logger.warning(f"[ORDER] Quantity rounds to 0 for {ticker} — skipping.")
                continue
            pending_orders.append({
                "signal_id":  signal["signal_id"],
                "strategy":   signal["strategy"],
                "symbol":     ticker,
                "action":     signal["action"],
                "quantity":   quantity,
                "order_type": "LMT",
                "tif":        "DAY",
                "created_at": signal["timestamp"],
            })
        await self.db.save_signals(clean_signals)

        # ── Order execution ───────────────────────────────────────────────────
        for pending in pending_orders:
            reference_price = current_prices.get(pending["symbol"])
            if not reference_price:
                logger.error(f"[ORDER] No reference price for {pending['symbol']} at execution — skipping.")
                continue

            ib_result = await self.ib.place_limit_order(
                action=pending["action"],
                symbol=pending["symbol"],
                quantity=pending["quantity"],
                reference_price=reference_price,
            )
            ib_order_id = ib_result["order_id"]
            order = {
                "order_id": str(ib_order_id),
                "signal_id": pending["signal_id"],
                "strategy": pending["strategy"],
                "symbol": pending["symbol"],
                "action": pending["action"],
                "quantity": pending["quantity"],
                "order_type": pending["order_type"],
                "tif": pending["tif"],
                "created_at": pending["created_at"],
            }
            await self.db.save_orders([order])
            fill = await self.ib.wait_for_order_fill(order_id=ib_order_id)

            if fill is not None:
                if fill["symbol"] != order["symbol"] or fill["quantity"] != order["quantity"]:
                    logger.error(
                        f"[FILL] Mismatch on order_id={ib_order_id}: "
                        f"expected {order['quantity']} {order['symbol']}, "
                        f"got {fill['quantity']} {fill['symbol']} — skipping DB write."
                    )
                    continue
                await self.db.log_trade_execution(fill)
                logger.info(
                    f"[FILL] {fill['action']} {fill['quantity']} {fill['symbol']} "
                    f"@ {fill['fill_price']} ✓"
                )

            else:
                logger.warning(
                    f"[TIMEOUT] order_id={ib_order_id} "
                    f"({order['action']} {order['quantity']} {order['symbol']}) "
                    f"did not fill in 15s — sending cancel."
                )
                final_status = await self.ib.cancel_order(ib_order_id)

                if final_status == "Filled":
                    late_fills = self.ib.get_fills_for_order(ib_order_id)
                    if late_fills:
                        f = late_fills[-1]
                        await self.db.log_trade_execution({
                            "order_id": str(ib_order_id),
                            "symbol": f.contract.symbol,
                            "action": "BUY" if f.execution.side == "BOT" else "SELL",
                            "quantity": int(f.execution.cumQty),
                            "fill_price": f.execution.price,
                            "filled_at": f.execution.time,
                            "fill_id": str(uuid.uuid4()),
                        })
                        logger.info(f"[FILL] Late fill on order_id={ib_order_id} captured and logged ✓")

                    elif final_status == "PartiallyFilled":
                        partial_trade = next(
                        (t for t in self.ib.ib.trades()
                         if t.order.orderId == ib_order_id), None
                    )
                        filled_so_far = int(partial_trade.orderStatus.filled) if partial_trade else 0

                        if filled_so_far > 0:
                            # Log the original partial fill FIRST
                            await self.db.log_trade_execution({
                                "fill_id": str(uuid.uuid4()),
                                "order_id": str(ib_order_id),
                                "symbol": order["symbol"],
                                "action": order["action"],
                                "quantity": filled_so_far,
                                "fill_price": partial_trade.orderStatus.avgFillPrice,
                                "filled_at": datetime.now(datetime.timezone.utc).isoformat(),
                            })
                            logger.warning(f"[PARTIAL] Logged original partial: {filled_so_far} {order['symbol']}")

                            # Then flatten
                            logger.warning(
                                f"[PARTIAL] order_id={ib_order_id}: "
                                f"{filled_so_far}/{order['quantity']} {order['symbol']} filled "
                                f"before cancel — flattening immediately."
                            )
                            flatten_result = await self.ib.flatten_position(
                                symbol=order["symbol"],
                                quantity=filled_so_far,
                                action=order["action"],
                            )
                            flatten_fill = await self.ib.wait_for_order_fill(
                                order_id=flatten_result["order_id"]
                            )
                            if flatten_fill:
                                await self.db.log_trade_execution(flatten_fill)
                                logger.warning(
                                    f"[FLATTEN] Fill logged: "
                                    f"{flatten_fill['action']} {flatten_fill['quantity']} "
                                    f"{flatten_fill['symbol']} @ {flatten_fill['fill_price']}"
                                )
                            else:
                                logger.critical(
                                    f"[FLATTEN] Flatten order for {order['symbol']} did not confirm fill — "
                                    f"position may be open. Check TWS immediately."
                                )

                elif final_status in ("CancelTimeout", "NotFound"):
                    logger.critical(
                        f"[CANCEL] order_id={ib_order_id} final status is '{final_status}' — "
                        f"order state unknown. Check TWS immediately. "
                        f"DB NOT updated for this order."
                    )

                else:
                    logger.info(
                        f"[CANCEL] order_id={ib_order_id} "
                        f"({order['action']} {order['quantity']} {order['symbol']}) "
                        f"cancelled cleanly — no DB write."
                    )

        # ── End-of-cycle: account snapshot + IB vs DB reconciliation ─────────
        await self.db.update_account_snapshot(nlv=nlv, cash=cash)
        await self._reconcile_positions()

    async def dry_run_cycle(self) -> Dict:
        if not self.strategies:
            raise RuntimeError("No strategies registered")

        current_prices     = await self.datafeed.fetch_current_prices()
        open_positions_raw = await self.db.get_open_positions_from_db()
        open_positions     = {p["symbol"]: p for p in open_positions_raw}
        account            = await self.ib.get_account_summary()
        nlv                = account["nlv"]

        all_bars = await self._fetch_bars_for_all_tickers()

        raw_signals: List[Dict] = []
        for strategy in self.strategies:
            signals = await strategy.generate_signals(all_bars, open_positions)
            raw_signals.extend(signals)

        if not raw_signals:
            return {"signals": [], "orders": [], "note": "no signals generated"}

        aggregated_signals = schemas.aggregate_signals(raw_signals)
        clean_signals = schemas.deduplicate_signals(aggregated_signals)
        clean_signals = self._apply_risk_management(clean_signals)

        orders = []
        for signal in clean_signals:
            ticker = signal["symbol"]
            price  = current_prices.get(ticker)
            if not price:
                continue

            strategy_name = signal["strategy"]
            strategy_capital_allocation = self._strategy_capital_allocations.get(strategy_name)
            if strategy_capital_allocation is None:
                logger.error(f"[DRY_RUN] No capital_allocation for strategy '{strategy_name}' — skipping signal.")
                continue

            quantity = floor((nlv * strategy_capital_allocation * TRADE_WEIGHT) / price)
            if quantity <= 0:
                continue
            orders.append({
                "order_id":   f"{signal['signal_id']}_order",
                "signal_id":  signal["signal_id"],
                "strategy":   signal["strategy"],
                "symbol":     ticker,
                "action":     signal["action"],
                "quantity":   quantity,
                "order_type": "LMT",
                "tif":        "DAY",
                "created_at": signal["timestamp"],
            })

        return {"signals": clean_signals, "orders": orders}
