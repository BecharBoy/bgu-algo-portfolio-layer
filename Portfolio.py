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
from datetime import datetime, UTC
from schemas import Signal

logger = logging.getLogger(__name__)

# Fraction of available cash used per new entry order.
# quantity = floor(available_cash * TRADE_WEIGHT / price)
TRADE_WEIGHT = 0.15


class Portfolio:
    """
    Live trading portfolio with per-strategy position isolation.

    Each strategy's capital is tracked separately using the DB.
    Positions are owned by the strategy that opened them (fills.strategy column).
    Only the owning strategy can generate exit signals for its positions.
    """

    def __init__(self, settings: Settings, db: DB, ib: IB_Connect, datafeed: DataFeed):
        self.settings   = settings
        self.db         = db
        self.ib         = ib
        self.datafeed   = datafeed
        self.strategies: List[BaseStrategy] = []
        self._cached_bars: Dict[str, pd.DataFrame] | None = None
        self._strategy_capital_allocations: Dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self.strategies.append(strategy)
        self._strategy_capital_allocations[strategy.name] = strategy.capital_allocation

    # ── Risk management ───────────────────────────────────────────────────────

    def _apply_risk_management(self, signals: List[Signal]) -> List[Signal]:
        """
        1. Exit signals are NEVER cancelled (they close real positions).
        2. Conflicting *entry* signals on the same symbol cancel both entries.
        3. Max concurrent position cap.
        """
        exits   = [s for s in signals if s.get("is_exit")]
        entries = [s for s in signals if not s.get("is_exit")]

        # Conflict resolution on entries only
        action_map: Dict[str, str] = {}
        conflicts: set = set()
        for sig in entries:
            sym = sig["symbol"]
            if sym in action_map:
                if action_map[sym] != sig["action"]:
                    conflicts.add(sym)
            else:
                action_map[sym] = sig["action"]

        if conflicts:
            logger.warning(f"[RISK] Cancelled conflicting entry signals for: {conflicts}")
        entries = [s for s in entries if s["symbol"] not in conflicts]

        max_positions = self.settings.max_concurrent_positions
        if len(entries) > max_positions:
            entries = entries[:max_positions]
            logger.warning(f"[RISK] Capped entry signals to {max_positions}")

        return exits + entries

    # ── Bar fetching ──────────────────────────────────────────────────────────

    async def _fetch_bars_for_all_tickers(self, lookback_days: int = 60) -> Dict[str, pd.DataFrame]:
        all_bars: Dict[str, pd.DataFrame] = {}
        for ticker in self.settings.universe:
            rows = await self.db.get_recent_bars(ticker, lookback_days=lookback_days)
            if rows:
                all_bars[ticker] = pd.DataFrame(rows).sort_values("date")
        return all_bars

    # ── Reconciliation ────────────────────────────────────────────────────────

    async def _reconcile_positions(self) -> None:
        ib_positions     = await self.ib.get_positions()
        db_positions_raw = await self.db.get_open_positions_from_db()
        db_positions     = {p["symbol"]: p["net_quantity"] for p in db_positions_raw}

        all_symbols = set(ib_positions.keys()) | set(db_positions.keys())
        has_mismatch = False
        for symbol in all_symbols:
            ib_qty = ib_positions.get(symbol, {}).get("quantity", 0)
            db_qty = db_positions.get(symbol, 0)
            if ib_qty != db_qty:
                has_mismatch = True
                logger.critical(
                    f"[RECONCILE] MISMATCH {symbol}: IB={ib_qty} vs DB={db_qty} "
                    f"— check TWS and fills table."
                )
        if not has_mismatch:
            logger.info("[RECONCILE] IB and DB positions are in sync.")

    # ── Pair pre-flight cash check ────────────────────────────────────────────

    def _can_afford_pair(
        self,
        entry_legs: List[Dict],
        prices: Dict[str, float],
        available_cash: float,
    ) -> bool:
        """
        Simulate sequential cash deduction for all new-entry legs of a pair.
        Returns False if any leg would exhaust cash.
        """
        sim_cash = available_cash
        for leg in entry_legs:
            price = prices.get(leg["symbol"])
            if not price:
                return False
            qty = floor(sim_cash * TRADE_WEIGHT / price)
            if qty <= 0:
                return False
            sim_cash -= qty * price
            if sim_cash < 0:
                return False
        return True

    # ── Main run cycle ────────────────────────────────────────────────────────

    async def run_cycle(self) -> None:
        if not self.strategies:
            raise RuntimeError("No strategies registered — call add_strategy() first")

        current_prices = await self.datafeed.fetch_current_prices()
        account        = await self.ib.get_account_summary()
        nlv            = account["nlv"]
        cash           = account["cash"]

        if self._cached_bars is not None:
            all_bars = self._cached_bars
            logger.info("[RUN] Using pre-validated bars from data_quality_job.")
        else:
            all_bars = await self._fetch_bars_for_all_tickers()
            logger.info("[RUN] No cached bars — fetched fresh from DB.")

        # ── Signal generation: each strategy sees ONLY its own positions ──────
        raw_signals: List[Dict] = []
        for strategy in self.strategies:
            # Filter DB positions to this strategy only
            own_positions_raw = await self.db.get_open_positions_from_db(strategy=strategy.name)
            own_positions     = {p["symbol"]: p for p in own_positions_raw}
            strategy_signals  = await strategy.generate_signals(all_bars, own_positions)
            raw_signals.extend(strategy_signals)

        if not raw_signals:
            logger.info("[STRATEGY] No signals generated this session.")
            await self.db.update_account_snapshot(nlv=nlv, cash=cash)
            return

        aggregated = schemas.aggregate_signals(raw_signals)
        clean      = schemas.deduplicate_signals(aggregated)
        clean      = self._apply_risk_management(clean)
        logger.info(f"[STRATEGY] {len(clean)} clean signal(s) after risk management.")

        # ── Group CointegArb pair legs together for atomic pre-flight ─────────
        from collections import defaultdict
        pair_buckets: Dict[str, List[Dict]] = defaultdict(list)
        singles:      List[Dict]            = []
        for sig in clean:
            pid = (sig.get("metadata") or {}).get("pair_id")
            if pid and sig.get("strategy") == "CointegrationArb" and not sig.get("is_exit"):
                pair_buckets[pid].append(sig)
            else:
                singles.append(sig)

        # Pre-flight for entry pairs
        approved_pair_legs: List[Dict] = []
        for pid, legs in pair_buckets.items():
            if self._can_afford_pair(legs, current_prices, cash):
                approved_pair_legs.extend(legs)
            else:
                logger.warning(f"[RISK] Pair pre-flight failed for pair_id={pid} "
                               f"symbols={[l['symbol'] for l in legs]} — dropping pair.")

        # Exits always approved
        exit_signals   = [s for s in clean if s.get("is_exit")]
        to_execute     = exit_signals + approved_pair_legs + singles

        # ── Sizing and execution ──────────────────────────────────────────────
        await self.db.save_signals(clean)
        pending_orders = []
        for signal in to_execute:
            ticker = signal["symbol"]
            price  = current_prices.get(ticker)
            if not price:
                logger.error(f"[ORDER] No reference price for {ticker} — skipping.")
                continue

            strategy_name = signal["strategy"]

            if signal.get("is_exit"):
                # Exit: close the full open position
                own_pos_raw = await self.db.get_open_positions_from_db(strategy=strategy_name)
                own_pos     = {p["symbol"]: p for p in own_pos_raw}
                pos         = own_pos.get(ticker)
                if not pos:
                    logger.warning(f"[ORDER] Exit signal for {ticker} but no open position — skipping.")
                    continue
                quantity = abs(pos["net_quantity"])
            else:
                quantity = floor(cash * TRADE_WEIGHT / price)

            if quantity <= 0:
                logger.warning(f"[ORDER] Quantity=0 for {ticker} — skipping.")
                continue

            pending_orders.append({
                "signal_id":  signal["signal_id"],
                "strategy":   strategy_name,
                "symbol":     ticker,
                "action":     signal["action"],
                "quantity":   quantity,
                "order_type": "LMT",
                "tif":        "DAY",
                "created_at": signal["timestamp"],
                "pair_id":    (signal.get("metadata") or {}).get("pair_id"),
            })

        # ── Place orders ──────────────────────────────────────────────────────
        for pending in pending_orders:
            ref_price = current_prices.get(pending["symbol"])
            if not ref_price:
                continue

            ib_result   = await self.ib.place_limit_order(
                action=pending["action"], symbol=pending["symbol"],
                quantity=pending["quantity"], reference_price=ref_price,
            )
            ib_order_id = ib_result["order_id"]
            order = {
                "order_id":   str(ib_order_id),
                "signal_id":  pending["signal_id"],
                "strategy":   pending["strategy"],
                "symbol":     pending["symbol"],
                "action":     pending["action"],
                "quantity":   pending["quantity"],
                "order_type": pending["order_type"],
                "tif":        pending["tif"],
                "created_at": pending["created_at"],
            }
            await self.db.save_orders([order])
            fill = await self.ib.wait_for_order_fill(order_id=ib_order_id)

            if fill is not None:
                if fill["symbol"] != order["symbol"] or fill["quantity"] != order["quantity"]:
                    logger.error(f"[FILL] Mismatch on order_id={ib_order_id} — skipping DB write.")
                    continue
                fill["strategy"] = pending["strategy"]
                fill["pair_id"]  = pending["pair_id"]
                await self.db.log_trade_execution(fill)
                logger.info(f"[FILL] {fill['action']} {fill['quantity']} {fill['symbol']} @ {fill['fill_price']} ✓")

            else:
                logger.warning(f"[TIMEOUT] order_id={ib_order_id} did not fill in 15s — cancelling.")
                final_status = await self.ib.cancel_order(ib_order_id)

                if final_status == "Filled":
                    late_fills = self.ib.get_fills_for_order(ib_order_id)
                    if late_fills:
                        f = late_fills[-1]
                        trade_data = {
                            "order_id":   str(ib_order_id),
                            "symbol":     f.contract.symbol,
                            "action":     "BUY" if f.execution.side == "BOT" else "SELL",
                            "quantity":   int(f.execution.cumQty),
                            "fill_price": f.execution.price,
                            "filled_at":  f.execution.time,
                            "fill_id":    str(uuid.uuid4()),
                            "strategy":   pending["strategy"],
                            "pair_id":    pending["pair_id"],
                        }
                        await self.db.log_trade_execution(trade_data)
                        logger.info(f"[FILL] Late fill on order_id={ib_order_id} captured ✓")

                elif final_status == "PartiallyFilled":
                    partial_trade = next(
                        (t for t in self.ib.ib.trades() if t.order.orderId == ib_order_id), None
                    )
                    filled_so_far = int(partial_trade.orderStatus.filled) if partial_trade else 0
                    if filled_so_far > 0:
                        await self.db.log_trade_execution({
                            "fill_id":    str(uuid.uuid4()),
                            "order_id":   str(ib_order_id),
                            "symbol":     order["symbol"],
                            "action":     order["action"],
                            "quantity":   filled_so_far,
                            "fill_price": partial_trade.orderStatus.avgFillPrice,
                            "filled_at":  datetime.now(UTC).isoformat(),
                            "strategy":   pending["strategy"],
                            "pair_id":    pending["pair_id"],
                        })
                        logger.warning(f"[PARTIAL] Logged {filled_so_far} {order['symbol']} — flattening.")
                        flatten_result = await self.ib.flatten_position(
                            symbol=order["symbol"], quantity=filled_so_far, action=order["action"]
                        )
                        flatten_fill = await self.ib.wait_for_order_fill(order_id=flatten_result["order_id"])
                        if flatten_fill:
                            flatten_fill["strategy"] = pending["strategy"]
                            flatten_fill["pair_id"]  = pending["pair_id"]
                            await self.db.log_trade_execution(flatten_fill)
                            logger.warning(f"[FLATTEN] Fill logged: {flatten_fill}")
                        else:
                            logger.critical(f"[FLATTEN] {order['symbol']} did not confirm — check TWS.")

                elif final_status in ("CancelTimeout", "NotFound"):
                    logger.critical(f"[CANCEL] order_id={ib_order_id} status='{final_status}' — check TWS.")
                else:
                    logger.info(f"[CANCEL] order_id={ib_order_id} cancelled cleanly.")

        await self.db.update_account_snapshot(nlv=nlv, cash=cash)
        await self._reconcile_positions()

    # ── Dry run ───────────────────────────────────────────────────────────────

    async def dry_run_cycle(self) -> Dict:
        if not self.strategies:
            raise RuntimeError("No strategies registered")

        current_prices = await self.datafeed.fetch_current_prices()
        account        = await self.ib.get_account_summary()
        nlv            = account["nlv"]
        cash           = account["cash"]
        all_bars       = await self._fetch_bars_for_all_tickers()

        raw_signals: List[Dict] = []
        for strategy in self.strategies:
            own_pos_raw = await self.db.get_open_positions_from_db(strategy=strategy.name)
            own_pos     = {p["symbol"]: p for p in own_pos_raw}
            raw_signals.extend(await strategy.generate_signals(all_bars, own_pos))

        if not raw_signals:
            return {"signals": [], "orders": [], "note": "no signals generated"}

        clean = schemas.deduplicate_signals(schemas.aggregate_signals(raw_signals))
        clean = self._apply_risk_management(clean)

        orders = []
        for signal in clean:
            ticker = signal["symbol"]
            price  = current_prices.get(ticker)
            if not price:
                continue
            quantity = floor(cash * TRADE_WEIGHT / price)
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

        return {"signals": clean, "orders": orders}
