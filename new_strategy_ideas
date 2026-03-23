# 03_ARCHITECTURE_AND_OPTIMIZATION.md

## What This Document Is

This document covers every design decision, known bottleneck,
concurrency problem, and correctness issue in my_traders.
Each problem is explained from first principles — why it is
a problem, what breaks, and how to fix it.

---

## 1. Current Architecture Overview

```
main.py
  └── Portfolio
        ├── DataFeed          (price fetching from IB Ticker API)
        ├── DB                (asyncpg → PostgreSQL)
        ├── IB_Connect        (ib_async → Interactive Brokers TWS)
        └── [Strategies]
              ├── MeanReversionMomentum   (pure Python, talib)
              └── StatArbStrategy         (Python → Wrapper → C++)
                                                         └── cointegration_engine.so
                                                               ├── MarketData.cpp
                                                               ├── PairScanner.cpp
                                                               ├── MathStats.cpp
                                                               └── binding.cpp (pybind11)
```

**Data flow per cycle:**
```
IB TWS → DataFeed.fetch_current_prices()
DB      → get_recent_bars() [per ticker]
                ↓
        Strategy.generate_signals()
                ↓
        schemas.deduplicate_signals()
        Portfolio._apply_risk_management()
                ↓
        DB.save_signals() + DB.save_orders()
        IB.place_market_order() [per order]
        IB.wait_for_order_updates()
        DB.log_trade_execution()
        DB.update_account_snapshot()
```

---

## 2. Concurrency Model

### 2.1 Python asyncio Layer

Python runs a single-threaded event loop (asyncio). All I/O
operations — DB queries, IB API calls, DataFeed HTTP requests
— are coroutines that yield control when waiting. This means:

- Multiple DB queries can be in-flight simultaneously
- IB order placement does not block price fetching
- No CPU work runs in parallel (GIL prevents it)

**This is correct for I/O-bound work.** Database reads,
network calls, broker API — all I/O. asyncio is the right
model.

### 2.2 C++ Thread Pool Layer

CPU-bound work — 124,750 correlation calculations — is
offloaded to a C++ thread pool via pybind11. The GIL is
released before `scanner.scan_all_pairs()` is called:

```cpp
py::gil_scoped_release release;
top_pairs = scanner.scan_all_pairs(market_data, num_threads, min_correlation);
```

This means the Python event loop is **free during C++
computation**. Other coroutines can run. This is correct.

### 2.3 The Boundary

```
Python asyncio (I/O concurrency)
        ↕  [GIL released at this boundary]
C++ thread pool (CPU parallelism)
```

These two concurrency models do not interfere because the
GIL is explicitly released before C++ threads are spawned
and recaptured before returning Python objects.

---

## 3. Known Bottlenecks

### 3.1 Sequential DB Fetching — O(N) Round Trips

**The Problem:**
```python
for ticker in self.settings.universe:
    rows = await self.db.get_recent_bars(ticker, lookback_days=60)
```

For 100 tickers, this is 100 sequential database round trips.
Each round trip has latency (network + query parsing +
PostgreSQL planner). On a local DB: ~1ms per query = 100ms
total. On a remote DB: ~5ms per query = 500ms total.

This is O(N) latency where N is universe size. As N grows,
this blocks the event loop for longer.

**Why It Blocks:**
Even though each `await` yields control, the queries execute
one-by-one. asyncio is concurrent but not parallel for DB
queries unless you explicitly fire them simultaneously.

**The Fix: Single Bulk Query**
```python
async def _fetch_bars_for_all_tickers(
    self, lookback_days: int = 60
) -> Dict[str, pd.DataFrame]:

    rows = await self.db.get_recent_bars_bulk(
        self.settings.universe, lookback_days
    )
    all_bars: Dict[str, pd.DataFrame] = {}
    for row in rows:
        ticker = row["ticker"]
        if ticker not in all_bars:
            all_bars[ticker] = []
        all_bars[ticker].append(row)

    return {
        t: pd.DataFrame(bars).sort_values("date")
        for t, bars in all_bars.items()
    }
```

```sql
-- In DB.py: get_recent_bars_bulk
SELECT ticker, date, open, high, low, close, volume
FROM ohlcv_bars
WHERE ticker = ANY($1)
  AND date >= CURRENT_DATE - INTERVAL '$2 days'
ORDER BY ticker, date ASC;
```

This is 1 query regardless of N. Speedup: 100x for 100 tickers.

### 3.2 Mutex Bottleneck in PairScanner — Fixed in Code Fix 7

**The Problem (original code):**
```cpp
if (correlation >= min_correlation) {
    std::lock_guard<std::mutex> lock(results_mutex);
    top_pairs.push_back(result);   // every hit acquires the same lock
}
```

For a universe with many correlated pairs (e.g., 100 tech
stocks during a bull market), many threads hit this lock
simultaneously. Lock contention causes threads to wait,
serializing what should be parallel work.

**Amdahl's Law implication:** If 20% of time is spent in
the critical section, even infinite threads give only 5x
speedup. You want 0% time in critical sections during
computation.

**The Fix (Code Fix 7):** Each thread writes to its own
`std::vector<PairResult>`. No lock. Single-threaded merge
after all threads join. The merge is O(total_pairs) and
runs once.

### 3.3 `asyncio.sleep(15)` in IB Fill Waiting

**The Problem:**
```python
fills = await self.ib.wait_for_order_updates(timeout_seconds=15)
```

This always waits 15 seconds. If all fills arrive in 0.5
seconds (common for market orders on liquid stocks), you
waste 14.5 seconds per cycle. This blocks the entire
run_cycle coroutine during that time.

**The Fix: Event-Based Waiting**
```python
# In IB.py — replace sleep-based waiting with event callback
async def wait_for_order_updates(
    self, expected_count: int, timeout_seconds: int = 15
) -> List[Dict]:
    fills = []
    done_event = asyncio.Event()

    def on_fill(trade, fill):
        fills.append({
            "order_id":    str(trade.order.orderId),
            "symbol":      trade.contract.symbol,
            "action":      trade.order.action,
            "quantity":    fill.execution.shares,
            "fill_price":  fill.execution.price,
            "fill_id":     fill.execution.execId,
            "timestamp":   fill.time.isoformat(),
        })
        if len(fills) >= expected_count:
            done_event.set()

    self.ib.execDetailsEvent += on_fill

    try:
        await asyncio.wait_for(done_event.wait(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        pass
    finally:
        self.ib.execDetailsEvent -= on_fill

    return fills
```

Pass `expected_count=len(orders)` from `run_cycle`. Returns
as soon as all fills arrive. Falls back to timeout if any
fill is missing (partial fill, rejection).

### 3.4 Scalability: 500+ Tickers

**The Problem:**
At N=500 tickers, PairScanner processes 124,750 pairs.
At N=2000 (Russell 2000), it processes 1,999,000 pairs.
Even in C++, this is ~80ms per cycle at 2000 tickers.

More importantly: 2000 tickers × 60 days = 120,000 prices
loaded from DB every cycle. The bulk query still runs, but
the `price_matrix` in MarketData becomes 960KB of data —
fine for memory, but the ADF test on every passing pair
becomes expensive.

**The Fix: PCA Pre-Filter**
Before correlation scanning, reduce 2000 price series to
their top-K principal components. Pairs that are not
in the same principal component cluster are unlikely to
be cointegrated. This prunes the candidate space from
O(N²) to O(K·N) where K << N.

```python
from sklearn.decomposition import PCA

def pca_prefilter(price_matrix: np.ndarray, n_components: int = 20):
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(price_matrix.T)
    # Cluster tickers by dominant component
    # Only scan pairs within the same cluster
    ...
```

This is an optimization for when you scale beyond ~500
tickers. At 100-200 tickers (your current realistic universe),
it is not needed.

---

## 4. Correctness Problem: IB ↔ DB State Consistency

### 4.1 The Problem

Your system has two independent sources of truth:
1. **Interactive Brokers TWS** — what positions you actually hold
2. **PostgreSQL DB** — what positions my_traders thinks you hold

These can diverge whenever:
- The process crashes between `place_market_order()` and
  `log_trade_execution()`
- IB rejects an order silently (no fill arrives)
- TWS disconnects mid-cycle and reconnects
- You manually close a position in TWS

When they diverge, the next `run_cycle()` reads wrong
`current_positions`, generates signals based on false state,
and potentially double-trades.

### 4.2 Why This Is a Distributed Systems Problem

IB TWS and PostgreSQL have no shared transaction boundary.
You cannot do:

```python
# This does NOT exist — two systems, no atomic transaction
with atomic(ib, db):
    ib.place_order(...)
    db.log_execution(...)
```

Any crash between these two operations leaves them
inconsistent. This is the classic **two-phase commit** problem.

### 4.3 The Fix: Reconciliation on Startup

Before any `run_cycle()` is called, run a reconciliation
check that compares IB positions to DB positions and resolves
discrepancies:

```python
# In main.py — call before first run_cycle()
async def reconcile_positions(ib: IB_Connect, db: DB) -> None:
    ib_positions  = await ib.get_positions()   # {symbol: quantity}
    db_positions  = await db.get_open_positions_from_db()
    db_pos_map    = {p["symbol"]: p for p in db_positions}

    for symbol, ib_qty in ib_positions.items():
        if symbol not in db_pos_map:
            # IB has a position DB doesn't know about
            # Log it as an externally opened position
            logging.warning(
                f"reconcile: IB has {symbol} x{ib_qty} not in DB — logging"
            )
            await db.insert_reconciled_position(symbol, ib_qty)

    for symbol, db_pos in db_pos_map.items():
        if symbol not in ib_positions:
            # DB thinks position is open, IB says it's closed
            logging.warning(
                f"reconcile: DB has open {symbol} but IB has no position — closing in DB"
            )
            await db.mark_position_closed(symbol)
```

This runs once at startup. If reconciliation finds
discrepancies, it logs them and corrects the DB state.
The system never enters a cycle with mismatched state.

---

## 5. Signal Aggregator Design

### 5.1 Why It's Needed

With multiple strategies, the same ticker can appear in
signals from different strategies with conflicting actions.
Without resolution, `Portfolio.run_cycle()` would place
both a BUY and a SELL for the same ticker in the same session
— wasting transaction costs and potentially netting to zero.

### 5.2 Architecture

The `SignalAggregator` sits between strategy output and
`_apply_risk_management`:

```
Strategy 1 signals ─┐
Strategy 2 signals ─┼──→ SignalAggregator → clean signal list
Strategy N signals ─┘
```

### 5.3 Implementation

```python
# Add to schemas.py

from typing import List, Dict, Any
import logging

def aggregate_signals(signals: List[Dict]) -> List[Dict]:
    """
    Resolves conflicts between signals from different strategies
    on the same ticker. Policy: opposing actions cancel both.
    Same actions from different strategies: keep highest-confidence
    (highest weight_allocation).
    """
    # Group by symbol
    by_symbol: Dict[str, List[Dict]] = {}
    for sig in signals:
        sym = sig["symbol"]
        if sym not in by_symbol:
            by_symbol[sym] = []
        by_symbol[sym].append(sig)

    result: List[Dict] = []

    for symbol, sym_signals in by_symbol.items():
        actions = {s["action"] for s in sym_signals}

        # Conflict: both BUY and SELL exist for same ticker
        if len(actions) > 1:
            logging.warning(
                f"aggregate_signals: conflict on {symbol} — "
                f"strategies disagree. Cancelling both."
            )
            continue

        # No conflict: keep the one with highest weight_allocation
        best = max(sym_signals, key=lambda s: s.get("weight_allocation", 0))
        result.append(best)

    return result
```

Add this call in `Portfolio.run_cycle()` between
`raw_signals` collection and `deduplicate_signals`:

```python
raw_signals = aggregate_signals(raw_signals)   # ← NEW
clean_signals = schemas.deduplicate_signals(raw_signals)
clean_signals = self._apply_risk_management(clean_signals)
```

### 5.4 Conflict Resolution Policies (Choose One)

| Policy | Behaviour | Best For |
|---|---|---|
| **Cancel-out** (current) | Opposing signals → discard both | Conservative, early stage |
| **Confidence-weighted** | Sum signal strengths, take net direction | When strategies have calibrated confidence scores |
| **Hierarchy** | Strategy A always overrides B | When one strategy is known to be more reliable |
| **Majority vote** | 3+ strategies needed to agree | Large strategy ensembles |

For now: cancel-out. Add confidence scores per signal when
strategies are backtested and their historical precision is
known.

---

## 6. Settings Extension Needed

```python
# Add to config.py Settings class:

max_concurrent_positions: int = 10
max_weight_per_ticker: float = 0.20
reconcile_on_startup: bool = True
bulk_db_fetch: bool = True
```
