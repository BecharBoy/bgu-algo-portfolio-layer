# README.md — my_traders

## What Is This Project?

my_traders is a systematic daily trading system built
to execute rule-based and quantitative strategies on
US equity markets via Interactive Brokers (IBKR TWS).

It is not a production-ready hedge fund system.
It is a serious engineering and research platform with
three explicit goals:

1. **Engineering:** Build production-quality infrastructure
   for algotrading (async Python, C++ compute engine,
   PostgreSQL persistence, IB broker integration)

2. **Research:** Empirically validate quantitative strategies
   on real market data, document findings, and iterate
   based on evidence — not assumptions

3. **Learning:** Develop deep understanding of the math
   behind every algorithm by implementing, testing, and
   breaking it

---

## Engineering Guidelines

These are non-negotiable rules enforced across the entire
codebase:

### 1. No Bare Exception Handling
```python
# FORBIDDEN
try:
    do_something()
except:
    pass

# REQUIRED: explicit exception types only where recovery is defined
```

Unhandled exceptions must propagate. Silent failures in
a trading system cause undetected money loss.

### 2. Strict Control Flow
Every code path must have a defined outcome. No code that
"might work" or "probably handles it." If an edge case
is not handled, it raises explicitly — not silently passed.

### 3. No Magic Numbers
All thresholds (z-score entry, min_correlation, RSI levels)
must be named constants in `config.py` — not inline literals.

### 4. Signals Execute Next Session Only
Signals generated on today's close data are never executed
same-day. This prevents lookahead bias and ensures signals
are based only on observable close prices.

### 5. DB is the System of Record
Every order, signal, fill, and position change is written
to PostgreSQL before any action is taken. No order goes to
IB without a DB record existing first.

### 6. Reconciliation Before Trading
On every startup, IB positions and DB positions are
compared. Any discrepancy is resolved and logged before
the first `run_cycle()` is called.

---

## Current Architecture

```
my_traders/
├── main.py              Entry point. Wires components, runs scheduler.
├── config.py            All settings: universe, thresholds, DB URL, IB host.
├── schemas.py           Signal/Order TypedDicts. validate_signal(). aggregate_signals().
├── Portfolio.py         Orchestrator. run_cycle(), dry_run_cycle().
├── Strategy.py          BaseStrategy abstract class.
├── DB.py                asyncpg PostgreSQL adapter. All queries here.
├── IB.py                ib_async Interactive Brokers adapter.
├── Data_Feed.py         Price fetching via IB Ticker API.
├── Wrapper.py           Python bridge to C++ cointegration engine.
├── agents.py            Scheduler / job wiring.
├── jobs.py              bootstrap_history_job and cycle job definitions.
├── tickers.csv          Ticker universe. One ticker per line. No header.
├── Makefile             Build C++ cointegration engine.
│
├── strategies/
│   ├── mean_reversion/
│   │   └── meanreversion.py     MeanReversionMomentum strategy.
│   └── cointegration/
│       ├── StatArbStrategy.py   Statistical arbitrage strategy.
│       ├── binding.cpp          pybind11 Python/C++ bridge.
│       ├── MarketData.cpp/h     Row-major price matrix. Cache-friendly.
│       ├── PairScanner.cpp/h    Multi-threaded O(N²) correlation scan.
│       ├── MathStats.cpp/h      OLS, ADF, Pearson, spread calculation.
│       └── Makefile             [see root Makefile — builds from root]
│
└── docs/                        [to be created]
    ├── 01_MATHEMATICS_IMPLEMENTED.md
    ├── 02_MATHEMATICS_KALMAN_KERNEL.md
    ├── 03_ARCHITECTURE_AND_OPTIMIZATION.md
    ├── 04_FUTURE_STRATEGIES.md
    ├── 05_ML_DL_SIGNAL_MODELS.md
    └── README.md  ← this file
```

---

## Design Decisions

### Why C++ for Math, Python for Orchestration?

C++ computes 124,750 correlations in ~4ms with 6 threads.
Python would take ~200ms and block the async event loop.
The GIL is explicitly released before C++ threads are
spawned, so async I/O (DB, IB) runs concurrently with
C++ computation.

Python is used for orchestration because: async/await for
I/O concurrency, rich ecosystem (talib, pandas, asyncpg,
ib_async), and rapid iteration on strategy logic.

### Why PostgreSQL and Not SQLite?

- asyncpg gives true async DB access (no blocking)
- PostgreSQL handles concurrent reads/writes safely
- `ON CONFLICT DO NOTHING` for idempotent fill logging
- Future: time-series extensions (TimescaleDB) for
  efficient OHLCV queries at scale

### Why Daily and Not Intraday?

Daily execution (close → next open) avoids:
- Co-location requirements for latency
- Intraday data costs (expensive)
- Continuous process uptime requirements
- Lookahead bias from same-session signals

Daily is the correct scope for strategies based on
close-price indicators (RSI, MACD, Bollinger Bands)
and statistical relationships measured over 60-day windows.

### Why IBKR TWS?

- Python SDK (ib_async) is well-documented
- Supports market orders, limit orders, futures
- Paper trading account for safe testing
- Margin accounts for shorting (required by stat arb)

---

## Build Instructions

### Prerequisites
- Python 3.11+
- C++ compiler (g++/clang++ with C++17)
- Eigen3 (`brew install eigen` / `apt install libeigen3-dev`)
- pybind11 (`pip install pybind11`)
- Python packages: `pip install -r requirements.txt`
- PostgreSQL running locally or remote

### Build C++ Engine
```bash
make build                          # uses default Eigen path
make build EIGEN_PATH=~/eigen3      # custom Eigen path
```

### Configure
```python
# config.py — set before running
DB_URL    = "postgresql://user:pass@localhost/traders"
IB_HOST   = "127.0.0.1"
IB_PORT   = 7497  # 7497=paper, 7496=live
UNIVERSE  = load_tickers("tickers.csv")
```

### Run (Dry Mode — No Orders Placed)
```bash
PYTHONPATH=. python main.py --dry-run
```

### Run (Live / Paper)
```bash
make run
```

---

## Milestone Roadmap

```
✅ M0: Core infrastructure (DB, IB, DataFeed, Portfolio)
✅ M1: MeanReversionMomentum strategy
✅ M2: StatArb C++ engine (MarketData, PairScanner, MathStats)
✅ M3: StatArbStrategy.generate_signals (z-score logic)
✅ M4: Signal aggregation + conflict resolution
✅ M5: Makefile Eigen fix + MathStats guards + PairScanner mutex fix
🔶 M6: Reconciliation on startup (IB ↔ DB consistency)
🔶 M7: Bulk DB fetch (_fetch_bars_for_all_tickers)
🔶 M8: tickers.csv populated + dry_run_cycle() passes end-to-end
🔶 M9: Backtest harness on historical data
🔶 M10: KalmanPairsStrategy implementation
🔶 M11: Reddit sentiment pipeline (praw + FinBERT)
🔶 M12: Polymarket/Kalshi overlay
🔷 M13: TFT forecasting model on full feature set
🔷 M14: RL agent for position sizing
```

---

## Research Log

All empirical findings go in `docs/FINDINGS.md`.
Format per finding:

```
## [DATE] [FINDING TITLE]
**Hypothesis:** ...
**Test:** ...
**Result:** ...
**Conclusion:** ...
**Next step:** ...
```

This creates an auditable research trail showing what
was tested, what worked, and what didn't — and why.
