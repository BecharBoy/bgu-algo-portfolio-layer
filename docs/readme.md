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
   breaking it in a real backtest environment

---

## Current Status (March 2026)

**The backtest engine is fully operational.**
A complete run from June 2024 to March 2026 on 10 tickers
with two live strategies produced:

| Metric | Result |
|---|---|
| Total Return | +1.26% |
| Max Drawdown | -8.48% |
| Win Rate | 66.7% |
| Profit Factor | 1.85 |
| Sharpe | -0.41 (needs improvement) |
| Closed Trades | 81 |

Both strategies (MeanReversionMomentum + CointegrationArb)
run in isolation with per-strategy portfolio separation.
The C++ pair scanner is integrated and functional.

---

## Architecture

```
my_traders/
├── main.py              Entry point. Wires components, runs scheduler.
├── config.py            All settings: universe, thresholds, DB URL, IB host.
├── schemas.py           Signal TypedDicts. aggregate_signals(). deduplicate_signals().
├── Portfolio.py         Live orchestrator. run_cycle(), dry_run_cycle().
├── Strategy.py          BaseStrategy abstract class.
├── DB.py                asyncpg PostgreSQL adapter. All queries here.
├── IB.py                ib_async Interactive Brokers adapter.
├── Data_Feed.py         OHLCV ingestion from IB into Postgres.
├── Wrapper.py           Python bridge to C++ cointegration engine.
├── agents.py            Background async agents.
├── jobs.py              APScheduler job definitions.
├── tickers.csv          Ticker universe. One ticker per line.
├── Makefile             Build C++ cointegration engine.
│
├── strategies/
│   ├── mean_reversion/
│   │   └── meanreversion.py     MeanReversionMomentum (BB + RSI + MACD).
│   └── cointegration/
│       ├── StatArbStrategy.py   Z-score stat arb signal generator.
│       ├── binding.cpp          pybind11 Python/C++ bridge.
│       ├── MarketData.cpp/h     Row-major price matrix. Cache-friendly.
│       ├── PairScanner.cpp/h    Multi-threaded O(N²) correlation scan.
│       ├── MathStats.cpp/h      OLS, ADF, Pearson, spread calculation.
│       └── setup.py             Builds cointegration_engine.so
│
├── backtest/
│   ├── engine.py        Multi-strategy event loop with portfolio isolation.
│   ├── portfolio.py     Per-strategy Portfolio: sizing, commission, PnL.
│   ├── data_loader.py   Pulls OHLCV from DB for any date range.
│   ├── metrics.py       Sharpe, Calmar, drawdown, win rate, profit factor.
│   ├── run_backtest.py  CLI entry point.
│   └── results/         Output CSVs and charts (git-ignored).
│
└── docs/
    ├── README.md                            ← this file
    ├── so_far.md                            ← full system architecture + math (primary reference)
    ├── SIGNAL_DISCOVERY_ALTERNATIVE_DATA.md ← Form 4, short interest, volume, PEAD, alt data
    ├── architecture_and_optimization.md     ← C++ deep dive: Eigen, BLAS, threading
    ├── PAIR_HEALTH_CHECK_AND_OU_MODEL.md    ← OU process, Kalman filter (planned upgrade)
    ├── MOMENTUM_TRAILING_STOP.md            ← Momentum + trailing stop strategy (planned)
    ├── later_approach.md                    ← Kalman, RL, TFT — long-horizon roadmap
    ├── ML_DL_for_future_predicting.md       ← GNN / DL signal models
    ├── future_strategies.md                 ← strategy pipeline and research ideas
    ├── what_can_be_predicted.md             ← predictability research foundations
    └── strategies_for_considiration.md      ← candidate strategies with academic backing
```

---

## Engineering Guidelines

Non-negotiable rules enforced across the entire codebase:

### 1. No Bare Exception Handling
```python
# FORBIDDEN
try:
    do_something()
except:
    pass
# REQUIRED: catch only specific exceptions with defined recovery
```
Unhandled exceptions must propagate. Silent failures in
a trading system cause undetected money loss.

### 2. Strict Control Flow
Every code path must have a defined outcome. No "probably
handles it." If an edge case is not handled, it raises
explicitly — never silently skipped.

### 3. No Magic Numbers
All thresholds (z-score entry, min_correlation, RSI levels)
must be named constants in `config.py` — not inline literals.

### 4. Signals Execute Next Session Only
Signals generated on today's close data execute at the
next day's open. This is enforced by the backtest engine
(`date < as_of_date` strict inequality) and is mandatory
for live trading correctness.

### 5. DB is the System of Record
Every signal, order, fill, and position change is written
to PostgreSQL before any action is taken. No order goes
to IB without a DB record existing first.

### 6. Strategy Isolation is Mandatory
Each strategy owns its positions exclusively. A strategy
must never read, modify, or close positions it did not open.
Enforced at both the engine level (per-strategy portfolios)
and at the strategy level (ownership check in generate_signals).

### 7. Reconciliation Before Trading
On every startup, IB positions and DB fills are compared.
Any discrepancy is resolved and logged before the first
`run_cycle()` is called.

---

## Design Decisions

### Why C++ for Math, Python for Orchestration?

C++ computes 124,750 correlations in ~4ms on 6 threads.
Python would take ~200ms and block the asyncio event loop
(GIL is held for CPU-bound work; IB callbacks and DB queries
would stall). C++ releases the GIL before spawning threads,
keeping asyncio free. Eigen's BLAS-backed operations reach
~90% of theoretical memory bandwidth via cache-friendly
row-major layout on the price matrix.

Python handles orchestration: async/await for I/O concurrency,
rapid strategy iteration, and access to talib, pandas,
asyncpg, and ib_async.

### Why Per-Strategy Portfolio Isolation?

Without isolation, strategies share a position namespace:
MeanReversionMomentum can accidentally close a
CointegrationArb pair leg when both hold the same symbol.
This caused catastrophic backtest bugs (one leg closed,
naked short held indefinitely). Isolation assigns each
strategy its own `Portfolio` instance with its own cash
and position dict. The engine passes `portfolio.get_positions()`
filtered per strategy to `generate_signals()`.

### Why PostgreSQL and Not SQLite?

- asyncpg provides true non-blocking DB access
- PostgreSQL handles concurrent pool connections safely
- `ON CONFLICT DO NOTHING` makes fill logging idempotent
- Positions are derived live from fills (no separate
  positions table that can desync)
- Future: TimescaleDB extension for OHLCV at scale

### Why Daily and Not Intraday?

Daily execution (close → next open) avoids co-location
requirements, intraday data costs, and continuous uptime.
It is the correct scope for strategies based on close-price
indicators (RSI, MACD, BB) and 60-day statistical windows.

---

## Build Instructions

### Prerequisites
- Python 3.11+
- C++ compiler with C++17 support (g++ or clang++)
- Eigen3 (`brew install eigen` / `apt install libeigen3-dev`)
- pybind11 (`pip install pybind11`)
- PostgreSQL running locally or remote
- `pip install -r requirements.txt`

### Build C++ Engine
```bash
# From repo root:
cd strategies/cointegration
python setup.py build_ext --inplace

# OR using Makefile:
make build                          # default Eigen path
make build EIGEN_PATH=~/eigen3      # custom Eigen path
```

### Configure
```python
# config.py
DB_URL   = "postgresql://user:pass@localhost/traders"
IB_HOST  = "127.0.0.1"
IB_PORT  = 7497    # 7497=paper, 7496=live
UNIVERSE = load_tickers("tickers.csv")
```

### Run Backtest
```bash
PYTHONPATH=. python backtest/run_backtest.py
# Output: backtest/results/equity_curve.csv
#         backtest/results/trade_log.csv
#         backtest/results/metrics.json
#         backtest/results/equity_curve.png
```

### Run Live (Dry Mode — No Orders Placed)
```bash
PYTHONPATH=. python main.py --dry-run
```

### Run Live (Paper / Live)
```bash
make run
```

---

## Milestone Roadmap

```
✅ M0:  Core infrastructure (DB, IB, DataFeed, Portfolio)
✅ M1:  MeanReversionMomentum strategy
✅ M2:  StatArb C++ engine (MarketData, PairScanner, MathStats)
✅ M3:  StatArbStrategy.generate_signals (z-score logic)
✅ M4:  Signal aggregation + conflict resolution (schemas.py)
✅ M5:  Makefile Eigen fix + MathStats guards + PairScanner mutex
✅ M6:  Backtest engine — per-strategy portfolio isolation
✅ M7:  Backtest engine — pair atomicity (can_afford_pair)
✅ M8:  Backtest engine — realistic cost model (IBKR commission + slippage)
✅ M9:  Full backtest run June 2024 → March 2026 — working equity curve
🔶 M10: ATR-based position sizing (replace flat 15% cash allocation)
🔶 M11: CointegArb exit reliability fix (debug pair_id match on exit)
🔶 M12: UNH / structural-trend filter (200d slope guard on MR longs)
🔶 M13: Sharpe improvement — max hold days cap (60d) on CointegArb
🔶 M14: Walk-forward validation (rolling 6m train / 3m test windows)
🔶 M15: Reconciliation on startup (IB ↔ DB consistency check)
🔷 M16: KalmanPairsStrategy — dynamic β (replaces static OLS)
🔷 M17: Reddit/news sentiment pipeline (praw + FinBERT)
🔷 M18: TFT forecasting model on full feature set
🔷 M19: RL agent for dynamic position sizing
```

---

## Research Log

All empirical findings go in `docs/FINDINGS.md`.
Format per entry:

```
## [DATE] [FINDING TITLE]
**Hypothesis:** ...
**Test:** ...
**Result:** ...
**Conclusion:** ...
**Next step:** ...
```

This creates an auditable research trail showing what
was tested, what worked, and what failed — and why.
