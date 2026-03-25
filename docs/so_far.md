# 01_SYSTEM_ARCHITECTURE_AND_MATH.md

## What This Document Is

A complete technical reference for the my_traders system as
it exists today. Covers: the live trading stack, the database
schema and access patterns, the backtest engine architecture,
every strategy's algorithm and the math behind it, and the
C++ performance layer. Written at senior-engineer depth.

---

## PART 1: SYSTEM OVERVIEW

### 1.1 Repository Structure

```
my_traders/
├── main.py                     # Async entry point, wires all components
├── Strategy.py                 # BaseStrategy abstract class
├── Portfolio.py                # Live portfolio manager (IB positions)
├── DB.py                       # PostgreSQL async access layer
├── IB.py                       # Interactive Brokers TWS/Gateway client
├── Data_Feed.py                # OHLCV ingestion from IB into Postgres
├── jobs.py                     # APScheduler daily job definitions
├── schemas.py                  # Signal dedup + conflict resolution
├── config.py                   # Strategy params + env config
├── API.py                      # FastAPI read endpoints
├── agents.py                   # Background async agents
├── tickers.csv                 # Universe of tracked symbols
│
├── strategies/
│   ├── cointegration/
│   │   ├── StatArbStrategy.py  # Python signal generator
│   │   ├── PairScanner.cpp     # Multi-threaded correlation scanner
│   │   ├── MathStats.cpp       # OLS, ADF, spread computation
│   │   ├── MarketData.cpp      # Price matrix preparation
│   │   ├── binding.cpp         # pybind11 Python-C++ bridge
│   │   └── setup.py            # Builds cointegration_engine.so
│   └── mean_reversion/
│       └── meanreversion.py    # MeanReversionMomentum strategy
│
└── backtest/
    ├── engine.py               # BacktestEngine: multi-strategy loop
    ├── portfolio.py            # Isolated per-strategy Portfolio
    ├── data_loader.py          # Pulls OHLCV from DB for backtest
    ├── metrics.py              # Sharpe, Calmar, drawdown, win rate
    └── run_backtest.py         # CLI entry point for backtests
```

### 1.2 The Live Execution Flow

Every trading day the following happens in order:

```
1. [07:30 ET]  jobs.py -> Data_Feed.py
               Pull yesterday's OHLCV bars from IB for all tickers.
               Upsert into PostgreSQL via DB.upsert_ohlcv_bars().

2. [08:00 ET]  jobs.py -> Portfolio.py
               Sync IB account positions into in-memory state.
               Update account_snapshots table.

3. [08:30 ET]  jobs.py -> Strategy.generate_signals()
               Each strategy independently reads market_data and
               current_positions (filtered per strategy). Returns
               a list of signal dicts.

4.             schemas.aggregate_signals() + deduplicate_signals()
               Merge signals from all strategies. Resolve conflicts
               (BUY + SELL on same symbol -> cancel both).

5.             Portfolio.py -> IB.py
               For each signal: size position, place MKT order via
               IB API, receive fill callback, log to DB.

6.             DB.save_signals(), DB.save_orders(), DB.log_trade_execution()
               Persist everything to Postgres for audit trail and
               future analysis.
```

---

## PART 2: DATABASE LAYER (DB.py)

### 2.1 Schema

The schema is created and migrated idempotently via
`DB.init_schema()` on every startup. No manual migration
scripts — just re-run and it's safe.

```sql
-- Price history (the core data store)
CREATE TABLE ohlcv_bars (
    ticker  TEXT   NOT NULL,
    date    DATE   NOT NULL,
    open    FLOAT  NOT NULL,
    high    FLOAT  NOT NULL,
    low     FLOAT  NOT NULL,
    close   FLOAT  NOT NULL,
    volume  BIGINT NOT NULL,
    PRIMARY KEY (ticker, date)   -- prevents duplicate bars
);

-- Every signal generated (audit trail)
CREATE TABLE signals (
    signal_id       TEXT PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL,
    strategy        TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    action          TEXT NOT NULL,   -- BUY | SELL
    price_reference FLOAT NOT NULL,
    reason          TEXT,
    metadata        JSONB            -- hedge_ratio, zscore, pair_id, etc.
);

-- Orders sent to IB
CREATE TABLE orders (
    order_id   TEXT PRIMARY KEY,
    signal_id  TEXT REFERENCES signals(signal_id),
    strategy   TEXT NOT NULL,
    symbol     TEXT NOT NULL,
    action     TEXT NOT NULL,
    quantity   INT  NOT NULL,
    order_type TEXT NOT NULL,   -- MKT | LMT
    tif        TEXT NOT NULL,   -- DAY | GTC
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Actual fills from IB
CREATE TABLE fills (
    fill_id    TEXT PRIMARY KEY,
    order_id   TEXT REFERENCES orders(order_id),
    symbol     TEXT NOT NULL,
    action     TEXT NOT NULL,
    quantity   INT  NOT NULL,
    fill_price FLOAT NOT NULL,
    filled_at  TIMESTAMPTZ NOT NULL,
    strategy   TEXT NOT NULL DEFAULT '',   -- owning strategy
    pair_id    TEXT                        -- for CointegArb pairs
);

-- Daily account snapshots
CREATE TABLE account_snapshots (
    snapshot_id SERIAL PRIMARY KEY,
    nlv         FLOAT NOT NULL,   -- net liquidation value
    cash        FLOAT NOT NULL,
    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Persistent system flags
CREATE TABLE system_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

### 2.2 Key Design Decisions

**asyncpg with a connection pool (min=2, max=10):**
All DB operations are async and use `pool.acquire()` as a
context manager. This means the asyncio event loop is never
blocked by a DB call. Multiple strategies can query
simultaneously on different pool connections.

**ON CONFLICT DO NOTHING on inserts:**
Signals, orders, and fills are idempotent. If the system
restarts mid-day, re-running signal generation won't
create duplicate rows. The primary key (signal_id, order_id,
fill_id) is a UUID generated at creation time.

**Open positions derived from fills, not stored directly:**
`get_open_positions_from_db()` computes positions on-the-fly:

```sql
SELECT symbol, strategy, pair_id,
       SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) AS net_quantity
FROM fills
GROUP BY symbol, strategy, pair_id
HAVING SUM(...) != 0
```

This means the position table is always derived from the
fills history. No separate positions table that can
desync. The cost: one aggregation query per position check.
Acceptable at daily frequency.

**strategy + pair_id on fills — ownership isolation:**
These two columns (added via idempotent ALTER IF NOT EXISTS
migration) allow each strategy to query only its own fills.
`StatArbStrategy` can never see `MeanReversionMomentum`
fills, and vice versa. This is enforced at the DB query
level, not just application level.

**OHLCV upsert pattern:**
`upsert_ohlcv_bars()` uses:
```sql
INSERT INTO ohlcv_bars ... ON CONFLICT (ticker, date)
DO UPDATE SET open=EXCLUDED.open, ...
```
So if IB sends a corrected bar (adjusted prices, split),
the latest value always wins.

### 2.3 system_state Table

Used for persistent flags that survive restarts:
- `trading_enabled`: "true"/"false" kill-switch
- `last_scan_date`: last date the pair scanner ran
- Any ad-hoc operational flags

```python
await db.set_system_flag("trading_enabled", "false")  # emergency stop
flag = await db.get_system_flag("trading_enabled")
```

---

## PART 3: BACKTEST ENGINE

### 3.1 Architecture Overview

The backtest system is a complete offline simulation of the
live execution flow. It reuses the same Strategy classes
unchanged — there is no "backtest mode" in the strategies
themselves. The engine simulates the environment the
strategies run in.

```
BacktestEngine
├── Receives: list[BaseStrategy], dict[str, Portfolio], dict[str, DataFrame]
├── For each trading date in [start_date, end_date]:
│   ├── _build_window(): slice last `lookback` bars BEFORE today
│   ├── strategy.generate_signals(window, own_positions)
│   ├── schemas.deduplicate_signals(aggregate_signals(raw))
│   ├── _group_by_pair() -> pair_groups, singles
│   ├── For pair_groups: portfolio.can_afford_pair() -> execute atomically
│   ├── For singles: portfolio.size_signal() -> execute_signal()
│   └── Record total equity across all portfolios
└── Returns: DataFrame[date, equity, cash, num_positions]
```

### 3.2 Per-Strategy Portfolio Isolation

This is the most important architectural decision in the
backtest engine. Each strategy gets its own `Portfolio`
instance seeded with a fraction of total capital:

```python
portfolios = {
    "MeanReversionMomentum": Portfolio(50_000, "MeanReversionMomentum"),
    "CointegrationArb":      Portfolio(50_000, "CointegrationArb"),
}
```

**Why this matters:**
Before this architecture, a single shared portfolio meant:
- MR strategy could accidentally close StatArb positions
- StatArb pair exits would trigger MR's "we don't own this" error
- Cash was shared, so one strategy depleting cash blocked the other

With isolated portfolios, each strategy sees only its own
positions via `portfolio.get_positions()`. The engine passes
`own_positions` to `strategy.generate_signals()`, which only
contains that strategy's holdings.

### 3.3 Lookahead Bias Prevention

`_build_window()` is critical for correctness:

```python
def _build_window(self, as_of_date):
    for ticker, frame in self.data.items():
        history = frame[frame["date"] < as_of_date].tail(self.lookback)
```

The strict `< as_of_date` (not `<=`) means the strategy
never sees today's close when generating today's signals.
Execution happens at `open_prices` of the signal date —
the earliest price available after the signal is generated.
This is the correct simulation of reality: you see
yesterday's close, generate a signal at night, and execute
at next day's open.

### 3.4 Pair Atomicity

CointegrationArb pairs must enter both legs or neither.
Entering only one leg creates an unhedged directional bet.

`_group_by_pair()` splits signals by `metadata["pair_id"]`
for `CointegrationArb` signals. Then `can_afford_pair()`
dry-runs the cash deduction for all new entry legs:

```python
def can_afford_pair(self, legs, prices, allocation=0.15):
    simulated_cash = self.cash
    for leg in legs:
        if leg.get("is_exit"): continue   # exits add cash, ignore
        fill_price = self._effective_price(leg["action"], prices[leg["symbol"]])
        qty = max(int(simulated_cash * allocation / fill_price), 0)
        if qty <= 0: return False
        cost = qty * fill_price + self._commission_paid(qty, fill_price)
        simulated_cash -= cost
        if simulated_cash < 0: return False
    return True
```

If either leg cannot be funded, the entire pair is dropped.
This prevents the common backtest bug of executing one leg
and carrying a naked short/long.

### 3.5 Cost Model

The Portfolio simulates realistic execution costs:

**Slippage:** 0.05% per side (configurable).
```python
fill_price = price * (1 + slippage)  # BUY pays more
fill_price = price * (1 - slippage)  # SELL receives less
```

**Commission (IBKR Pro schedule):**
```python
_IBKR_RATE    = $0.005 per share
_IBKR_MIN     = $1.00 minimum per order
_IBKR_MAX_PCT = 1.0% of trade notional (cap)

commission = min(max(shares * 0.005, 1.00), notional * 0.01)
```

**Realized PnL calculation on close:**
```python
realized_pnl = (exit_price - avg_entry) * shares
             - exit_commission
             - proportional_entry_commission
```
The entry commission is prorated proportionally if only
part of a position is closed.

### 3.6 Metrics (backtest/metrics.py)

After `engine.run()` returns the equity DataFrame, metrics
are computed:

| Metric | Formula |
|---|---|
| Total Return | (final_equity / initial) - 1 |
| Ann. Return | total_return * (252 / trading_days) |
| Ann. Volatility | daily_returns.std() * sqrt(252) |
| Sharpe | ann_return / ann_volatility |
| Max Drawdown | min((equity - running_max) / running_max) |
| Calmar | ann_return / abs(max_drawdown) |
| Win Rate | profitable_trades / closed_trades |
| Profit Factor | gross_profit / abs(gross_loss) |
| Avg Hold Days | mean(hold_days) across closed trades |

---

## PART 4: COINTEGRATION STRATEGY (StatArbStrategy + C++ Engine)

### 4.1 The Full Pipeline

```
market_data dict
      │
      ▼
[C++ cointegration_engine.run_cpp_scan()]
      │  ├── PairScanner: Pearson correlation filter (>= 0.85)
      │  ├── MathStats::calculate_OLS: hedge ratio β
      │  ├── MathStats::calculate_spread: residual series
      │  └── MathStats::calculate_adf_statistic: stationarity test (τ < -3.0)
      │
      ▼ list[CointegPair(stock_x, stock_y, hedge_ratio, adf_stat)]
      │
[StatArbStrategy.generate_signals()]
      │  ├── Compute spread: Yₜ - β·Xₜ
      │  ├── Compute z-score: (spread - μ) / σ
      │  ├── If pair is open:
      │  │     validate pair_id ownership (canonical = sorted tickers)
      │  │     if |z| < 0.5: generate exit signals (is_exit=True)
      │  └── If pair is closed:
      │        if z > +2.0: SELL Y, BUY X
      │        if z < -2.0: BUY Y, SELL X
      ▼
list[signal_dict] with pair_id, hedge_ratio, zscore in metadata
```

### 4.2 Math: Pearson Correlation

Given price series A = [a₁..aₙ], B = [b₁..bₙ]:

```
         Σᵢ (aᵢ - ā)(bᵢ - b̄)
ρ = ─────────────────────────────────
    √[Σᵢ(aᵢ-ā)²] · √[Σᵢ(bᵢ-b̄)²]
```

Bounded [-1, 1] by Cauchy-Schwarz.
Threshold: **ρ ≥ 0.85** — empirically ~61% of these
pass the ADF test, vs ~8% for ρ < 0.70.

**Why C++:** At N=500 tickers, we need N(N-1)/2 = 124,750
dot products over 60-day windows (~7.5M float ops).
Python (NumPy, single-threaded): ~200ms, which blocks the
asyncio event loop. C++ with Eigen + 6 threads: ~4ms,
freeing asyncio for IB callbacks and DB queries.

### 4.3 Math: OLS Hedge Ratio

Minimize Σ(Yᵢ - α - βXᵢ)². Closed-form solution:

```
β = Cov(X,Y) / Var(X) = Σ(Xᵢ-x̄)(Yᵢ-ȳ) / Σ(Xᵢ-x̄)²
α = ȳ - β·x̄
```

β is the **hedge ratio**: hold 1 share of Y and β shares of
X (opposite direction) to construct a dollar-neutral spread.
BLUE (best linear unbiased estimator) under Gauss-Markov.

**Limitation:** β is static over the lookback window.
If the relationship changes mid-window (earnings, sector
rotation), β is stale until the next re-fit. Kalman Filter
is the correct upgrade (see docs/later_approach.md).

### 4.4 Math: ADF Test

Fit regression: Δsₜ = γ·sₜ₋₁ + c + ηₜ

Test statistic: τ = γ̂ / SE(γ̂)

If γ < 0 and |τ| is large: there is a mean-reverting force.
The spread is stationary — safe to trade.

| τ threshold | Significance |
|---|---|
| < -3.43 | 1% — very strong |
| < -3.00 | ~2-3% — our threshold |
| < -2.86 | 5% — standard academic |

**Critical:** τ follows Dickey-Fuller distribution, NOT the
standard t-distribution. The DF distribution has heavier
left tails, requiring more negative τ to reject H₀.

### 4.5 Math: Z-Score Signal

```
spread_t  = Y_t - β·X_t
z_t       = (spread_t - μ_spread) / σ_spread
```

| z | Action | Logic |
|---|---|---|
| z > +2.0 | SELL Y, BUY X | Y overpriced vs X |
| z < -2.0 | BUY Y, SELL X | Y underpriced vs X |
| \|z\| < 0.5 | Close position | Spread reverted to mean |
| else | Hold | Within normal range |

±2.0 = 95th/5th percentile of a normal distribution.
~12 signal events per pair per year.

### 4.6 Canonical Pair ID

The pair_id stored at entry uses `_canonical_pair_id()`:
```python
def _canonical_pair_id(a: str, b: str) -> str:
    return "_".join(sorted([a, b]))
# BAC_MSFT and MSFT_BAC both resolve to "BAC_MSFT"
```

This ensures exit signals always match the pair_id stored
at entry, regardless of the scanner's iteration order on
any given day. Without this, a scanner that returns
`(MSFT, BAC)` on exit day would generate a different pair_id
than `(BAC, MSFT)` on entry day — silently holding the
position forever.

---

## PART 5: MEAN REVERSION STRATEGY (MeanReversionMomentum)

### 5.1 Entry Conditions (ALL must be true)

```
Price < Lower Bollinger Band (30-day SMA ± 2σ)
AND
(RSI(14) < 35   OR   MACD(24,52,18) bullish crossover)
```

This is a **confirming filter**: the price must be
statistically cheap (BB) AND momentum must be turning
(RSI oversold or MACD flip). Either momentum condition
alone — without the BB condition — generates no signal.

### 5.2 Exit Conditions (ANY is sufficient)

```
RSI(14) > 70          (overbought)
OR
Price > Upper BB      (statistically expensive)
OR
MACD bearish crossover (momentum turning negative)
```

Exits are not symmetric with entries. The strategy enters
with a price + momentum filter but exits on any one
overbought signal. This creates an asymmetric hold:
entries are selective, exits are responsive.

### 5.3 Math: Bollinger Bands (N=30)

```
SMA_t   = (1/30) Σᵢ₌ₜ₋₂₉ᵗ Closeᵢ
σ_t     = √[(1/30) Σᵢ(Closeᵢ - SMA_t)²]
Upper_t = SMA_t + 2·σ_t
Lower_t = SMA_t - 2·σ_t
```

N=30 chosen over standard N=20: smoother bands reduce
false signals from daily noise while remaining responsive
enough for 5-15 day holding periods. N=30 maximizes
directional accuracy on S&P 500 daily data per Lento et al.

### 5.4 Math: RSI (N=14)

```
RS    = Avg_Gain(14) / Avg_Loss(14)
RSI   = 100 - 100 / (1 + RS)
```

RSI < 35: oversold entry threshold (stricter than standard
30 to reduce false positives in trending markets).
RSI > 70: standard overbought exit.

### 5.5 Math: MACD (24, 52, 18)

```
EMA(N)_t    = α·Close_t + (1-α)·EMA(N)_{t-1},  α = 2/(N+1)
MACD_Line   = EMA(24) - EMA(52)
Signal_Line = EMA(18, MACD_Line)
```

Configuration (24, 52, 18) vs standard (12, 26, 9):
doubled periods filter out signals that would reverse
within 2 weeks. Keeps only moves with momentum sufficient
to remain profitable after transaction costs on a daily
swing strategy.

### 5.6 Strategy Isolation from CointegArb

```python
if ticker in current_positions:
    pos = current_positions[ticker]
    if pos.strategy != self.name:
        continue   # not ours — skip unconditionally
```

The engine passes only MR-owned positions to
`generate_signals()`. But a second guard exists: if somehow
a foreign position appears (e.g., engine bug), the strategy
explicitly checks the `strategy` attribute and skips.
Defense in depth.

---

## PART 6: SIGNAL PIPELINE (schemas.py)

### 6.1 Signal Format

Every signal is a dict:
```python
{
    "signal_id":         str(uuid4()),
    "timestamp":         ISO 8601 UTC,
    "symbol":            "AAPL",
    "action":            "BUY" | "SELL",
    "strategy":          "MeanReversionMomentum" | "CointegrationArb",
    "weight_allocation": 0.15,
    "price_reference":   float,        # last Close at signal time
    "reason":            str,          # human-readable tag
    "metadata":          dict | None,  # pair_id, zscore, hedge_ratio, etc.
    "is_exit":           bool,
}
```

### 6.2 Conflict Resolution (aggregate_signals)

When two strategies generate signals for the same symbol:
- BUY from MR + SELL from StatArb → conflict → both dropped
- BUY from both → keep one (deduplicate by symbol+action)
- SELL + SELL → keep one

This prevents contradictory orders from reaching IB and
ensures the portfolio never receives simultaneous opposing
fills on the same symbol.

### 6.3 Deduplication

`deduplicate_signals()` removes duplicate (symbol, action)
pairs. The first signal in the list wins (ordered by
strategy priority if needed). Idempotent: safe to call
multiple times.

---

## PART 7: OPEN ISSUES AND NEXT STEPS

### 7.1 Sharpe Still Negative in Backtest

Current Sharpe: -0.41 despite +1.26% total return.
Root cause: 80-day average hold means positions bleed
unrealized PnL daily against the equity curve. Fix:
reduce `min_correlation` slightly (0.80) to increase
pair turnover, or add a max-hold-days exit rule (60 days).

### 7.2 CointegArb Exits Not Firing Reliably

Hypothesis: `_canonical_pair_id()` comparison is correct,
but the exit z-score threshold (0.5) is too tight for
pairs in a trending regime. The spread never crosses back
below 0.5 if the underlying relationship drifted.
Debug step: log `(stored_pair_id, scanned_pair_id, zscore)`
in `_generate_exit_signals()` every bar to confirm match.

### 7.3 ATR-Based Position Sizing

Currently sizing is a flat 15% of available cash per trade.
The correct upgrade is ATR-normalized sizing:
```python
quantity = risk_per_trade_dollars / ATR(14)
```
This equalizes dollar-risk across all positions regardless
of individual stock volatility. A $5 ATR stock vs $20 ATR
stock should carry different share counts for equal risk.

### 7.4 Kalman Filter for Dynamic β

OLS computes a static hedge ratio over the lookback window.
When the economic relationship between two stocks shifts
(sector rotation, earnings shock), β is stale for up to
60 days. Kalman Filter tracks β as a time-varying state
variable updated daily. See docs/later_approach.md.

### 7.5 UNH — Structural Loser

6 trades, all losers, total -$13.5k in backtest.
UNH is in structural regulatory decline —  a mean-reversion
strategy fighting a persistent downtrend. Fix: add a 200-day
trend filter. Only enter MR long if slope(Close, 200d) > 0.
If the long-term trend is down, MR longs are fighting physics.
