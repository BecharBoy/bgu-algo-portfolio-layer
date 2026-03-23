# 06_BACKTEST_ENGINE.md

## What This Document Is

A full specification for the Python backtesting engine that
will validate every strategy in my_traders before it touches
real capital. This document explains what every file must
contain, why each design decision was made, and the strict
rules every contributor must follow. Written for someone
with zero background.

---

## Why We Backtest

A backtest simulates what would have happened if you had
run your strategy on historical data. It answers: "Before
I risk real money, does this strategy actually make money
over time?" Without backtesting, you are gambling.

**The cardinal rule of backtesting:**
The engine must NEVER let the strategy see data from the
future. If today is March 15th and you are simulating
March 15th, the strategy can only see data up to and
including March 14th close. Any violation of this rule
produces results that look good historically but fail
immediately in live trading. This is called **lookahead
bias** and it is the most common and most fatal mistake
in algorithmic trading research.

---

## Why Python (Not C++)

Our strategies (MeanReversionMomentum, StatArbStrategy)
are written in Python. They produce signals via
`generate_signals(market_data, current_positions)`.

If the backtest engine were in C++, we would have to
rewrite every strategy in C++ just to test it. That means:
- Two codebases to maintain
- Risk of subtle differences between the Python live
  version and the C++ test version
- Bugs that exist in one but not the other

In Python, the backtest feeds the EXACT same strategy
class the exact same interface it receives in live trading.
We test what we ship.

**C++ is still used** for the pair-scanning step inside
the backtest (via the existing pybind11 binding). The
Python loop calls `scanner.scan_all_pairs()` exactly as
the live system does. C++ handles CPU-heavy computation;
Python handles the orchestration and analytics.

For daily strategies, a full 3-year backtest on 200 tickers
runs in under 15 seconds in Python. There is no performance
case for a C++ event loop at this frequency.

---

## Project Structure

```
backtest/
├── engine.py          ← core event loop, bar-by-bar simulation
├── portfolio.py       ← position tracking, P&L, commission model
├── data_loader.py     ← load from PostgreSQL OR CSV
├── metrics.py         ← Sharpe, max drawdown, win rate, Calmar
├── run_backtest.py    ← entry point: configure and launch
└── results/           ← output CSVs and charts (git-ignored)
```

---

## FILE 1: `data_loader.py`

### Purpose

Feeds historical OHLCV data to the engine. Must support
two modes:
1. **PostgreSQL** — pull from the same DB used in live
   trading, ensuring data consistency
2. **CSV** — for offline testing or when DB is unavailable

The loader is the ONLY file in the backtest that touches
data storage. All other files receive clean DataFrames.

### Interface

```python
async def load_from_db(
    tickers: list[str],
    start_date: str,          # "YYYY-MM-DD"
    end_date: str,
    lookback_days: int = 60,  # warm-up bars before start_date
) -> dict[str, pd.DataFrame]:
    """
    Returns {ticker: DataFrame} where each DataFrame has
    columns: date, open, high, low, close, volume.
    Sorted ascending by date.
    Includes lookback_days extra bars before start_date
    so indicators are warm on the first tradeable bar.
    """

def load_from_csv(
    path: str,
    ticker: str,
) -> dict[str, pd.DataFrame]:
    """
    Single-ticker CSV loader for offline testing.
    CSV must have header: date,open,high,low,close,volume
    """
```

### Rules

1. **Always include warm-up bars.** If the simulation
   starts on Jan 1 2024, load data from ~Nov 1 2023
   (60 extra bars). This ensures RSI(14), SMA(30),
   MACD(52) are fully calculated by the first trade date.
   Without warm-up bars, the first 52 bars of the
   simulation have NaN indicators and generate no signals
   — silently losing 2 months of potential trades.

2. **Validate column presence.** After loading, assert
   that `date, open, high, low, close, volume` all exist.
   Raise a clear error if not — never silently proceed
   with missing columns.

3. **No forward fill on close prices.** If a ticker has
   missing days (halted stock, delisting), do not fill
   forward. Drop those tickers or raise a warning. Forward
   filling creates phantom liquidity and corrupts P&L.

4. **Same DB, same data.** When loading from PostgreSQL,
   use the EXACT same `get_recent_bars_bulk` query used in
   live trading. Data used in backtesting must be
   identical to data received in live trading. Any
   discrepancy (different timezone, different adjustment
   factor) invalidates the backtest.

---

## FILE 2: `portfolio.py`

### Purpose

Tracks all positions, cash, and P&L throughout the
simulation. Executes signals at realistic prices with
commission and slippage. Is the single source of truth
for what the portfolio holds at any point in time.

### What "Realistic Execution" Means

In a real backtest, you do not buy at exactly the close
price you used to generate the signal. By the time you
act, the price has moved. To model this:

- **Execution price** = next bar's open (not current
  bar's close). You see the signal at close, you execute
  at next open. This is the standard for daily strategies.
- **Commission** = 0.1% per side (0.2% round-trip).
  This is consistent with IBKR tiered pricing for
  moderate-sized accounts.
- **Slippage** = 0.05% per side. Models the bid-ask
  spread on liquid US equities. For illiquid stocks,
  increase this.

### Interface

```python
class Portfolio:
    def __init__(
        self,
        initial_cash: float,
        commission: float = 0.001,   # 0.1% per side
        slippage: float  = 0.0005,   # 0.05% per side
    ): ...

    def execute_signal(
        self,
        signal: dict,
        execution_price: float,      # next bar open
        date: str,
    ) -> None:
        """
        Executes one signal. Applies commission and slippage.
        Logs the trade to internal trade_log list.
        Rejects the trade if:
          - BUY: insufficient cash
          - SELL: ticker not in positions
        Never raises — logs a warning and skips.
        """

    def get_equity(self, current_prices: dict[str, float]) -> float:
        """
        Total portfolio value = cash + sum(position_value).
        Called once per bar to record the equity curve.
        """

    def get_positions(self) -> dict[str, dict]:
        """
        Returns current open positions.
        Format: {ticker: {quantity, entry_price, entry_date}}
        """

    def get_trade_log(self) -> list[dict]:
        """
        Full history of every executed trade.
        Each entry: date, symbol, action, quantity,
                    price, commission, slippage, pnl.
        """
```

### Position Sizing Rule

Do NOT use all-in sizing (what the C++ prototype does).
All-in sizing is how you blow up an account.

Use **fixed fractional sizing**: each signal is allocated
`weight_allocation` percent of current portfolio equity.

```python
position_value = portfolio_equity * signal["weight_allocation"]
quantity = int(position_value / execution_price)
```

`weight_allocation` comes directly from the signal dict
(already set in each strategy's `_build_signal`).
MeanReversionMomentum defaults to 0.5 (50% of equity
per position). For a multi-strategy portfolio, this
must be reduced — if 3 strategies each send a signal
simultaneously, 3 × 50% = 150% of equity, which is
impossible without leverage.

**Cap total exposure:** `sum(weight_allocation) ≤ 0.95`
at all times. If adding a new signal would breach this,
reject it.

### Rules

1. **Never modify historical data.** Portfolio state
   changes (cash, positions) must never touch the input
   DataFrames. Work on internal state only.

2. **Log every rejection.** If a trade is rejected
   (insufficient cash, no position to sell), log it with
   reason. Silent rejections hide bugs.

3. **Pairs positions are one unit.** For StatArbStrategy,
   a long/short pair (BUY stock A, SELL stock B) is one
   logical position. Track them together under a
   `pair_id` key so they can be closed together.

---

## FILE 3: `engine.py`

### Purpose

The core simulation loop. Iterates through trading dates
bar by bar. At each bar: feeds data to the strategy,
receives signals, passes signals to Portfolio for
execution, records equity. This is the heart of the
backtest.

### The Bar-by-Bar Loop (Explained Simply)

Imagine you are replaying history one day at a time.
On each day:
1. You look at all the price data UP TO AND INCLUDING
   yesterday's close (never today's)
2. You give that data to your strategy
3. Your strategy says "buy AAPL" or "sell MSFT" or
   "do nothing"
4. You execute those orders at TODAY's open price
   (because you decided at close, you act at next open)
5. You record how much your portfolio is worth at
   today's close
6. Move to the next day

This loop runs for every trading day in your backtest
period. At the end, you have a full equity curve.

### Interface

```python
class BacktestEngine:
    def __init__(
        self,
        strategy: BaseStrategy,
        portfolio: Portfolio,
        data: dict[str, pd.DataFrame],   # from data_loader
        start_date: str,
        end_date: str,
    ): ...

    async def run(self) -> pd.DataFrame:
        """
        Executes the full simulation.
        Returns equity_curve DataFrame:
          columns: date, equity, cash, num_positions
        """
```

### The Loop Implementation Pattern

```python
async def run(self) -> pd.DataFrame:
    equity_curve = []
    trading_dates = self._get_trading_dates()

    for date in trading_dates:
        # STRICT: only feed data strictly before this date
        # to prevent lookahead bias
        window = {
            ticker: df[df["date"] < date].tail(self.lookback)
            for ticker, df in self.data.items()
            if len(df[df["date"] < date]) >= self.min_bars
        }

        if not window:
            continue

        # Get current prices for execution (today's open)
        execution_prices = {
            ticker: df[df["date"] == date]["open"].iloc
            for ticker, df in self.data.items()
            if not df[df["date"] == date].empty
        }

        # Run strategy on historical window (no future data)
        signals = await self.strategy.generate_signals(
            window, self.portfolio.get_positions()
        )

        # Execute at today's open (not yesterday's close)
        for signal in signals:
            price = execution_prices.get(signal["symbol"])
            if price:
                self.portfolio.execute_signal(signal, price, date)

        # Record equity at today's close
        close_prices = {
            ticker: df[df["date"] == date]["close"].iloc
            for ticker, df in self.data.items()
            if not df[df["date"] == date].empty
        }
        equity_curve.append({
            "date":          date,
            "equity":        self.portfolio.get_equity(close_prices),
            "cash":          self.portfolio.get_positions(),
            "num_positions": len(self.portfolio.get_positions()),
        })

    return pd.DataFrame(equity_curve)
```

### Rules

1. **The lookahead guard is non-negotiable.**
   `df[df["date"] < date]` — strictly less than. Never
   `<=`. One character difference, catastrophic effect.

2. **Execute at next open, not current close.**
   Signal is generated on bar N close. Execution happens
   at bar N+1 open. This is mandatory for realistic
   simulation of daily strategies.

3. **Skip bars with insufficient data.**
   If a ticker has fewer than `min_bars` (e.g. 52, the
   MACD warmup requirement) of history at this date,
   exclude it from the window. Never pass a ticker with
   NaN indicators to the strategy.

4. **The strategy receives the SAME interface as live.**
   `generate_signals(window, current_positions)` — exact
   same signature as in live `Portfolio.run_cycle()`.
   Any deviation means the backtest is testing different
   code than what runs live.

---

## FILE 4: `metrics.py`

### Purpose

Takes the equity curve output by the engine and computes
every performance metric needed to evaluate a strategy.
The goal is not to make the strategy look good — it is to
understand its true risk-adjusted performance.

### Required Metrics

#### Sharpe Ratio
Measures return per unit of risk. The single most
important metric for comparing strategies.

```
Sharpe = (mean(daily_returns) - risk_free_rate) /
          std(daily_returns)
       × √252    (annualized)
```

Risk-free rate: use 5.0% annual / 252 = 0.000198 daily
(approximate US T-bill rate in 2024-2025).

**Interpretation:**
- Sharpe < 0.5: not worth trading
- Sharpe 0.5–1.0: acceptable, marginal
- Sharpe 1.0–2.0: good
- Sharpe > 2.0: excellent (suspect if too high — check
  for lookahead bias)

#### Maximum Drawdown
The largest peak-to-trough decline in portfolio value.
The most important risk metric.

```
drawdown_t = (equity_t - peak_t) / peak_t
max_drawdown = min(drawdown_t) for all t
```

Where `peak_t = max(equity_0, ..., equity_t)`.

**Interpretation:**
- Max drawdown > 30%: dangerous for a daily strategy
- Max drawdown < 15%: acceptable
- Max drawdown < 10%: very well-controlled risk

#### Calmar Ratio
Annual return divided by max drawdown. Measures how
much return you get per unit of drawdown risk.

```
Calmar = annualized_return / abs(max_drawdown)
```

Calmar > 1.0 is the minimum acceptable threshold.

#### Win Rate
Percentage of closed trades that were profitable.

```
win_rate = profitable_trades / total_closed_trades
```

A strategy can be profitable with a 40% win rate if
average winners are 3x average losers. Never evaluate
win rate in isolation.

#### Profit Factor
Total gross profit divided by total gross loss.

```
profit_factor = sum(winning_trade_pnl) /
                abs(sum(losing_trade_pnl))
```

Profit factor > 1.5 is the minimum for a viable strategy.

### Interface

```python
def compute_metrics(
    equity_curve: pd.DataFrame,
    trade_log: list[dict],
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Returns all metrics as a single dict.
    Keys: sharpe, max_drawdown, calmar, win_rate,
          profit_factor, total_return, annualized_return,
          num_trades, avg_hold_days.
    """

def plot_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: str = "results/equity_curve.png",
) -> None:
    """
    Saves equity curve chart with drawdown subplot.
    Uses matplotlib. Saves PNG to results/.
    """
```

### Rules

1. **Never cherry-pick the date range.** Compute metrics
   over the full declared backtest period. Do not start
   the metrics from the first profitable month.

2. **Report drawdown, always.** A strategy that shows
   only Sharpe ratio without max drawdown is hiding
   its worst characteristic. Both must always be
   reported together.

3. **Separate in-sample from out-of-sample.**
   If you tuned any parameters (RSI threshold, BB window,
   correlation cutoff) on the data, those parameters
   are in-sample. Report metrics separately on a held-out
   period you never touched during development.

---

## FILE 5: `run_backtest.py`

### Purpose

The single entry point. Configure everything here and
run. No logic lives here — it only wires together the
other components and prints/saves results.

### Template

```python
import asyncio
from data_loader import load_from_db
from engine import BacktestEngine
from portfolio import Portfolio
from metrics import compute_metrics, plot_equity_curve
from strategies.mean_reversion.meanreversion import MeanReversionMomentum

# ── Configuration ──────────────────────────────────────
TICKERS     = ["AAPL", "MSFT", "GOOGL", "NVDA", "META"]
START_DATE  = "2022-01-01"
END_DATE    = "2024-12-31"
INITIAL_CASH = 100_000.0
COMMISSION   = 0.001    # 0.1% per side
SLIPPAGE     = 0.0005   # 0.05% per side
# ───────────────────────────────────────────────────────

async def main():
    # 1. Load data (includes 60-bar warmup before START_DATE)
    data = await load_from_db(
        tickers=TICKERS,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback_days=60,
    )

    # 2. Initialize components
    strategy  = MeanReversionMomentum(weight_allocation=0.3)
    portfolio = Portfolio(INITIAL_CASH, COMMISSION, SLIPPAGE)

    # 3. Run simulation
    engine = BacktestEngine(
        strategy=strategy,
        portfolio=portfolio,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
    )
    equity_curve = await engine.run()

    # 4. Compute and print metrics
    metrics = compute_metrics(equity_curve, portfolio.get_trade_log())
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")

    # 5. Save results
    equity_curve.to_csv("results/equity_curve.csv", index=False)
    import pandas as pd
    pd.DataFrame(portfolio.get_trade_log()).to_csv(
        "results/trade_log.csv", index=False
    )
    plot_equity_curve(equity_curve, "results/equity_curve.png")
    print("Results saved to results/")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Walk-Forward Validation (Phase 2)

A single backtest over the full period is called
**in-sample testing**. If you pick parameters that work
well on this specific 3-year window, they may not work
on the next 3 years. This is overfitting.

**Walk-forward validation** is the correct method:

```
Year 1 (train: fit params) → Year 2 (test: trade signals)
Year 2 (train: fit params) → Year 3 (test: trade signals)
Year 3 (train: fit params) → Year 4 (test: trade signals)
```

At each step:
1. Fit parameters (correlation threshold, BB window,
   RSI threshold) on the TRAIN window only
2. Run strategy with those parameters on the TEST window
3. Record TEST window performance
4. Roll forward by 3 months
5. Repeat

The final reported performance is the concatenation of
all TEST windows — none of which were used for fitting.
This is the only academically valid backtest methodology.

```python
# Walk-forward loop concept (Phase 2)
for train_start, train_end, test_start, test_end in windows:
    params = optimize_params(data, train_start, train_end)
    strategy = MeanReversionMomentum(**params)
    result   = await engine.run(strategy, test_start, test_end)
    results.append(result)
```

---

## The `results/` Directory

All output files go here. This directory is git-ignored
(never commit raw backtest results — they are derived
artifacts, not source code).

Expected output files after a full run:

| File | Contents |
|---|---|
| `equity_curve.csv` | date, equity, cash, num_positions |
| `trade_log.csv` | Every trade: date, symbol, action, qty, price, pnl |
| `metrics.json` | All computed metrics as JSON |
| `equity_curve.png` | Equity curve + drawdown chart |

---

## Code Quality Rules (Same as my_traders)

These rules apply to every file in `backtest/` without
exception:

1. **Type hints on every function signature.** No
   untyped functions. Use `dict[str, pd.DataFrame]`,
   not just `dict`.

2. **No silent failures.** Every data validation failure
   must log a warning with the ticker, date, and reason.
   Never pass silently over bad data.

3. **No magic numbers.** Commission (0.001), slippage
   (0.0005), min_bars (52) must be named constants or
   constructor parameters. Never hardcode them inline.

4. **Single responsibility.** Each file does one thing.
   `engine.py` runs the loop. `metrics.py` computes
   metrics. `portfolio.py` tracks money. No function
   does two jobs.

5. **The strategy interface is frozen.** The engine
   calls `await strategy.generate_signals(window,
   positions)`. This signature must never change. It is
   the contract between the backtest and the live system.

6. **async throughout.** `engine.run()` and
   `data_loader.load_from_db()` are async. `run_backtest.py`
   uses `asyncio.run(main())`. This keeps the backtest
   consistent with the async live trading system and
   allows non-blocking DB access during data loading.
