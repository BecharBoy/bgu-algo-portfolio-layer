from __future__ import annotations
from datetime import datetime

import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from theory_testing.backtest_cointegration.data_loader import load_from_csv, load_from_db
from theory_testing.backtest_cointegration.engine import BacktestEngine
from theory_testing.backtest_cointegration.metrics import compute_metrics, plot_equity_curve
from theory_testing.backtest_cointegration.portfolio import Portfolio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies" / "cointegration"))

from strategies.mean_reversion.meanreversion import MeanReversionMomentum
from strategies.cointegration.StatArbStrategy import StatArbStrategy

# ── Config ────────────────────────────────────────────────────────────────────
LOAD_MODE     = "db"
CSV_PATH      = "backtest_cointegration/sample_data"
TICKERS: list[str] | None = None
START_DATE    = "2024-04-01"
END_DATE      = "2026-03-01"
LOOKBACK_DAYS = 252
INITIAL_CASH  = 100_000.0
SLIPPAGE      = 0.0005
RUN_ID        = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR   = Path(__file__).resolve().parent / "results" / RUN_ID

strategies = [
    MeanReversionMomentum(capital_allocation=0.15),
    StatArbStrategy(capital_allocation=0.15),
]


def _load_default_tickers() -> list[str]:
    tickers_path = Path(__file__).resolve().parent.parent / "tickers.csv"
    if not tickers_path.exists():
        raise FileNotFoundError(f"tickers.csv not found at {tickers_path}")
    with tickers_path.open("r", encoding="utf-8") as handle:
        tickers = [line.strip() for line in handle if line.strip()]
    if not tickers:
        raise ValueError("tickers.csv is empty")
    return tickers


async def _load_data() -> dict[str, pd.DataFrame]:
    tickers = TICKERS or _load_default_tickers()
    if LOAD_MODE == "db":
        return await load_from_db(
            tickers=tickers,
            start_date=START_DATE,
            end_date=END_DATE,
            lookback_days=LOOKBACK_DAYS,
        )
    if LOAD_MODE == "csv":
        return load_from_csv(CSV_PATH)
    raise ValueError(f"Unsupported LOAD_MODE={LOAD_MODE}")


async def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    data = await _load_data()
    if not data:
        raise RuntimeError("No market data loaded — has bootstrap_history_job run?")

    # ── Per-strategy portfolios: each gets 50% of capital ────────────────────
    half_cash = INITIAL_CASH / 2.0
    portfolios: dict[str, Portfolio] = {
        s.name: Portfolio(
            initial_cash=half_cash,
            strategy_name=s.name,
            slippage=SLIPPAGE,
        )
        for s in strategies
    }

    engine = BacktestEngine(
        strategies=strategies,
        portfolios=portfolios,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback=LOOKBACK_DAYS,
        min_bars=52,
    )

    equity_curve = await engine.run()

    # ── Merge trade + rejection logs from all portfolios ─────────────────────
    all_trades     = []
    all_rejections = []
    for portfolio in portfolios.values():
        all_trades.extend(portfolio.get_trade_log())
        all_rejections.extend(portfolio.get_rejection_log())

    metrics = compute_metrics(equity_curve, all_trades)

    equity_curve.to_csv(RESULTS_DIR / "equity_curve.csv", index=False)
    pd.DataFrame(all_trades).to_csv(RESULTS_DIR / "trade_log.csv", index=False)
    pd.DataFrame(all_rejections).to_csv(RESULTS_DIR / "rejections.csv", index=False)
    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    plot_equity_curve(equity_curve, str(RESULTS_DIR / "equity_curve.png"))

    print("\n" + "=" * 45)
    for key, value in metrics.items():
        print(f"  {key:22s}: {value}")
    print("=" * 45)
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
