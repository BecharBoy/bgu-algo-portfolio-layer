from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # must be before any os.environ access

from backtest.data_loader import load_from_csv, load_from_db
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics, plot_equity_curve
from backtest.portfolio import Portfolio

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "strategies" / "cointegration"))


from strategies.mean_reversion.meanreversion import MeanReversionMomentum
from strategies.cointegration.StatArbStrategy import StatArbStrategy


strategies = [
    MeanReversionMomentum(capital_allocation=0.50),
    StatArbStrategy(capital_allocation=0.50),
]

LOAD_MODE    = "db"
CSV_PATH     = "backtest/sample_data"
TICKERS: list[str] | None = None
START_DATE   = "2024-04-01"   # matches bootstrapped 2y history
END_DATE     = "2026-03-01"
LOOKBACK_DAYS = 252            # warmup aligned with engine window
INITIAL_CASH  = 100_000.0
COMMISSION    = 0.001
SLIPPAGE      = 0.0005
RESULTS_DIR   = Path(__file__).resolve().parent / "results"


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

    from strategies.mean_reversion.meanreversion import MeanReversionMomentum

    strategies = [MeanReversionMomentum(capital_allocation=0.30)]

    portfolio = Portfolio(
        initial_cash=INITIAL_CASH,
        commission=COMMISSION,
        slippage=SLIPPAGE,
    )
    engine = BacktestEngine(
        strategies=strategies,
        portfolio=portfolio,
        data=data,
        start_date=START_DATE,
        end_date=END_DATE,
        lookback=LOOKBACK_DAYS,
        min_bars=52,
    )

    equity_curve = await engine.run()
    trade_log    = portfolio.get_trade_log()
    metrics      = compute_metrics(equity_curve, trade_log)

    equity_curve.to_csv(RESULTS_DIR / "equity_curve.csv", index=False)
    pd.DataFrame(trade_log).to_csv(RESULTS_DIR / "trade_log.csv", index=False)
    pd.DataFrame(portfolio.get_rejection_log()).to_csv(RESULTS_DIR / "rejections.csv", index=False)
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
