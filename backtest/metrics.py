from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd


def compute_metrics(
    equity_curve: pd.DataFrame,
    trade_log: list[dict[str, Any]],
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    if equity_curve.empty:
        return {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "num_trades": 0.0,
            "avg_hold_days": 0.0,
        }

    curve = equity_curve.copy()
    curve["returns"] = curve["equity"].pct_change().fillna(0.0)
    daily_rf = risk_free_rate / 252.0
    excess_returns = curve["returns"] - daily_rf
    volatility = curve["returns"].std(ddof=1)
    sharpe = 0.0 if volatility == 0 or math.isnan(volatility) else (excess_returns.mean() / volatility) * math.sqrt(252.0)

    running_peak = curve["equity"].cummax()
    drawdown = (curve["equity"] - running_peak) / running_peak
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    starting_equity = float(curve["equity"].iloc[0])
    ending_equity = float(curve["equity"].iloc[-1])
    total_return = 0.0 if starting_equity == 0 else (ending_equity / starting_equity) - 1.0

    num_periods = max(len(curve) - 1, 1)
    annualized_return = (ending_equity / starting_equity) ** (252.0 / num_periods) - 1.0 if starting_equity > 0 else 0.0
    calmar = 0.0 if max_drawdown == 0 else annualized_return / abs(max_drawdown)

    closed_trades = [trade for trade in trade_log if trade.get("realized_pnl") is not None]
    winners = [trade for trade in closed_trades if float(trade["realized_pnl"]) > 0]
    losers = [trade for trade in closed_trades if float(trade["realized_pnl"]) < 0]

    win_rate = 0.0 if not closed_trades else len(winners) / len(closed_trades)
    gross_profit = sum(float(trade["realized_pnl"]) for trade in winners)
    gross_loss = abs(sum(float(trade["realized_pnl"]) for trade in losers))
    profit_factor = 0.0 if gross_loss == 0 else gross_profit / gross_loss

    hold_days = [float(trade["hold_days"]) for trade in closed_trades if trade.get("hold_days") is not None]
    avg_hold_days = 0.0 if not hold_days else sum(hold_days) / len(hold_days)

    return {
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "num_trades": float(len(closed_trades)),
        "avg_hold_days": float(avg_hold_days),
    }


def plot_equity_curve(
    equity_curve: pd.DataFrame,
    output_path: str = "backtest/results/equity_curve.png",
) -> None:
    if equity_curve.empty:
        return

    import matplotlib.pyplot as plt

    curve = equity_curve.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    running_peak = curve["equity"].cummax()
    curve["drawdown"] = (curve["equity"] - running_peak) / running_peak

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax_equity, ax_drawdown) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_equity.plot(curve["date"], curve["equity"], color="navy", linewidth=1.8)
    ax_equity.set_title("Backtest Equity Curve")
    ax_equity.set_ylabel("Equity")
    ax_equity.grid(True, alpha=0.3)

    ax_drawdown.fill_between(curve["date"], curve["drawdown"], 0.0, color="firebrick", alpha=0.35)
    ax_drawdown.set_ylabel("Drawdown")
    ax_drawdown.set_xlabel("Date")
    ax_drawdown.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
