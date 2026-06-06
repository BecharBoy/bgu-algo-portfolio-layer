from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from main_backtesting.models import PriceBar, ProbabilityPoint, Trade


def _plot_end(trade: Trade, simulation_end: datetime) -> datetime:
    return trade.exit_at or simulation_end


def create_trade_graph(
    trade: Trade,
    *,
    bars: list[PriceBar],
    probabilities: list[ProbabilityPoint],
    simulation_end: datetime,
    graph_dir: Path,
) -> Path:
    end = _plot_end(trade, simulation_end)
    trade_bars = [bar for bar in bars if trade.entry_at <= bar.timestamp <= end]
    probability_points = [
        point for point in probabilities if trade.entry_at <= point.timestamp <= end
    ]
    stop_points = [
        item
        for item in trade.stop_history
        if trade.entry_at <= item["timestamp"] <= end
    ]

    figure, price_axis = plt.subplots(figsize=(14, 7))
    if trade_bars:
        price_axis.plot(
            [bar.timestamp for bar in trade_bars],
            [bar.close for bar in trade_bars],
            color="#2563eb",
            linewidth=1.8,
            label=f"{trade.symbol} close",
        )
    if stop_points:
        price_axis.step(
            [item["timestamp"] for item in stop_points],
            [item["stop"] for item in stop_points],
            where="post",
            color="#dc2626",
            linewidth=1.5,
            label="Trailing stop-loss",
        )
    price_axis.scatter(
        [trade.entry_at],
        [trade.entry_price],
        color="#16a34a",
        marker="^",
        s=90,
        label="Entry",
        zorder=5,
    )
    if trade.exit_at and trade.exit_price is not None:
        price_axis.scatter(
            [trade.exit_at],
            [trade.exit_price],
            color="#dc2626",
            marker="v",
            s=90,
            label="Trailing-stop exit",
            zorder=5,
        )
    elif trade.final_mark_price is not None:
        price_axis.scatter(
            [end],
            [trade.final_mark_price],
            color="#7c3aed",
            marker="s",
            s=70,
            label="Simulation end",
            zorder=5,
        )

    probability_axis = price_axis.twinx()
    if probability_points:
        probability_axis.plot(
            [point.timestamp for point in probability_points],
            [point.probability * 100 for point in probability_points],
            color="#f59e0b",
            linestyle="--",
            linewidth=1.5,
            label="Polymarket Yes probability",
        )
    probability_axis.axhline(55, color="#f59e0b", alpha=0.4, linestyle=":")
    probability_axis.set_ylim(0, 100)
    probability_axis.set_ylabel("Polymarket Yes probability (%)")

    price_axis.set_title(
        f"{trade.symbol} | pass {trade.pass_number} | final market result: "
        f"{trade.final_outcome or 'unknown'}"
    )
    price_axis.set_ylabel("Asset price")
    price_axis.set_xlabel("UTC time")
    price_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    price_axis.grid(alpha=0.25)

    handles_1, labels_1 = price_axis.get_legend_handles_labels()
    handles_2, labels_2 = probability_axis.get_legend_handles_labels()
    price_axis.legend(handles_1 + handles_2, labels_1 + labels_2, loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()

    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / f"{trade.market_id}_pass_{trade.pass_number}_{trade.symbol}_{trade.trade_id}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_reports(trades: list[Trade], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    attribute_columns = [
        "trade_id",
        "market_id",
        "event_id",
        "question",
        "symbol",
        "asset_name",
        "pass_number",
        "trigger_at",
        "entry_at",
        "entry_price",
        "quantity",
        "entry_commission",
        "initial_stop",
        "exit_at",
        "exit_price",
        "exit_commission",
        "exit_reason",
        "final_mark_price",
        "maximum_price",
        "minimum_price",
        "final_outcome",
        "gross_profit",
        "net_profit",
        "graph_path",
    ]
    computed_columns = ["maximum_profit", "maximum_loss"]
    columns = attribute_columns[:-1] + computed_columns + ["graph_path"]
    with (report_dir / "trades.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for trade in trades:
            writer.writerow({column: getattr(trade, column) for column in columns})

    closed = [trade for trade in trades if trade.exit_at is not None]
    open_trades = [trade for trade in trades if trade.exit_at is None]
    summary: dict[str, Any] = {
        "trade_count": len(trades),
        "closed_by_trailing_stop": len(closed),
        "open_at_simulation_end": len(open_trades),
        "total_net_profit": sum(trade.net_profit or 0.0 for trade in trades),
        "profitable_trades": sum((trade.net_profit or 0.0) > 0 for trade in trades),
        "unprofitable_trades": sum((trade.net_profit or 0.0) < 0 for trade in trades),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


async def export_database_reports(
    conn: asyncpg.Connection,
    run_id: Any,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "threshold_passes.csv": """
            SELECT * FROM checking_relevant_events.backtest_market_passes
            WHERE run_id = $1 ORDER BY market_id, pass_number
        """,
        "sentiment_comparison.csv": """
            SELECT * FROM checking_relevant_events.backtest_sentiment_results
            WHERE run_id = $1 ORDER BY market_id, pass_number, symbol, provider
        """,
        "skipped_candidates.csv": """
            SELECT * FROM checking_relevant_events.backtest_skips
            WHERE run_id = $1 ORDER BY skip_id
        """,
    }
    for filename, query in queries.items():
        rows = await conn.fetch(query, run_id)
        if not rows:
            (report_dir / filename).write_text("", encoding="utf-8")
            continue
        columns = list(rows[0].keys())
        with (report_dir / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(value, ensure_ascii=False, default=str)
                        if isinstance(value, (dict, list))
                        else value
                        for key, value in dict(row).items()
                    }
                )
