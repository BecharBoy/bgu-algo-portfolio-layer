from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import asyncpg
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from main_backtesting.models import PriceBar, ProbabilityPoint, ThresholdPass, Trade

SCHEMA = "checking_relevant_events"


def create_probability_graph(
    *,
    market_id: str,
    question: str,
    probabilities: list[ProbabilityPoint],
    passes: list[ThresholdPass],
    event_end: datetime,
    final_outcome: str | None,
    graph_dir: Path,
) -> Path:
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(
        [point.timestamp for point in probabilities],
        [point.probability * 100 for point in probabilities],
        color="#d97706",
        linewidth=1.6,
        label="Yes probability",
    )
    axis.axhline(55, color="#dc2626", linestyle="--", label="55% threshold")
    axis.axvline(event_end, color="#334155", linestyle=":", label="Event end")
    for item in passes:
        axis.scatter(
            [item.above_at],
            [item.above_probability * 100],
            marker="^",
            s=70,
            label=f"Pass {item.pass_number}",
        )
    axis.set_ylim(0, 100)
    axis.set_ylabel("Polymarket Yes probability (%)")
    axis.set_xlabel("UTC time")
    axis.set_title(f"{question}\nFinal result: {final_outcome or 'unknown'}")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    graph_dir.mkdir(parents=True, exist_ok=True)
    path = graph_dir / f"probability_{market_id}.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def create_trade_graph(
    trade: Trade,
    *,
    bars: list[PriceBar],
    probabilities: list[ProbabilityPoint],
    simulation_end: datetime,
    graph_dir: Path,
    event_end: datetime | None = None,
) -> Path:
    graph_end = event_end or trade.exit_at or simulation_end
    trade_bars = [bar for bar in bars if trade.entry_at <= bar.timestamp <= graph_end]
    probability_points = [
        point for point in probabilities if trade.entry_at <= point.timestamp <= graph_end
    ]
    stop_points = [
        item for item in trade.stop_history if trade.entry_at <= item["timestamp"] <= graph_end
    ]

    figure, price_axis = plt.subplots(figsize=(14, 7))
    if trade_bars:
        price_axis.plot(
            [bar.timestamp for bar in trade_bars],
            [bar.close for bar in trade_bars],
            color="#2563eb",
            linewidth=1.7,
            label=f"{trade.symbol} close",
        )
    if stop_points:
        price_axis.step(
            [item["timestamp"] for item in stop_points],
            [item["stop"] for item in stop_points],
            where="post",
            color="#dc2626",
            linewidth=1.4,
            label="Trailing stop",
        )
    if trade.predicted_target_price is not None:
        price_axis.axhline(
            trade.predicted_target_price,
            color="#7c3aed",
            linestyle="--",
            label="Machine-learning target",
        )
    if event_end is not None:
        price_axis.axvline(event_end, color="#334155", linestyle=":", label="Event end")
    price_axis.scatter(
        [trade.entry_at],
        [trade.entry_price],
        color="#16a34a",
        marker="^" if trade.direction == "long" else "v",
        s=90,
        label=f"{trade.direction.title()} entry",
        zorder=5,
    )
    if trade.exit_at and trade.exit_price is not None:
        price_axis.scatter(
            [trade.exit_at],
            [trade.exit_price],
            color="#dc2626",
            marker="x",
            s=90,
            label=f"Exit: {trade.exit_reason or 'unknown'}",
            zorder=5,
        )
    elif trade.final_mark_price is not None:
        price_axis.scatter(
            [simulation_end],
            [trade.final_mark_price],
            color="#0f766e",
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
            linewidth=1.3,
            label="Polymarket Yes probability",
        )
    probability_axis.axhline(55, color="#f59e0b", alpha=0.4, linestyle=":")
    probability_axis.set_ylim(0, 100)
    probability_axis.set_ylabel("Polymarket Yes probability (%)")

    title = (
        f"{trade.symbol} | {trade.portfolio} | {trade.strategy_branch} | "
        f"{trade.direction} | pass {trade.pass_number} | {trade.resolution}"
    )
    if trade.predicted_target_price is not None:
        title += f" | target reached: {trade.predicted_target_reached}"
    price_axis.set_title(title)
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
    path = graph_dir / (
        f"{trade.portfolio}_{trade.market_id}_pass_{trade.pass_number}_"
        f"{trade.symbol}_{trade.trade_id}.png"
    )
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


async def _write_query_csv(
    conn: asyncpg.Connection,
    *,
    path: Path,
    query: str,
    run_id: UUID,
) -> list[asyncpg.Record]:
    rows = await conn.fetch(query, run_id)
    _write_records_csv(path, rows)
    return rows


def _write_records_csv(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
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


async def generate_run_reports(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    run_dir: Path,
) -> None:
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    queries = {
        "trades.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 ORDER BY entry_at",
        "trades_hourly.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND resolution='1h' ORDER BY entry_at",
        "trades_daily.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND resolution='1d' ORDER BY entry_at",
        "polymarket_momentum.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio='polymarket_momentum' ORDER BY entry_at",
        "machine_learning_long.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio='machine_learning' AND direction='long' ORDER BY entry_at",
        "machine_learning_short.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio='machine_learning' AND direction='short' ORDER BY entry_at",
        "market_passes.csv": f"SELECT * FROM {SCHEMA}.historical_run_market_passes WHERE run_id=$1 ORDER BY above_at",
        "processed_markets.csv": f"SELECT * FROM {SCHEMA}.historical_run_markets WHERE run_id=$1 ORDER BY created_at, market_id",
        "market_filter_decisions.csv": f"""
            SELECT d.* FROM {SCHEMA}.historical_run_market_decisions r
            JOIN {SCHEMA}.historical_market_decisions d ON d.input_hash=r.input_hash
            WHERE r.run_id=$1 ORDER BY d.processed_at, d.market_id
        """,
        "deleted_non_relevant_markets.csv": f"""
            SELECT d.* FROM {SCHEMA}.historical_run_market_decisions r
            JOIN {SCHEMA}.historical_market_decisions d ON d.input_hash=r.input_hash
            WHERE r.run_id=$1 AND NOT d.relevant
            ORDER BY d.processed_at, d.market_id
        """,
        "asset_worlds.csv": f"""
            SELECT w.*, a.symbol, a.asset_name, a.asset_class, a.reason AS asset_reason
            FROM {SCHEMA}.historical_run_worlds r
            JOIN {SCHEMA}.historical_asset_worlds w ON w.world_id=r.world_id
            JOIN {SCHEMA}.historical_asset_world_assets a ON a.world_id=w.world_id
            WHERE r.run_id=$1 ORDER BY w.as_of, w.market_id, w.pass_number, a.symbol
        """,
        "ml_observations.csv": f"SELECT * FROM {SCHEMA}.historical_ml_observations WHERE run_id=$1 ORDER BY first_pass_at",
        "ml_model_snapshots.csv": f"SELECT * FROM {SCHEMA}.historical_ml_model_snapshots WHERE run_id=$1 ORDER BY training_cutoff",
        "ml_predictions.csv": f"SELECT * FROM {SCHEMA}.historical_ml_predictions WHERE run_id=$1 ORDER BY created_at",
        "stage_work.csv": f"SELECT * FROM {SCHEMA}.historical_backtest_stage_work WHERE run_id=$1 ORDER BY stage, work_key",
        "entry_decisions.csv": f"""
            SELECT work_key, payload, result
            FROM {SCHEMA}.historical_backtest_stage_work
            WHERE run_id=$1 AND stage='simulation'
            ORDER BY work_key
        """,
        "failures.csv": f"SELECT * FROM {SCHEMA}.historical_run_failures WHERE run_id=$1 ORDER BY failure_id",
    }
    outputs: dict[str, list[asyncpg.Record]] = {}
    for filename, query in queries.items():
        outputs[filename] = await _write_query_csv(
            conn,
            path=report_dir / filename,
            query=query,
            run_id=run_id,
        )

    trades = outputs["trades.csv"]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in trades:
        grouped[f"{row['portfolio']}:{row['resolution']}"].append(float(row["net_profit"] or 0.0))
    summary = {
        "run_id": str(run_id),
        "trade_count": len(trades),
        "portfolios": {
            key: {
                "trade_count": len(values),
                "total_net_profit": sum(values),
                "profitable_trades": sum(value > 0 for value in values),
                "unprofitable_trades": sum(value < 0 for value in values),
            }
            for key, values in grouped.items()
        },
    }
    predictions = outputs["ml_predictions.csv"]
    classified = [row for row in predictions if row["classification_correct"] is not None]
    long_predictions = [row for row in classified if row["direction"] == "long"]
    actual_longs = [row for row in classified if row["actual_direction"] == "long"]
    true_positive_longs = [
        row
        for row in classified
        if row["direction"] == "long" and row["actual_direction"] == "long"
    ]
    short_predictions = [row for row in classified if row["direction"] == "short"]
    actual_shorts = [row for row in classified if row["actual_direction"] == "short"]
    true_positive_shorts = [
        row
        for row in classified
        if row["direction"] == "short" and row["actual_direction"] == "short"
    ]
    regression_errors = [
        float(row["regression_error"])
        for row in predictions
        if row["regression_error"] is not None
    ]
    goal_rows = [row for row in predictions if row["target_reached"] is not None]
    ml_metrics = {
        "classification_observations": len(classified),
        "classification_accuracy": (
            sum(bool(row["classification_correct"]) for row in classified) / len(classified)
            if classified
            else None
        ),
        "long_precision": (
            len(true_positive_longs) / len(long_predictions) if long_predictions else None
        ),
        "long_recall": (
            len(true_positive_longs) / len(actual_longs) if actual_longs else None
        ),
        "short_precision": (
            len(true_positive_shorts) / len(short_predictions) if short_predictions else None
        ),
        "short_recall": (
            len(true_positive_shorts) / len(actual_shorts) if actual_shorts else None
        ),
        "ridge_mean_absolute_error": (
            sum(abs(value) for value in regression_errors) / len(regression_errors)
            if regression_errors
            else None
        ),
        "ridge_root_mean_square_error": (
            math.sqrt(sum(value * value for value in regression_errors) / len(regression_errors))
            if regression_errors
            else None
        ),
        "predicted_goal_hit_rate": (
            sum(bool(row["target_reached"]) for row in goal_rows) / len(goal_rows)
            if goal_rows
            else None
        ),
    }
    summary["machine_learning"] = ml_metrics
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (report_dir / "machine_learning_metrics.json").write_text(
        json.dumps(ml_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stage_rows = outputs["stage_work.csv"]
    market_decisions = outputs["market_filter_decisions.csv"]
    passes = outputs["market_passes.csv"]
    funnel = {
        "candidate_markets_filtered": len(market_decisions),
        "llm_accepted_markets": sum(bool(row["relevant"]) for row in market_decisions),
        "llm_deleted_markets": sum(not bool(row["relevant"]) for row in market_decisions),
        "processed_markets": len(outputs["processed_markets.csv"]),
        "markets_with_probability_passes": len({row["market_id"] for row in passes}),
        "asset_worlds": sum(
            row["stage"] == "asset_worlds" and row["status"] == "complete"
            for row in stage_rows
        ),
        "trades": len(trades),
    }
    (report_dir / "event_processing_funnel.json").write_text(
        json.dumps(funnel, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    calibration_rows = await conn.fetch(
        f"SELECT * FROM {SCHEMA}.historical_batch_calibrations ORDER BY created_at, task"
    )
    _write_records_csv(report_dir / "batch_calibration.csv", calibration_rows)

    if trades:
        figure, axis = plt.subplots(figsize=(12, 6))
        for key, values in grouped.items():
            cumulative: list[float] = []
            total = 0.0
            for value in values:
                total += value
                cumulative.append(total)
            axis.plot(range(1, len(cumulative) + 1), cumulative, label=key)
        axis.set_title("Cumulative net profit by portfolio and resolution")
        axis.set_xlabel("Trade number")
        axis.set_ylabel("Net profit ($)")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(report_dir / "portfolio_cumulative_profit.png", dpi=140)
        plt.close(figure)

        combined_values = [float(row["net_profit"] or 0.0) for row in trades]
        cumulative: list[float] = []
        drawdown: list[float] = []
        total = 0.0
        peak = 0.0
        for value in combined_values:
            total += value
            peak = max(peak, total)
            cumulative.append(total)
            drawdown.append(total - peak)
        figure, (equity_axis, drawdown_axis) = plt.subplots(2, 1, figsize=(12, 8))
        equity_axis.plot(range(1, len(cumulative) + 1), cumulative, color="#2563eb")
        equity_axis.set_title("Combined portfolio equity")
        equity_axis.set_ylabel("Net profit ($)")
        equity_axis.grid(alpha=0.25)
        drawdown_axis.fill_between(
            range(1, len(drawdown) + 1), drawdown, 0, color="#dc2626", alpha=0.35
        )
        drawdown_axis.set_title("Combined portfolio drawdown")
        drawdown_axis.set_xlabel("Trade number")
        drawdown_axis.set_ylabel("Drawdown ($)")
        drawdown_axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(report_dir / "combined_equity_and_drawdown.png", dpi=140)
        plt.close(figure)


def write_reports(trades: list[Trade], report_dir: Path) -> None:
    """Compatibility helper for the original focused tests."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "trade_count": len(trades),
                "total_net_profit": sum(trade.net_profit or 0.0 for trade in trades),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


async def export_database_reports(
    conn: asyncpg.Connection,
    run_id: UUID,
    report_dir: Path,
) -> None:
    await generate_run_reports(conn, run_id=run_id, run_dir=report_dir.parent)
