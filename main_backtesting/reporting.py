from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
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


def matplotlib_literal_text(value: object) -> str:
    return str(value).replace("$", r"\$")


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
    axis.set_title(
        f"{matplotlib_literal_text(question)}\n"
        f"Final result: {matplotlib_literal_text(final_outcome or 'unknown')}"
    )
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
    graph_end = event_end + timedelta(days=3) if event_end else trade.exit_at or simulation_end
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


def _performance_summary(values: list[float]) -> dict[str, float | int | None]:
    wins = [value for value in values if value > 0]
    losses = [value for value in values if value < 0]
    average_win = sum(wins) / len(wins) if wins else None
    average_loss = sum(losses) / len(losses) if losses else None
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    consecutive = maximum_consecutive = 0
    for value in values:
        consecutive = consecutive + 1 if value < 0 else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
    return {
        "trade_count": len(values),
        "total_net_profit": sum(values),
        "win_rate": len(wins) / len(values) if values else None,
        "average_winner": average_win,
        "average_loser": average_loss,
        "payoff_ratio": (
            average_win / abs(average_loss)
            if average_win is not None and average_loss not in {None, 0}
            else None
        ),
        "profit_factor": gross_profit / gross_loss if gross_loss else None,
        "expectancy": sum(values) / len(values) if values else None,
        "maximum_consecutive_losses": maximum_consecutive,
    }


def _group_performance(rows: list[Any], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key] if row[key] is not None else "unknown")].append(
            float(row["net_profit"] or 0.0)
        )
    return [
        {key: value, **_performance_summary(profits)}
        for value, profits in sorted(grouped.items())
    ]


def _daily_portfolio_metrics(
    trades: list[Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pnl_by_day: dict[date, float] = defaultdict(float)
    notional_by_day: dict[date, float] = defaultdict(float)
    for trade in trades:
        realized_at = trade["exit_at"] or trade["entry_at"]
        day = realized_at.date()
        pnl_by_day[day] += float(trade["net_profit"] or 0.0)
        notional_by_day[day] += 1_000.0
    daily_rows: list[dict[str, Any]] = []
    cumulative = peak = 0.0
    maximum_drawdown = 0.0
    returns: list[float] = []
    if pnl_by_day:
        cursor = min(pnl_by_day)
        final_day = max(pnl_by_day)
    else:
        cursor = final_day = None
    while cursor is not None and final_day is not None and cursor <= final_day:
        if cursor.weekday() >= 5:
            cursor += timedelta(days=1)
            continue
        pnl = pnl_by_day.get(cursor, 0.0)
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = cumulative - peak
        maximum_drawdown = min(maximum_drawdown, drawdown)
        realized_notional = notional_by_day.get(cursor, 0.0)
        daily_return = pnl / realized_notional if realized_notional else 0.0
        returns.append(daily_return)
        daily_rows.append(
            {
                "date": cursor.isoformat(),
                "realized_daily_pnl": pnl,
                "realized_signal_notional": realized_notional,
                "daily_return": daily_return,
                "cumulative_pnl": cumulative,
                "drawdown": drawdown,
            }
        )
        cursor += timedelta(days=1)
    mean_return = sum(returns) / len(returns) if returns else None
    variance = (
        sum((value - mean_return) ** 2 for value in returns) / len(returns)
        if returns and mean_return is not None
        else None
    )
    downside = [min(value, 0.0) for value in returns]
    downside_variance = (
        sum(value * value for value in downside) / len(downside) if downside else None
    )
    sharpe = (
        mean_return / math.sqrt(variance) * math.sqrt(252)
        if mean_return is not None and variance and variance > 0
        else None
    )
    sortino = (
        mean_return / math.sqrt(downside_variance) * math.sqrt(252)
        if mean_return is not None and downside_variance and downside_variance > 0
        else None
    )
    total_notional = len(trades) * 1_000.0
    return (
        {
            "method": "daily_realized_executable_trade_returns",
            "daily_realized_days": len(daily_rows),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "maximum_drawdown_dollars": maximum_drawdown,
            "maximum_drawdown_percent_of_total_signal_notional": (
                maximum_drawdown / total_notional if total_notional else None
            ),
            "total_signal_notional": total_notional,
        },
        daily_rows,
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
        "polymarket_momentum.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio LIKE 'polymarket_momentum%' ORDER BY entry_at",
        "machine_learning_long.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio LIKE 'machine_learning%' AND direction='long' ORDER BY entry_at",
        "machine_learning_short.csv": f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id=$1 AND portfolio LIKE 'machine_learning%' AND direction='short' ORDER BY entry_at",
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
        "asset_symbol_resolutions.csv": f"""
            SELECT * FROM {SCHEMA}.historical_run_asset_resolutions
            WHERE run_id=$1 ORDER BY original_symbol
        """,
        "rejected_asset_symbols.csv": f"""
            SELECT * FROM {SCHEMA}.historical_run_asset_resolutions
            WHERE run_id=$1 AND resolved_symbol IS NULL
            ORDER BY original_symbol
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
        "momentum_parameter_results.csv": f"""
            SELECT * FROM {SCHEMA}.historical_momentum_parameter_results
            WHERE run_id=$1
            ORDER BY trigger_at, market_id, pass_number, symbol,
                     range_period, range_multiplier
        """,
        "momentum_parameter_performance.csv": f"""
            SELECT resolution, range_period, range_multiplier,
                   COUNT(*) AS signal_count,
                   COUNT(*) FILTER (WHERE opened) AS opened_trade_count,
                   SUM(COALESCE(net_profit, 0.0)) AS total_signal_net_profit,
                   AVG(COALESCE(net_profit, 0.0)) AS average_signal_net_profit,
                   AVG(net_profit) FILTER (WHERE opened) AS average_opened_trade_net_profit
            FROM {SCHEMA}.historical_momentum_parameter_results
            WHERE run_id=$1
            GROUP BY resolution, range_period, range_multiplier
            ORDER BY resolution, average_signal_net_profit DESC,
                     range_period, range_multiplier
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
        grouped[f"{row['strategy_branch']}:{row['resolution']}"].append(
            float(row["net_profit"] or 0.0)
        )
    overall_performance = _performance_summary(
        [float(row["net_profit"] or 0.0) for row in trades]
    )
    portfolio_metrics, daily_rows = _daily_portfolio_metrics(trades)
    _write_records_csv(report_dir / "daily_realized_portfolio.csv", daily_rows)
    strategy_metrics: dict[str, Any] = {}
    for branch in sorted({str(row["strategy_branch"]) for row in trades}):
        branch_trades = [row for row in trades if row["strategy_branch"] == branch]
        branch_portfolio, _ = _daily_portfolio_metrics(branch_trades)
        strategy_metrics[branch] = {
            "performance": _performance_summary(
                [float(row["net_profit"] or 0.0) for row in branch_trades]
            ),
            "realized_portfolio": branch_portfolio,
        }
    summary = {
        "run_id": str(run_id),
        "trade_count": len(trades),
        "overall_performance": overall_performance,
        "equal_notional_signal_portfolio": portfolio_metrics,
        "equal_notional_executable_signal_portfolio": portfolio_metrics,
        "strategies": strategy_metrics,
        "strategy_resolutions": {
            key: _performance_summary(values)
            for key, values in grouped.items()
        },
        "portfolios": {
            key: _performance_summary(values)
            for key, values in grouped.items()
        },
    }
    performance_rows = await conn.fetch(
        f"""
        SELECT t.*,
               COALESCE(o.event_archetype, 'not_ml_eligible') AS event_archetype,
               EXTRACT(YEAR FROM t.entry_at)::INTEGER AS entry_year,
               CASE WHEN COUNT(*) OVER (
                   PARTITION BY t.market_id, t.pass_number, t.symbol, t.direction
               ) > 1 THEN 'repeated_identical_signal' ELSE 'single_signal' END
                   AS duplicate_status
        FROM {SCHEMA}.historical_trades t
        LEFT JOIN {SCHEMA}.historical_ml_observations o
          ON o.run_id=t.run_id AND o.event_id=t.event_id AND o.symbol=t.symbol
        WHERE t.run_id=$1
        ORDER BY t.entry_at, t.trade_id
        """,
        run_id,
    )
    for key, filename in [
        ("entry_year", "performance_by_year.csv"),
        ("event_archetype", "performance_by_archetype.csv"),
        ("symbol", "performance_by_ticker.csv"),
        ("exit_reason", "performance_by_exit_reason.csv"),
        ("duplicate_status", "performance_by_duplicate_status.csv"),
    ]:
        _write_records_csv(report_dir / filename, _group_performance(performance_rows, key))
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
        axis.set_title("Cumulative net profit by executable strategy and resolution")
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
