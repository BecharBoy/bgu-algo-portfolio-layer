from __future__ import annotations

import csv
import json
from pathlib import Path

from portfolio.models import PortfolioDecision
from portfolio.portfolio import Portfolio


def write_portfolio_decisions(path: Path, decisions: list[PortfolioDecision]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "candidate_id",
        "run_id",
        "evaluated_at",
        "status",
        "reason",
        "binding_constraint",
        "symbol",
        "event_id",
        "market_id",
        "strategy_branch",
        "portfolio_label",
        "direction",
        "requested_quantity",
        "quantity",
        "notional",
        "risk_dollars",
        "effective_risk_pct",
        "risk_pct_of_equity",
        "equity_before",
        "cash_before",
        "heat_before",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(
                {
                    "candidate_id": decision.candidate_id,
                    "run_id": str(decision.run_id),
                    "evaluated_at": decision.evaluated_at.isoformat(),
                    "status": decision.status.value,
                    "reason": decision.reason,
                    "binding_constraint": decision.binding_constraint,
                    "symbol": decision.symbol,
                    "event_id": decision.event_id,
                    "market_id": decision.market_id,
                    "strategy_branch": decision.strategy_branch,
                    "portfolio_label": decision.portfolio_label,
                    "direction": decision.direction,
                    "requested_quantity": decision.requested_quantity,
                    "quantity": decision.quantity,
                    "notional": decision.notional,
                    "risk_dollars": decision.risk_dollars,
                    "effective_risk_pct": decision.effective_risk_pct,
                    "risk_pct_of_equity": decision.risk_pct_of_equity,
                    "equity_before": decision.equity_before,
                    "cash_before": decision.cash_before,
                    "heat_before": decision.heat_before,
                }
            )


def write_equity_curve(path: Path, snapshots: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", "cash", "equity", "drawdown"])
        writer.writeheader()
        writer.writerows(snapshots)


def write_exposure_csv(path: Path, snapshots: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timestamp",
                "gross_exposure",
                "net_exposure",
                "heat",
                "open_position_count",
            ],
        )
        writer.writeheader()
        writer.writerows(snapshots)


def write_portfolio_summary(path: Path, portfolio: Portfolio) -> None:
    counts: dict[str, int] = {}
    for decision in portfolio.decisions:
        counts[decision.status.value] = counts.get(decision.status.value, 0) + 1
    payload = {
        "starting_equity": portfolio.config.starting_capital,
        "final_equity": portfolio.state.equity,
        "final_cash": portfolio.state.cash,
        "max_drawdown": portfolio.state.drawdown,
        "decision_counts": counts,
        "booked_trade_count": len(portfolio.booked_trades),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_integration_notes(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Portfolio Integration Notes",
                "",
                "- Two-pass replay: Pass 1 computes invariant trade paths; Pass 2 replays portfolio decisions.",
                "- Momentum shadow variants remain capital-free; only the selected variant books capital.",
                "- Short positions use 100% cash margin with zero borrow fees in MVP.",
                "- Kill-switch and drawdown-aware sizing are evaluated at entry.",
                "- Slippage is zero in MVP; fills occur at modeled entry/exit prices.",
            ]
        ),
        encoding="utf-8",
    )


def generate_portfolio_reports(portfolio: Portfolio, report_dir: Path) -> None:
    write_portfolio_decisions(report_dir / "portfolio_decisions.csv", portfolio.decisions)
    write_equity_curve(report_dir / "portfolio_equity_curve.csv", portfolio.equity_snapshots)
    write_exposure_csv(report_dir / "portfolio_exposure.csv", portfolio.exposure_snapshots)
    write_portfolio_summary(report_dir / "portfolio_summary.json", portfolio)
    write_integration_notes(report_dir / "PORTFOLIO_INTEGRATION_NOTES.md")
    alpha_rows = compute_trade_alpha_rows(portfolio, {})
    alpha_path = report_dir / "trade_alpha.csv"
    alpha_path.parent.mkdir(parents=True, exist_ok=True)
    if alpha_rows:
        with alpha_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(alpha_rows[0].keys()))
            writer.writeheader()
            writer.writerows(alpha_rows)
    baseline_path = report_dir / "baseline_vs_portfolio.csv"
    with baseline_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["metric", "legacy_fixed_notional", "portfolio_layer"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "metric": "booked_trade_count",
                "legacy_fixed_notional": "",
                "portfolio_layer": len(portfolio.booked_trades),
            }
        )
