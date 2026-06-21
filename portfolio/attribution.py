from __future__ import annotations

from portfolio.models import TradeCandidate
from portfolio.portfolio import Portfolio


def compute_trade_alpha_rows(
    portfolio: Portfolio,
    candidates_by_id: dict[str, TradeCandidate | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for trade in portfolio.booked_trades:
        candidate_key = (
            f"{trade.run_id}:{trade.market_id}:{trade.pass_number}:"
            f"{trade.symbol}:{trade.strategy_branch}"
        )
        candidate = candidates_by_id.get(candidate_key)
        rows.append(
            {
                "symbol": trade.symbol,
                "event_id": trade.event_id,
                "strategy_branch": trade.strategy_branch,
                "direction": trade.direction,
                "entry_at": trade.entry_at.isoformat(),
                "exit_at": trade.exit_at.isoformat() if trade.exit_at else None,
                "net_profit": trade.net_profit,
                "event_archetype": candidate.event_archetype if candidate else None,
            }
        )
    return rows
