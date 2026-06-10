from __future__ import annotations

from uuid import UUID
import asyncpg
from main_backtesting.models import Trade

from database.backtesting.repositories._shared import SCHEMA, json_text


async def save_trade(conn: asyncpg.Connection, trade: Trade) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_trades (
            trade_id, run_id, portfolio, strategy_branch, resolution, direction,
            market_id, event_id, question, symbol, asset_name, pass_number,
            trigger_at, entry_at, entry_price, quantity, entry_commission,
            initial_stop, exit_at, exit_price, exit_commission, exit_reason,
            final_mark_price, maximum_price, minimum_price, final_outcome,
            predicted_target_price, range_period, range_multiplier,
            parameter_selection, gross_profit, net_profit, maximum_profit,
            maximum_loss, stop_history, graph_path
        )
        VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
            $18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30::JSONB,
            $31,$32,$33,$34,$35::JSONB,$36
        )
        ON CONFLICT (run_id, portfolio, market_id, pass_number, symbol) DO NOTHING
        """,
        trade.trade_id,
        trade.run_id,
        trade.portfolio,
        trade.strategy_branch,
        trade.resolution,
        trade.direction,
        trade.market_id,
        trade.event_id,
        trade.question,
        trade.symbol,
        trade.asset_name,
        trade.pass_number,
        trade.trigger_at,
        trade.entry_at,
        trade.entry_price,
        trade.quantity,
        trade.entry_commission,
        trade.initial_stop,
        trade.exit_at,
        trade.exit_price,
        trade.exit_commission,
        trade.exit_reason,
        trade.final_mark_price,
        trade.maximum_price,
        trade.minimum_price,
        trade.final_outcome,
        trade.predicted_target_price,
        trade.range_period,
        trade.range_multiplier,
        json_text(trade.parameter_selection),
        trade.gross_profit,
        trade.net_profit,
        trade.maximum_profit,
        trade.maximum_loss,
        json_text(trade.stop_history),
        trade.graph_path,
    )


async def save_momentum_parameter_results(
    conn: asyncpg.Connection,
    rows: list[dict[str, object]],
) -> None:
    if not rows:
        return
    await conn.executemany(
        f"""
        INSERT INTO {SCHEMA}.historical_momentum_parameter_results (
            run_id, market_id, event_id, pass_number, symbol, trigger_at,
            resolution, range_period, range_multiplier, opened, reason, net_profit
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
        ON CONFLICT (
            run_id, market_id, pass_number, symbol, range_period, range_multiplier
        ) DO UPDATE SET
            opened=EXCLUDED.opened,
            reason=EXCLUDED.reason,
            net_profit=EXCLUDED.net_profit,
            created_at=NOW()
        """,
        [
            (
                row["run_id"],
                row["market_id"],
                row["event_id"],
                row["pass_number"],
                str(row["symbol"]).upper(),
                row["trigger_at"],
                row["resolution"],
                row["range_period"],
                row["range_multiplier"],
                row["opened"],
                row["reason"],
                row["net_profit"],
            )
            for row in rows
        ],
    )


async def select_walk_forward_momentum_parameters(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    before: object,
    resolution: str,
    minimum_samples: int,
    fallback_period: int,
    fallback_multiplier: float,
) -> dict[str, object]:
    row = await conn.fetchrow(
        f"""
        SELECT range_period, range_multiplier, COUNT(*) AS sample_count,
               AVG(COALESCE(net_profit, 0.0)) AS average_net_profit
        FROM {SCHEMA}.historical_momentum_parameter_results
        WHERE run_id=$1 AND resolution=$2 AND trigger_at < $3
        GROUP BY range_period, range_multiplier
        HAVING COUNT(*) >= $4
        ORDER BY AVG(COALESCE(net_profit, 0.0)) DESC, COUNT(*) DESC,
                 range_period, range_multiplier
        LIMIT 1
        """,
        run_id,
        resolution,
        before,
        minimum_samples,
    )
    if row is None:
        return {
            "range_period": fallback_period,
            "range_multiplier": fallback_multiplier,
            "selection_method": "configured_fallback",
            "prior_sample_count": 0,
            "prior_average_net_profit": None,
        }
    return {
        "range_period": row["range_period"],
        "range_multiplier": row["range_multiplier"],
        "selection_method": "walk_forward_best_prior_net_expectancy",
        "prior_sample_count": row["sample_count"],
        "prior_average_net_profit": row["average_net_profit"],
    }


async def run_trade_rows(conn: asyncpg.Connection, run_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id = $1 ORDER BY entry_at",
        run_id,
    )
