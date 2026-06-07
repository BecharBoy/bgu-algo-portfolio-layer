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
            predicted_target_price, gross_profit, net_profit, maximum_profit,
            maximum_loss, stop_history, graph_path
        )
        VALUES (
            $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17,
            $18,$19,$20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32::JSONB,$33
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
        trade.gross_profit,
        trade.net_profit,
        trade.maximum_profit,
        trade.maximum_loss,
        json_text(trade.stop_history),
        trade.graph_path,
    )


async def run_trade_rows(conn: asyncpg.Connection, run_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"SELECT * FROM {SCHEMA}.historical_trades WHERE run_id = $1 ORDER BY entry_at",
        run_id,
    )
