from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4
import asyncpg
from main_backtesting.models import Asset, IBTradableAsset, SourceMarket

from database.backtesting.repositories._shared import SCHEMA, json_text


async def ib_tradable_symbols(conn: asyncpg.Connection) -> set[str]:
    exists = await conn.fetchval("SELECT to_regclass('checking_relevant_events.ib_assets')")
    if exists is None:
        return set()
    rows = await conn.fetch(
        """
        SELECT DISTINCT symbol
        FROM checking_relevant_events.ib_assets
        WHERE currency = 'USD'
          AND 'SMART' = ANY(valid_exchanges)
          AND CARDINALITY(order_types) > 0
        ORDER BY symbol
        """
    )
    return {str(row["symbol"]).upper() for row in rows}


async def ib_tradable_assets(conn: asyncpg.Connection) -> list[IBTradableAsset]:
    exists = await conn.fetchval("SELECT to_regclass('checking_relevant_events.ib_assets')")
    if exists is None:
        return []
    rows = await conn.fetch(
        """
        SELECT symbol, security_name, primary_exchange, stock_type, is_etf,
               industry, category, subcategory
        FROM checking_relevant_events.ib_assets
        WHERE currency = 'USD'
          AND 'SMART' = ANY(valid_exchanges)
          AND CARDINALITY(order_types) > 0
          AND (
              stock_type = 'ETF'
              OR stock_type IN (
                  'COMMON', 'ADR', 'REIT', 'MLP', 'LTD PART',
                  'NY REG SHRS', 'TRACKING STK', 'US DOMESTIC'
              )
          )
        ORDER BY symbol
        """
    )
    return [
        IBTradableAsset(
            symbol=str(row["symbol"]).upper(),
            asset_name=str(row["security_name"]),
            asset_class="etf" if row["stock_type"] == "ETF" or row["is_etf"] else "stock",
            primary_exchange=str(row["primary_exchange"]),
            stock_type=str(row["stock_type"]),
            industry=row["industry"],
            category=row["category"],
            subcategory=row["subcategory"],
        )
        for row in rows
    ]


async def reusable_world(conn: asyncpg.Connection, input_hash: str) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_asset_worlds WHERE input_hash = $1",
        input_hash,
    )


async def reusable_world_for_pass(
    conn: asyncpg.Connection,
    *,
    market_id: str,
    pass_number: int,
    as_of: datetime,
    model_name: str,
    prompt_version: str,
) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"""
        SELECT * FROM {SCHEMA}.historical_asset_worlds
        WHERE market_id=$1 AND pass_number=$2 AND as_of=$3
          AND model_name=$4 AND prompt_version=$5
        ORDER BY created_at DESC
        LIMIT 1
        """,
        market_id,
        pass_number,
        as_of,
        model_name,
        prompt_version,
    )


async def link_run_world(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    market_id: str,
    pass_number: int,
    world_id: UUID,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_worlds (run_id, market_id, pass_number, world_id)
        VALUES ($1,$2,$3,$4)
        ON CONFLICT (run_id, market_id, pass_number) DO UPDATE SET world_id=EXCLUDED.world_id
        """,
        run_id,
        market_id,
        pass_number,
        world_id,
    )


async def save_world(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    input_hash: str,
    market: SourceMarket,
    pass_number: int,
    as_of: datetime,
    model_name: str,
    prompt_version: str,
    llm_input: dict[str, Any],
    llm_output: dict[str, Any],
    universe_name: str,
    universe_reason: str,
    assets: list[Asset],
) -> UUID:
    existing = await reusable_world(conn, input_hash)
    world_id = existing["world_id"] if existing else uuid4()
    if existing is None:
        await conn.execute(
            f"""
            INSERT INTO {SCHEMA}.historical_asset_worlds (
                world_id, input_hash, market_id, event_id, pass_number, as_of,
                model_name, prompt_version, llm_input, llm_output,
                universe_name, universe_reason
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::JSONB,$10::JSONB,$11,$12)
            """,
            world_id,
            input_hash,
            market.market_id,
            market.event_id,
            pass_number,
            as_of,
            model_name,
            prompt_version,
            json_text(llm_input),
            json_text(llm_output),
            universe_name,
            universe_reason,
        )
        await conn.executemany(
            f"""
            INSERT INTO {SCHEMA}.historical_asset_world_assets
                (world_id, symbol, asset_name, asset_class, reason)
            VALUES ($1,$2,$3,$4,$5)
            """,
            [
                (world_id, asset.symbol, asset.asset_name, asset.asset_class, asset.reason)
                for asset in assets
            ],
        )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_run_worlds (run_id, market_id, pass_number, world_id)
        VALUES ($1,$2,$3,$4)
        ON CONFLICT (run_id, market_id, pass_number) DO UPDATE SET world_id = EXCLUDED.world_id
        """,
        run_id,
        market.market_id,
        pass_number,
        world_id,
    )
    return world_id


async def run_world_assets(conn: asyncpg.Connection, run_id: UUID) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT rw.market_id, rw.pass_number, rw.world_id, w.event_id, w.as_of,
               a.symbol, a.asset_name, a.asset_class, a.reason
        FROM {SCHEMA}.historical_run_worlds rw
        JOIN {SCHEMA}.historical_asset_worlds w ON w.world_id = rw.world_id
        JOIN {SCHEMA}.historical_asset_world_assets a ON a.world_id = rw.world_id
        WHERE rw.run_id = $1
        ORDER BY w.as_of, rw.market_id, rw.pass_number, a.symbol
        """,
        run_id,
    )


async def run_resolved_world_assets(
    conn: asyncpg.Connection,
    run_id: UUID,
) -> list[asyncpg.Record]:
    return await conn.fetch(
        f"""
        SELECT *
        FROM (
            SELECT DISTINCT ON (rw.market_id, rw.pass_number, r.resolved_symbol)
                   rw.market_id, rw.pass_number, rw.world_id, w.event_id, w.as_of,
                   r.resolved_symbol AS symbol, r.original_symbol,
                   COALESCE(r.security_name, a.asset_name) AS asset_name,
                   a.asset_class, a.reason
            FROM {SCHEMA}.historical_run_worlds rw
            JOIN {SCHEMA}.historical_asset_worlds w ON w.world_id = rw.world_id
            JOIN {SCHEMA}.historical_asset_world_assets a ON a.world_id = rw.world_id
            JOIN {SCHEMA}.historical_run_asset_resolutions r
              ON r.run_id = rw.run_id AND r.original_symbol = a.symbol
            WHERE rw.run_id = $1 AND r.resolved_symbol IS NOT NULL
            ORDER BY rw.market_id, rw.pass_number, r.resolved_symbol,
                     CASE WHEN a.symbol = r.resolved_symbol THEN 0 ELSE 1 END,
                     a.symbol
        ) resolved
        ORDER BY as_of, market_id, pass_number, symbol
        """,
        run_id,
    )


async def save_world_feedback(
    conn: asyncpg.Connection,
    *,
    run_id: UUID,
    world_id: UUID,
    symbol: str,
    metrics: dict[str, Any],
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_world_feedback (
            run_id, world_id, symbol, realized_volatility, baseline_volatility,
            volatility_increase, probability_correlation, maximum_favorable_move,
            maximum_adverse_move, return_vs_spy, return_vs_sector, ml_goal_reached,
            trade_net_profit, metrics
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14::JSONB)
        ON CONFLICT (run_id, world_id, symbol) DO UPDATE SET
            realized_volatility = EXCLUDED.realized_volatility,
            baseline_volatility = EXCLUDED.baseline_volatility,
            volatility_increase = EXCLUDED.volatility_increase,
            probability_correlation = EXCLUDED.probability_correlation,
            maximum_favorable_move = EXCLUDED.maximum_favorable_move,
            maximum_adverse_move = EXCLUDED.maximum_adverse_move,
            return_vs_spy = EXCLUDED.return_vs_spy,
            return_vs_sector = EXCLUDED.return_vs_sector,
            ml_goal_reached = EXCLUDED.ml_goal_reached,
            trade_net_profit = EXCLUDED.trade_net_profit,
            metrics = EXCLUDED.metrics
        """,
        run_id,
        world_id,
        symbol.upper(),
        metrics.get("realized_volatility"),
        metrics.get("baseline_volatility"),
        metrics.get("volatility_increase"),
        metrics.get("probability_correlation"),
        metrics.get("maximum_favorable_move"),
        metrics.get("maximum_adverse_move"),
        metrics.get("return_vs_spy"),
        metrics.get("return_vs_sector"),
        metrics.get("ml_goal_reached"),
        metrics.get("trade_net_profit"),
        json_text(metrics),
    )
