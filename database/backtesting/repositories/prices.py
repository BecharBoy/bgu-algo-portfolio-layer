from __future__ import annotations

from datetime import datetime
from typing import Any
import asyncpg
from main_backtesting.models import PriceBar

from database.backtesting.repositories._shared import SCHEMA, json_text


async def save_price_bars(
    conn: asyncpg.Connection,
    *,
    symbol: str,
    resolution: str,
    requested_start: datetime,
    requested_end: datetime,
    bars: list[PriceBar],
) -> None:
    if bars:
        await conn.executemany(
            f"""
            INSERT INTO {SCHEMA}.historical_price_bars
                (symbol, resolution, ts, open, high, low, close, volume)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
            ON CONFLICT (symbol, resolution, ts) DO NOTHING
            """,
            [
                (
                    symbol.upper(),
                    resolution,
                    bar.timestamp,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                )
                for bar in bars
            ],
        )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_price_coverage (
            symbol, resolution, requested_start, requested_end, first_ts, last_ts, row_count
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (symbol, resolution) DO UPDATE SET
            requested_start = LEAST({SCHEMA}.historical_price_coverage.requested_start, EXCLUDED.requested_start),
            requested_end = GREATEST({SCHEMA}.historical_price_coverage.requested_end, EXCLUDED.requested_end),
            first_ts = LEAST({SCHEMA}.historical_price_coverage.first_ts, EXCLUDED.first_ts),
            last_ts = GREATEST({SCHEMA}.historical_price_coverage.last_ts, EXCLUDED.last_ts),
            row_count = {SCHEMA}.historical_price_coverage.row_count + EXCLUDED.row_count,
            completed_at = NOW()
        """,
        symbol.upper(),
        resolution,
        requested_start,
        requested_end,
        bars[0].timestamp if bars else None,
        bars[-1].timestamp if bars else None,
        len(bars),
    )
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_price_download_windows (
            symbol, resolution, requested_start, requested_end, first_ts, last_ts, row_count
        )
        VALUES ($1,$2,$3,$4,$5,$6,$7)
        ON CONFLICT (symbol, resolution, requested_start, requested_end) DO UPDATE SET
            first_ts = EXCLUDED.first_ts,
            last_ts = EXCLUDED.last_ts,
            row_count = EXCLUDED.row_count,
            completed_at = NOW()
        """,
        symbol.upper(),
        resolution,
        requested_start,
        requested_end,
        bars[0].timestamp if bars else None,
        bars[-1].timestamp if bars else None,
        len(bars),
    )


async def price_is_covered(
    conn: asyncpg.Connection,
    *,
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
) -> bool:
    return bool(
        await conn.fetchval(
            f"""
            SELECT EXISTS (
                SELECT 1
                FROM {SCHEMA}.historical_price_download_windows
                WHERE symbol = $1 AND resolution = $2
                  AND requested_start <= $3 AND requested_end >= $4
            )
            """,
            symbol.upper(),
            resolution,
            start,
            end,
        )
    )


async def price_bars(
    conn: asyncpg.Connection,
    *,
    symbol: str,
    resolution: str,
    start: datetime,
    end: datetime,
) -> list[PriceBar]:
    rows = await conn.fetch(
        f"""
        SELECT ts, open, high, low, close, volume
        FROM {SCHEMA}.historical_price_bars
        WHERE symbol = $1 AND resolution = $2 AND ts >= $3 AND ts < $4
        ORDER BY ts
        """,
        symbol.upper(),
        resolution,
        start,
        end,
    )
    return [PriceBar(r["ts"], r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows]


async def save_asset_metadata(
    conn: asyncpg.Connection,
    *,
    symbol: str,
    metadata: dict[str, Any],
    missing_reason: str | None = None,
) -> None:
    await conn.execute(
        f"""
        INSERT INTO {SCHEMA}.historical_asset_metadata (
            symbol, asset_name, sector, sector_etf, metadata, missing_reason
        )
        VALUES ($1,$2,$3,$4,$5::JSONB,$6)
        ON CONFLICT (symbol) DO UPDATE SET
            asset_name = EXCLUDED.asset_name,
            sector = EXCLUDED.sector,
            sector_etf = EXCLUDED.sector_etf,
            metadata = EXCLUDED.metadata,
            missing_reason = EXCLUDED.missing_reason,
            updated_at = NOW()
        """,
        symbol.upper(),
        metadata.get("asset_name"),
        metadata.get("sector"),
        metadata.get("sector_etf"),
        json_text(metadata),
        missing_reason,
    )


async def asset_metadata(conn: asyncpg.Connection, symbol: str) -> asyncpg.Record | None:
    return await conn.fetchrow(
        f"SELECT * FROM {SCHEMA}.historical_asset_metadata WHERE symbol = $1",
        symbol.upper(),
    )
