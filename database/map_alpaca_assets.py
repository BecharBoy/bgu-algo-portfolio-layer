from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.db_connection import connect

ENV_PATH = REPO_ROOT / ".env"
SCHEMA_NAME = "checking_relevant_events"

TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.alpaca_assets (
    asset_id                         UUID PRIMARY KEY,
    symbol                           TEXT NOT NULL UNIQUE,
    name                             TEXT NOT NULL,
    asset_class                      TEXT NOT NULL,
    exchange                         TEXT,
    status                           TEXT NOT NULL,
    tradable                         BOOLEAN NOT NULL,
    marginable                       BOOLEAN NOT NULL,
    shortable                        BOOLEAN NOT NULL,
    easy_to_borrow                   BOOLEAN NOT NULL,
    fractionable                     BOOLEAN NOT NULL,
    maintenance_margin_requirement   DOUBLE PRECISION,
    attributes                       TEXT[] NOT NULL,
    raw_asset                        JSONB NOT NULL,
    updated_at                       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_alpaca_assets_symbol
    ON {SCHEMA_NAME}.alpaca_assets(symbol);
"""


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def asset_row(asset: dict[str, Any]) -> tuple[Any, ...]:
    return (
        asset["id"],
        asset["symbol"],
        asset["name"],
        asset["class"],
        asset.get("exchange"),
        asset["status"],
        asset["tradable"],
        asset["marginable"],
        asset["shortable"],
        asset["easy_to_borrow"],
        asset["fractionable"],
        asset.get("maintenance_margin_requirement"),
        asset.get("attributes") or [],
        json.dumps(asset, ensure_ascii=False),
    )


async def main() -> None:
    load_dotenv(ENV_PATH)
    base_url = required_env("ALPACA_BASE_URL").rstrip("/")
    api_key = required_env("ALPACA_API_KEY_ID")
    secret_key = required_env("ALPACA_API_SECRET_KEY")

    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    async with httpx.AsyncClient(
        base_url=base_url,
        headers=headers,
        timeout=httpx.Timeout(60),
    ) as client:
        response = await client.get(
            "/v2/assets",
            params={"status": "active", "asset_class": "us_equity"},
        )
        response.raise_for_status()
        assets = response.json()

    if not isinstance(assets, list):
        raise TypeError("Alpaca /v2/assets response must be a list")
    tradable_assets = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("status") == "active"
        and asset.get("tradable") is True
    ]

    conn = await connect()
    try:
        await conn.execute(TABLE_SQL)
        async with conn.transaction():
            await conn.execute(f"TRUNCATE {SCHEMA_NAME}.alpaca_assets")
            await conn.executemany(
                f"""
                INSERT INTO {SCHEMA_NAME}.alpaca_assets (
                    asset_id, symbol, name, asset_class, exchange, status,
                    tradable, marginable, shortable, easy_to_borrow, fractionable,
                    maintenance_margin_requirement, attributes, raw_asset, updated_at
                )
                VALUES (
                    $1::UUID, $2, $3, $4, $5, $6,
                    $7, $8, $9, $10, $11,
                    $12, $13, $14::JSONB, NOW()
                )
                """,
                [asset_row(asset) for asset in tradable_assets],
            )
        stored = await conn.fetchval(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.alpaca_assets")
        print(f"[complete] Alpaca active tradable US equities stored={stored}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
