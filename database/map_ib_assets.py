from __future__ import annotations

import asyncio
import csv
import io
import os
import sys
from pathlib import Path
from typing import Any

import asyncpg
import httpx
from dotenv import load_dotenv
from ib_async import IB, Stock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.db_connection import connect

ENV_PATH = REPO_ROOT / ".env"
SCHEMA_NAME = "checking_relevant_events"
TARGET_TABLE = "ib_assets"
STAGING_TABLE = "ib_assets_staging"
NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
WRITE_BATCH_SIZE = 250
REQUEST_DELAY_SECONDS = 0.10

TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{TARGET_TABLE} (
    con_id             BIGINT PRIMARY KEY,
    symbol             TEXT NOT NULL,
    local_symbol       TEXT,
    trading_class      TEXT,
    security_name      TEXT NOT NULL,
    long_name          TEXT,
    primary_exchange   TEXT,
    currency           TEXT NOT NULL,
    valid_exchanges    TEXT[] NOT NULL,
    order_types        TEXT[] NOT NULL,
    industry           TEXT,
    category           TEXT,
    subcategory        TEXT,
    stock_type         TEXT,
    source_exchange    TEXT,
    is_etf             BOOLEAN NOT NULL,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ib_assets_symbol
    ON {SCHEMA_NAME}.{TARGET_TABLE}(symbol);

CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{STAGING_TABLE}
    (LIKE {SCHEMA_NAME}.{TARGET_TABLE} INCLUDING ALL);
"""


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def ib_symbol(symbol: str) -> str:
    return symbol.replace(".", " ").replace("$", " ")


async def download_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


def parse_pipe_file(text: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="|"))
    return [
        row
        for row in rows
        if row
        and not any((value or "").startswith("File Creation Time") for value in row.values())
    ]


async def candidate_symbols() -> dict[str, dict[str, Any]]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
        nasdaq_text, other_text = await asyncio.gather(
            download_text(client, NASDAQ_LISTED_URL),
            download_text(client, OTHER_LISTED_URL),
        )

    candidates: dict[str, dict[str, Any]] = {}
    for row in parse_pipe_file(nasdaq_text):
        symbol = (row.get("Symbol") or "").strip()
        if not symbol or row.get("Test Issue") == "Y":
            continue
        candidates[symbol] = {
            "symbol": symbol,
            "security_name": (row.get("Security Name") or "").strip(),
            "source_exchange": "NASDAQ",
            "is_etf": row.get("ETF") == "Y",
        }

    for row in parse_pipe_file(other_text):
        symbol = (row.get("ACT Symbol") or "").strip()
        if not symbol or row.get("Test Issue") == "Y":
            continue
        candidates[symbol] = {
            "symbol": symbol,
            "security_name": (row.get("Security Name") or "").strip(),
            "source_exchange": (row.get("Exchange") or "").strip(),
            "is_etf": row.get("ETF") == "Y",
        }
    return candidates


def choose_us_smart_contract(details: list[Any]) -> Any | None:
    matches = []
    for detail in details:
        contract = detail.contract
        valid_exchanges = split_csv(detail.validExchanges)
        order_types = split_csv(detail.orderTypes)
        if contract.secType != "STK" or contract.currency != "USD":
            continue
        if "SMART" not in valid_exchanges or not order_types:
            continue
        matches.append(detail)
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            item.contract.primaryExchange not in {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"},
            item.contract.conId,
        )
    )
    return matches[0]


def asset_row(candidate: dict[str, Any], detail: Any) -> tuple[Any, ...]:
    contract = detail.contract
    return (
        contract.conId,
        contract.symbol,
        contract.localSymbol or None,
        contract.tradingClass or None,
        candidate["security_name"],
        detail.longName or None,
        contract.primaryExchange or None,
        contract.currency,
        split_csv(detail.validExchanges),
        split_csv(detail.orderTypes),
        detail.industry or None,
        detail.category or None,
        detail.subcategory or None,
        detail.stockType or None,
        candidate["source_exchange"],
        candidate["is_etf"],
    )


async def write_rows(
    conn: asyncpg.Connection,
    table_name: str,
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    await conn.executemany(
        f"""
        INSERT INTO {SCHEMA_NAME}.{table_name} (
            con_id, symbol, local_symbol, trading_class, security_name,
            long_name, primary_exchange, currency, valid_exchanges, order_types,
            industry, category, subcategory, stock_type, source_exchange,
            is_etf, updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5,
            $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15,
            $16, NOW()
        )
        ON CONFLICT (con_id) DO UPDATE SET
            symbol = EXCLUDED.symbol,
            local_symbol = EXCLUDED.local_symbol,
            trading_class = EXCLUDED.trading_class,
            security_name = EXCLUDED.security_name,
            long_name = EXCLUDED.long_name,
            primary_exchange = EXCLUDED.primary_exchange,
            currency = EXCLUDED.currency,
            valid_exchanges = EXCLUDED.valid_exchanges,
            order_types = EXCLUDED.order_types,
            industry = EXCLUDED.industry,
            category = EXCLUDED.category,
            subcategory = EXCLUDED.subcategory,
            stock_type = EXCLUDED.stock_type,
            source_exchange = EXCLUDED.source_exchange,
            is_etf = EXCLUDED.is_etf,
            updated_at = NOW()
        """,
        rows,
    )


async def main() -> None:
    load_dotenv(ENV_PATH)
    ib_host = required_env("IB_HOST")
    ib_port = int(required_env("IB_PORT"))
    ib_client_id = int(required_env("IB_CLIENT_ID"))

    candidates = await candidate_symbols()
    print(f"[candidates] US-listed symbols={len(candidates)}")

    conn = await connect()
    ib = IB()
    await conn.execute(TABLE_SQL)
    try:
        await ib.connectAsync(ib_host, ib_port, clientId=ib_client_id, timeout=20)
        await conn.execute(f"TRUNCATE {SCHEMA_NAME}.{STAGING_TABLE}")
        pending_rows: list[tuple[Any, ...]] = []
        confirmed = 0

        for index, candidate in enumerate(candidates.values(), start=1):
            contract = Stock(ib_symbol(candidate["symbol"]), "SMART", "USD")
            details = await ib.reqContractDetailsAsync(contract)
            selected = choose_us_smart_contract(details)
            if selected is not None:
                pending_rows.append(asset_row(candidate, selected))
                confirmed += 1

            if len(pending_rows) >= WRITE_BATCH_SIZE:
                await write_rows(conn, STAGING_TABLE, pending_rows)
                pending_rows.clear()

            if index % 100 == 0:
                print(f"[progress] checked={index}/{len(candidates)} confirmed={confirmed}")
            await asyncio.sleep(REQUEST_DELAY_SECONDS)

        await write_rows(conn, STAGING_TABLE, pending_rows)
        async with conn.transaction():
            await conn.execute(f"TRUNCATE {SCHEMA_NAME}.{TARGET_TABLE}")
            await conn.execute(
                f"""
                INSERT INTO {SCHEMA_NAME}.{TARGET_TABLE}
                SELECT * FROM {SCHEMA_NAME}.{STAGING_TABLE}
                """
            )
        stored = await conn.fetchval(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{TARGET_TABLE}")
        print(f"[complete] IB-confirmed US stocks/ETFs stored={stored}")
    finally:
        if ib.isConnected():
            ib.disconnect()
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
