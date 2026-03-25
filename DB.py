from typing import List, Dict, Optional
import asyncpg


class DB:
    def __init__(self, connection_string: str):
        self.conn_str = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(dsn=self.conn_str, min_size=2, max_size=10)
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")

    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()

    # ── Signals ──────────────────────────────────────────────────────────────

    async def save_signals(self, signals: List[Dict]) -> bool:
        if not self.pool:
            raise RuntimeError("DB not connected")
        if not signals:
            return False
        rows = [
            (sig["signal_id"], sig["timestamp"], sig["strategy"], sig["symbol"],
             sig["action"], sig["price_reference"], sig.get("reason"), sig.get("metadata"))
            for sig in signals
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO signals
                    (signal_id, timestamp, strategy, symbol, action, price_reference, reason, metadata)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                ON CONFLICT (signal_id) DO NOTHING
                """,
                rows,
            )
        return True

    # ── Fills ─────────────────────────────────────────────────────────────────

    async def log_trade_execution(self, trade_data: Dict):
        if not self.pool:
            raise RuntimeError("DB not connected")
        if not trade_data:
            return False
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO fills
                    (fill_id, order_id, symbol, action, quantity, fill_price, filled_at, strategy, pair_id)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (fill_id) DO NOTHING
                """,
                [(
                    trade_data["fill_id"],
                    trade_data["order_id"],
                    trade_data["symbol"],
                    trade_data["action"],
                    trade_data["quantity"],
                    trade_data["fill_price"],
                    trade_data["filled_at"],
                    trade_data.get("strategy", ""),
                    trade_data.get("pair_id", None),
                )],
            )

    # ── Positions (live) ──────────────────────────────────────────────────────

    async def get_open_positions_from_db(self, strategy: str | None = None) -> List[Dict]:
        """
        Returns open positions.
        If `strategy` is provided, filters to that strategy only.
        Includes strategy and pair_id columns for ownership checks.
        """
        if not self.pool:
            raise RuntimeError("DB not connected")

        where_clause = "WHERE strategy = $1" if strategy else ""
        args = (strategy,) if strategy else ()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    symbol,
                    strategy,
                    pair_id,
                    SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) AS net_quantity
                FROM fills
                {where_clause}
                GROUP BY symbol, strategy, pair_id
                HAVING SUM(CASE WHEN action='BUY' THEN quantity ELSE -quantity END) != 0
                """,
                *args,
            )
        return [dict(row) for row in rows]

    # ── Orders ────────────────────────────────────────────────────────────────

    async def save_orders(self, orders: List[Dict]) -> bool:
        if not self.pool:
            raise RuntimeError("DB not connected")
        if not orders:
            return False
        rows = [
            (o["order_id"], o["signal_id"], o["strategy"], o["symbol"],
             o["action"], o["quantity"], o["order_type"], o["tif"], o["created_at"])
            for o in orders
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO orders
                    (order_id, signal_id, strategy, symbol, action,
                     quantity, order_type, tif, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                ON CONFLICT (order_id) DO NOTHING
                """,
                rows,
            )
        return True

    # ── Account snapshot ──────────────────────────────────────────────────────

    async def update_account_snapshot(self, nlv: float, cash: float):
        if not self.pool:
            raise RuntimeError("DB not connected")
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO account_snapshots (nlv, cash) VALUES ($1,$2)",
                nlv, cash,
            )

    # ── Recent bars ───────────────────────────────────────────────────────────

    async def get_recent_bars(self, ticker: str, lookback_days: int) -> List[Dict]:
        if not self.pool:
            raise RuntimeError("DB not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT ticker, date, open, high, low, close, volume
                FROM ohlcv_bars WHERE ticker=$1 ORDER BY date DESC LIMIT $2
                """,
                ticker, lookback_days,
            )
        return [
            {"ticker": r["ticker"], "date": r["date"],
             "Open": r["open"], "High": r["high"],
             "Low":  r["low"],  "Close": r["close"], "Volume": r["volume"]}
            for r in rows
        ]

    # ── Signals API ───────────────────────────────────────────────────────────

    async def get_signals_for_api(self, limit: int = 100) -> List[Dict]:
        if not self.pool:
            raise RuntimeError("DB not connected")
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT signal_id, timestamp, strategy, symbol, action, price_reference, reason
                FROM signals ORDER BY timestamp DESC LIMIT $1
                """,
                limit,
            )
        return [dict(r) for r in rows]

    # ── System flags ──────────────────────────────────────────────────────────

    async def get_system_flag(self, key: str) -> str | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("SELECT value FROM system_state WHERE key=$1", key)
        return row["value"] if row else None

    async def set_system_flag(self, key: str, value: str) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO system_state (key, value) VALUES ($1,$2)
                ON CONFLICT (key) DO UPDATE SET value=EXCLUDED.value
                """,
                key, value,
            )

    # ── Schema init + migrations ──────────────────────────────────────────────

    async def init_schema(self) -> None:
        if not self.pool:
            raise RuntimeError("DB not connected")
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_bars (
                    ticker  TEXT   NOT NULL,
                    date    DATE   NOT NULL,
                    open    FLOAT  NOT NULL,
                    high    FLOAT  NOT NULL,
                    low     FLOAT  NOT NULL,
                    close   FLOAT  NOT NULL,
                    volume  BIGINT NOT NULL,
                    PRIMARY KEY (ticker, date)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    signal_id       TEXT        PRIMARY KEY,
                    timestamp       TIMESTAMPTZ NOT NULL,
                    strategy        TEXT        NOT NULL,
                    symbol          TEXT        NOT NULL,
                    action          TEXT        NOT NULL,
                    price_reference FLOAT       NOT NULL,
                    reason          TEXT,
                    metadata        JSONB
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id   TEXT        PRIMARY KEY,
                    signal_id  TEXT        REFERENCES signals(signal_id),
                    strategy   TEXT        NOT NULL,
                    symbol     TEXT        NOT NULL,
                    action     TEXT        NOT NULL,
                    quantity   INT         NOT NULL,
                    order_type TEXT        NOT NULL,
                    tif        TEXT        NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS fills (
                    fill_id    TEXT        PRIMARY KEY,
                    order_id   TEXT        REFERENCES orders(order_id),
                    symbol     TEXT        NOT NULL,
                    action     TEXT        NOT NULL,
                    quantity   INT         NOT NULL,
                    fill_price FLOAT       NOT NULL,
                    filled_at  TIMESTAMPTZ NOT NULL
                );

                CREATE TABLE IF NOT EXISTS account_snapshots (
                    snapshot_id SERIAL      PRIMARY KEY,
                    nlv         FLOAT       NOT NULL,
                    cash        FLOAT       NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE TABLE IF NOT EXISTS system_state (
                    key   TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
            """)

            # ── Idempotent migrations ─────────────────────────────────────────
            # strategy + pair_id columns on fills: support ownership isolation
            await conn.execute("""
                ALTER TABLE fills ADD COLUMN IF NOT EXISTS action   TEXT  NOT NULL DEFAULT 'BUY';
                ALTER TABLE fills ADD COLUMN IF NOT EXISTS strategy TEXT  NOT NULL DEFAULT '';
                ALTER TABLE fills ADD COLUMN IF NOT EXISTS pair_id  TEXT;
            """)

    # ── OHLCV upsert ──────────────────────────────────────────────────────────

    async def upsert_ohlcv_bars(self, ticker: str, bars: List[Dict]) -> bool:
        if not self.pool:
            raise RuntimeError("DB not connected")
        if not bars:
            return False
        rows = [
            (ticker, b["date"], b["open"], b["high"], b["low"], b["close"], b["volume"])
            for b in bars
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv_bars (ticker,date,open,high,low,close,volume)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                ON CONFLICT (ticker, date) DO UPDATE
                    SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                        close=EXCLUDED.close, volume=EXCLUDED.volume
                """,
                rows,
            )
        return True
