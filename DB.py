from typing import List, Dict
import asyncpg

class DB:
    def __init__(self, connection_string: str):
        """ setting up connection to the data base """
        self.conn_str = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            dsn = self.conn_str,
            min_size = 2,
            max_size = 10
        )
        async with self.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        
    async def disconnect(self) -> None:
        if self.pool:
            await self.pool.close()


    async def save_signals(self, signals: List[Dict]) -> bool:
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")
        
        if not signals:
            return False

        rows = [
            (
                sig["signal_id"],
                sig["timestamp"],
                sig["strategy"],
                sig["symbol"],
                sig["action"],
                sig["price_reference"],
                sig.get("reason"), 
                sig.get("metadata"),      
            )
            for sig in signals
        ]
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO signals
                    (signal_id, timestamp, strategy, symbol, action, price_reference, reason, metadata)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (signal_id) DO NOTHING
                """,
                rows
            )
            


    async def log_trade_execution(self, trade_data: Dict):
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")
        
        if not trade_data:
            return False
        
        rows = [(
        trade_data["fill_id"],
        trade_data["order_id"],
        trade_data["symbol"],
        trade_data["quantity"],
        trade_data["fill_price"],
        trade_data["filled_at"],   
        )]
        async with self.pool.acquire() as conn:
            await conn.executemany(
            """
            INSERT INTO fills
                (fill_id, order_id, symbol, quantity, fill_price, filled_at)
            VALUES
                ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (order_id) DO NOTHING
            """,
            rows
            )

    async def update_account_snapshot(self, nlv: float, cash: float):
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")
        
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO account_snapshots
                    (nlv, cash)
                VALUES
                    ($1, $2)
                """,
                nlv, cash
            )

    async def get_open_positions_from_db(self):
        if self.pool is None:
           raise RuntimeError("DB not connected — call connect() first")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT symbol, SUM(quantity) as net_quantity
                FROM fills
                GROUP BY symbol
                HAVING SUM(quantity) != 0
                """
            )
        return [dict(row) for row in rows]

    async def init_schema(self) -> None:
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv_bars (
                    ticker      TEXT        NOT NULL,
                    date        DATE        NOT NULL,
                    open        FLOAT       NOT NULL,
                    high        FLOAT       NOT NULL,
                    low         FLOAT       NOT NULL,
                    close       FLOAT       NOT NULL,
                    volume      BIGINT      NOT NULL,
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
                    order_id    TEXT        PRIMARY KEY,
                    signal_id   TEXT        REFERENCES signals(signal_id),
                    strategy    TEXT        NOT NULL,
                    symbol      TEXT        NOT NULL,
                    action      TEXT        NOT NULL,
                    quantity    INT         NOT NULL,
                    order_type  TEXT        NOT NULL,
                    tif         TEXT        NOT NULL,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
    
                CREATE TABLE IF NOT EXISTS fills (
                    fill_id     TEXT        PRIMARY KEY,
                    order_id    TEXT        REFERENCES orders(order_id),
                    symbol      TEXT        NOT NULL,
                    quantity    INT         NOT NULL,
                    fill_price  FLOAT       NOT NULL,
                    filled_at   TIMESTAMPTZ NOT NULL
                );
    
                CREATE TABLE IF NOT EXISTS account_snapshots (
                    snapshot_id SERIAL      PRIMARY KEY,
                    nlv         FLOAT       NOT NULL,
                    cash        FLOAT       NOT NULL,
                    recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
            """)

    async def upsert_ohlcv_bars(self, ticker: str, bars: List[Dict]):
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")
        
        if not bars:
            return False
        
        rows = [
            (   ticker,
                bar["date"],
                bar["open"],
                bar["high"],
                bar["low"],
                bar["close"],
                bar["volume"],      
            )
            for bar in bars
        ]
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
            """
            INSERT INTO ohlcv_bars
                (ticker, date, open, high, low, close, volume)
            VALUES
                ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (ticker, date) DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
                close=EXCLUDED.close, volume=EXCLUDED.volume
            """,
            rows
            )


    async def get_recent_bars(self, ticker: str, lookback_days: int) -> List[Dict]:
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
            """
            SELECT ticker, date, open, high, low, close, volume
            FROM ohlcv_bars
            WHERE ticker = $1
            ORDER BY date DESC
            LIMIT $2
            """,
            ticker, lookback_days
        )
    return [dict(row) for row in rows]
        return [dict(row) for row in rows]

    async def save_orders(self, orders: List[Dict]):
        if self.pool is None:
            raise RuntimeError("DB not connected — call connect() first")

        if not orders:
            return False

        rows = [
            (
                order["order_id"],
                order["signal_id"],
                order["strategy"],
                order["symbol"],
                order["action"],
                order["quantity"],
                order["order_type"],
                order["tif"],
                order["created_at"]
            )
            for order in orders
        ]
        
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO orders
                    (order_id, signal_id, strategy, symbol, action, quantity, order_type, tif, created_at)
                VALUES
                    ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (order_id) DO NOTHING
                """,
                rows
            )

    async def get_signals_for_api(self, limit: int = 100) -> List[Dict]:
         if self.pool is None:
           raise RuntimeError("DB not connected — call connect() first")

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT signal_id, timestamp, strategy, symbol, action, price_reference, reason
                FROM signals
                ORDER BY timestamp DESC
                LIMIT $1
                """,
                limit
            )
        return [dict(row) for row in rows]
