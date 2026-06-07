from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from database.backtesting.repositories.prices import (
    asset_metadata,
    price_is_covered,
    save_asset_metadata,
    save_price_bars,
)
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.worlds import run_world_assets
from main_backtesting.utils import chunks

from main_backtesting.stages.event_filter import accepted_markets, run_events


async def download_and_save_prices(
    self,
    conn: Any,
    requests: list[tuple[str, str, datetime, datetime]],
) -> None:
    pending: list[tuple[str, str, datetime, datetime]] = []
    for symbol, resolution, start, end in requests:
        if not await price_is_covered(
            conn, symbol=symbol, resolution=resolution, start=start, end=end
        ):
            pending.append((symbol, resolution, start, end))
    for batch in chunks(pending, self.config.price_download_concurrency):
        downloaded = await asyncio.gather(
            *[
                self.prices.bars(
                    symbol,
                    start=start,
                    end=end,
                    resolution=resolution,  # type: ignore[arg-type]
                )
                for symbol, resolution, start, end in batch
            ]
        )
        for request, bars in zip(batch, downloaded):
            symbol, resolution, start, end = request
            if not bars:
                raise ValueError(f"yfinance returned no {resolution} bars for {symbol}")
            await save_price_bars(
                conn,
                symbol=symbol,
                resolution=resolution,
                requested_start=start,
                requested_end=end,
                bars=bars,
            )


async def run(self, conn: Any) -> None:
    world_rows = list(await run_world_assets(conn, self.run_id))
    asset_symbols = sorted({row["symbol"] for row in world_rows})
    for symbol in asset_symbols:
        self.current_work_key = f"metadata:{symbol}"
        if await start_work(
            conn,
            run_id=self.run_id,
            stage="prices",
            work_key=self.current_work_key,
            payload={"symbol": symbol, "kind": "metadata"},
        ):
            if await asset_metadata(conn, symbol) is None:
                metadata = await self.prices.metadata(symbol)
                await save_asset_metadata(
                    conn,
                    symbol=symbol,
                    metadata=metadata,
                    missing_reason=None if metadata.get("sector") else "yfinance_sector_unavailable",
                )
            await finish_work(
                conn,
                run_id=self.run_id,
                stage="prices",
                work_key=self.current_work_key,
                result={},
            )
    sector_symbols = {
        row["sector_etf"]
        for symbol in asset_symbols
        if (row := await asset_metadata(conn, symbol)) is not None and row["sector_etf"]
    }
    events = {event.event_id: event for event in await run_events(self, conn)}
    markets = {market.market_id: market for market in await accepted_markets(self, conn)}
    requests: list[tuple[str, str, datetime, datetime]] = []
    first_by_event_asset: dict[tuple[str, str], Any] = {}
    for row in world_rows:
        key = (row["event_id"], row["symbol"])
        if key not in first_by_event_asset or row["as_of"] < first_by_event_asset[key]["as_of"]:
            first_by_event_asset[key] = row

        market = markets[row["market_id"]]
        resolution = "1h" if row["as_of"] >= self.hourly_boundary else "1d"
        trade_start = (
            max(market.created_at, self.hourly_boundary)
            if resolution == "1h"
            else market.created_at
        )
        trade_end = min(self.config.end, market.end_at)
        if trade_start < trade_end:
            requests.append((row["symbol"], resolution, trade_start, trade_end))

    for (event_id, symbol), row in first_by_event_asset.items():
        event = events[event_id]
        feature_year_start = datetime(row["as_of"].year, 1, 1, tzinfo=timezone.utc)
        daily_start = min(feature_year_start, event.created_at)
        daily_end = min(
            event.end_at + timedelta(days=1),
            self.config.historical_data_cutoff + timedelta(days=1),
        )
        metadata = await asset_metadata(conn, symbol)
        sector_etf = metadata["sector_etf"] if metadata else None
        for required_symbol in {symbol, "SPY", sector_etf} - {None}:
            requests.append((required_symbol, "1d", daily_start, daily_end))
        benchmark_start = row["as_of"] - timedelta(days=21)
        benchmark_end = row["as_of"] + timedelta(days=1)
        for benchmark_symbol in {"QQQ", "IWM", "TLT"}:
            requests.append(
                (benchmark_symbol, "1d", benchmark_start, benchmark_end)
            )

    requests = _merge_requests(requests)
    self.current_work_key = "all-price-bars"
    if await start_work(
        conn,
        run_id=self.run_id,
        stage="prices",
        work_key=self.current_work_key,
            payload={
                "asset_symbol_count": len(asset_symbols),
                "sector_symbol_count": len(sector_symbols),
                "request_count": len(requests),
            },
    ):
        await download_and_save_prices(self, conn, requests)
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="prices",
            work_key=self.current_work_key,
            result={"request_count": len(requests)},
        )


def _merge_requests(
    requests: list[tuple[str, str, datetime, datetime]],
) -> list[tuple[str, str, datetime, datetime]]:
    grouped: dict[tuple[str, str], list[tuple[datetime, datetime]]] = {}
    for symbol, resolution, start, end in requests:
        if start >= end:
            continue
        grouped.setdefault((symbol, resolution), []).append((start, end))

    merged: list[tuple[str, str, datetime, datetime]] = []
    for (symbol, resolution), windows in grouped.items():
        windows.sort()
        current_start, current_end = windows[0]
        for start, end in windows[1:]:
            if start <= current_end:
                current_end = max(current_end, end)
                continue
            merged.append((symbol, resolution, current_start, current_end))
            current_start, current_end = start, end
        merged.append((symbol, resolution, current_start, current_end))
    return sorted(merged)
