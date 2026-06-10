from __future__ import annotations

import asyncio
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from database.backtesting.market_data import benchmark_symbol
from database.backtesting.security_master import SecurityMaster, download_security_master_entries
from database.backtesting.repositories.prices import (
    asset_metadata,
    missing_price_windows,
    save_asset_metadata,
    save_price_bars,
)
from database.backtesting.repositories import json_value
from database.backtesting.repositories.runs import finish_work, start_work
from database.backtesting.repositories.security_master import (
    run_asset_resolutions,
    save_run_asset_resolution,
    save_security_master_entries,
    security_master_entries,
)
from database.backtesting.repositories.worlds import run_resolved_world_assets, run_world_assets
from main_backtesting.utils import chunks

from main_backtesting.stages.event_filter import accepted_markets, run_events

MAX_EMPTY_PRICE_RETRIES = 2
PRICE_PROGRESS_INTERVAL = 100


async def load_security_master(conn: Any) -> SecurityMaster:
    entries = await security_master_entries(conn)
    if not entries:
        entries = await download_security_master_entries()
        await save_security_master_entries(conn, entries)
        print(f"[security master] downloaded and stored symbols={len(entries)}")
    else:
        print(f"[security master] loaded cached symbols={len(entries)}")
    return SecurityMaster(entries)


def write_rejected_asset_log(path: Path, rows: list[Any]) -> None:
    rejected = [row for row in rows if row["resolved_symbol"] is None]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["original_symbol", "rejection_reason"],
        )
        writer.writeheader()
        for row in rejected:
            writer.writerow(
                {
                    "original_symbol": row["original_symbol"],
                    "rejection_reason": row["rejection_reason"],
                }
            )


async def download_and_save_prices(
    self,
    conn: Any,
    requests: list[tuple[str, str, datetime, datetime]],
) -> dict[str, int]:
    pending: list[tuple[str, str, datetime, datetime]] = []
    for symbol, resolution, start, end in requests:
        for missing_start, missing_end in await missing_price_windows(
            conn, symbol=symbol, resolution=resolution, start=start, end=end
        ):
            pending.append((symbol, resolution, missing_start, missing_end))
    completed = 0
    empty_windows = 0
    retryable_failures = 0
    for batch in chunks(pending, self.config.price_download_concurrency):
        async def download_with_empty_retry(
            request: tuple[str, str, datetime, datetime],
        ) -> tuple[list[Any], str | None]:
            symbol, resolution, start, end = request
            for attempt in range(MAX_EMPTY_PRICE_RETRIES + 1):
                try:
                    bars = await self.prices.bars(
                        symbol,
                        start=start,
                        end=end,
                        resolution=resolution,  # type: ignore[arg-type]
                    )
                except Exception as error:
                    if attempt == MAX_EMPTY_PRICE_RETRIES:
                        return [], str(error)
                    await asyncio.sleep(2 ** attempt)
                    continue
                if bars or attempt == MAX_EMPTY_PRICE_RETRIES:
                    return bars, None
            return [], "price download attempts exhausted"

        downloaded = await asyncio.gather(
            *[download_with_empty_retry(request) for request in batch]
        )
        for request, (bars, error) in zip(batch, downloaded):
            symbol, resolution, start, end = request
            if error is not None:
                retryable_failures += 1
                print(
                    f"[prices] retryable failure symbol={symbol} resolution={resolution} "
                    f"start={start.date()} end={end.date()} error={error}"
                )
                await save_price_bars(
                    conn,
                    symbol=symbol,
                    resolution=resolution,
                    requested_start=start,
                    requested_end=end,
                    bars=[],
                    status="retryable_failure",
                    error=error,
                )
                completed += 1
                continue
            if not bars:
                empty_windows += 1
                print(
                    f"[prices] no data after {MAX_EMPTY_PRICE_RETRIES + 1} attempts "
                    f"symbol={symbol} resolution={resolution} "
                    f"start={start.date()} end={end.date()}"
                )
            await save_price_bars(
                conn,
                symbol=symbol,
                resolution=resolution,
                requested_start=start,
                requested_end=end,
                bars=bars,
                status="complete" if bars else "no_data",
            )
            completed += 1
        if completed % PRICE_PROGRESS_INTERVAL < len(batch) or completed == len(pending):
            print(
                f"[prices] completed={completed}/{len(pending)} "
                f"no_data_windows={empty_windows}"
            )
    return {
        "pending_request_count": len(pending),
        "downloaded_window_count": completed,
        "no_data_window_count": empty_windows,
        "retryable_failure_count": retryable_failures,
    }


async def run(self, conn: Any) -> None:
    raw_world_rows = list(await run_world_assets(conn, self.run_id))
    master = await load_security_master(conn)
    names_by_symbol: dict[str, set[str]] = {}
    for row in raw_world_rows:
        names_by_symbol.setdefault(row["symbol"], set()).add(row["asset_name"])
    for original_symbol in sorted(names_by_symbol):
        resolution = master.resolve(
            original_symbol,
            asset_names=sorted(names_by_symbol[original_symbol]),
        )
        await save_run_asset_resolution(
            conn,
            run_id=self.run_id,
            resolution=resolution,
        )
    resolution_rows = list(await run_asset_resolutions(conn, self.run_id))
    rejected_count = sum(row["resolved_symbol"] is None for row in resolution_rows)
    write_rejected_asset_log(
        self.run_dir / "logs" / "rejected_asset_symbols.csv",
        resolution_rows,
    )
    print(
        f"[asset validation] original={len(resolution_rows)} "
        f"resolved={len(resolution_rows) - rejected_count} rejected={rejected_count}"
    )

    world_rows = list(await run_resolved_world_assets(conn, self.run_id))
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
            existing_metadata = await asset_metadata(conn, symbol)
            if existing_metadata is None:
                metadata = await self.prices.metadata(symbol)
                await save_asset_metadata(
                    conn,
                    symbol=symbol,
                    metadata=metadata,
                    missing_reason=None if metadata.get("sector") else "yfinance_sector_unavailable",
                )
            elif existing_metadata["benchmark_symbol"] is None:
                metadata = dict(json_value(existing_metadata["metadata"]))
                metadata.setdefault("asset_name", existing_metadata["asset_name"])
                metadata.setdefault("sector", existing_metadata["sector"])
                metadata.setdefault("sector_etf", existing_metadata["sector_etf"])
                metadata["benchmark_symbol"] = benchmark_symbol(
                    symbol,
                    quote_type=metadata.get("quote_type"),
                    sector=metadata.get("sector"),
                )
                await save_asset_metadata(
                    conn,
                    symbol=symbol,
                    metadata=metadata,
                    missing_reason=existing_metadata["missing_reason"],
                )
            await finish_work(
                conn,
                run_id=self.run_id,
                stage="prices",
                work_key=self.current_work_key,
                result={},
            )
    benchmark_symbols = {
        row["benchmark_symbol"]
        for symbol in asset_symbols
        if (row := await asset_metadata(conn, symbol)) is not None
        and row["benchmark_symbol"]
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
        warmup_days = 14 if resolution == "1h" else 45
        trade_start = market.created_at - timedelta(days=warmup_days)
        if resolution == "1h":
            trade_start = max(trade_start, self.hourly_boundary)
        trade_end = market.end_at + timedelta(days=4)
        if trade_start < trade_end:
            requests.append((row["symbol"], resolution, trade_start, trade_end))
        if resolution == "1h":
            requests.append(
                (
                    row["symbol"],
                    "1d",
                    row["as_of"] - timedelta(days=45),
                    trade_end,
                )
            )

    for (event_id, symbol), row in first_by_event_asset.items():
        event = events[event_id]
        feature_year_start = datetime(row["as_of"].year, 1, 1, tzinfo=timezone.utc)
        daily_start = min(
            feature_year_start - timedelta(days=40),
            row["as_of"] - timedelta(days=40),
            event.created_at,
        )
        daily_end = min(
            event.end_at + timedelta(days=1),
            self.config.historical_data_cutoff + timedelta(days=1),
        )
        metadata = await asset_metadata(conn, symbol)
        asset_benchmark = metadata["benchmark_symbol"] if metadata else None
        for required_symbol in {symbol, "SPY", asset_benchmark} - {None}:
            requests.append((required_symbol, "1d", daily_start, daily_end))
        benchmark_start = row["as_of"] - timedelta(days=21)
        benchmark_end = row["as_of"] + timedelta(days=1)
        for reference_symbol in {"QQQ", "IWM", "TLT"}:
            requests.append(
                (reference_symbol, "1d", benchmark_start, benchmark_end)
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
                "sector_symbol_count": len(benchmark_symbols),
                "request_count": len(requests),
            },
    ):
        result = await download_and_save_prices(self, conn, requests)
        await finish_work(
            conn,
            run_id=self.run_id,
            stage="prices",
            work_key=self.current_work_key,
            result={"request_count": len(requests), **result},
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
