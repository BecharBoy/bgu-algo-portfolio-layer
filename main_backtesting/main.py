from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from database.backtesting.historical_repository import historical_run
from database.backtesting.repository import candidate_events, event_markets
from database.backtesting.schema import initialize_historical_schema
from database.db_connection import connect
from main_backtesting.calibration import calibrate_batches
from main_backtesting.asset_selection_experiment import run_asset_selection_experiment
from main_backtesting.config import BacktestConfig
from main_backtesting.engine import STAGES, HistoricalBacktestEngine, purge_historical_run
from main_backtesting.reporting import generate_run_reports


def utc_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected date in YYYY-MM-DD format") from error


def add_smoke_date_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start", type=utc_date, default=utc_date("2026-01-01"))
    parser.add_argument(
        "--end",
        type=utc_date,
        default=utc_date("2026-02-01"),
        help="Exclusive end date.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full resumable historical backtest.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Create and execute a new historical backtest.")
    run.add_argument("--max-events", type=int, default=0)

    candidates = commands.add_parser(
        "list-candidates",
        help="List eligible markets without calling Ollama or downloading market data.",
    )
    add_smoke_date_arguments(candidates)
    candidates.add_argument("--limit", type=int, default=20)

    smoke = commands.add_parser(
        "smoke-test",
        help="Run a bounded backtest for one or more selected events.",
    )
    add_smoke_date_arguments(smoke)
    smoke.add_argument(
        "--event-id",
        action="append",
        default=[],
        help="Exact eligible event ID. Repeat to select multiple events.",
    )
    smoke.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Maximum events; defaults to one or to the number of selected event IDs.",
    )
    smoke.add_argument(
        "--through",
        choices=STAGES,
        default="event_filter",
        help="Stop safely after this stage. Resume later with the printed run ID.",
    )

    resume = commands.add_parser("resume", help="Resume an interrupted historical backtest.")
    resume.add_argument("--run-id", type=UUID, required=True)
    resume.add_argument(
        "--through",
        choices=STAGES,
        default=None,
        help="Stop safely after this stage instead of completing the entire run.",
    )

    report = commands.add_parser("report", help="Regenerate reports for an existing run.")
    report.add_argument("--run-id", type=UUID, required=True)

    purge = commands.add_parser("purge-run", help="Delete one run without deleting reusable data.")
    purge.add_argument("--run-id", type=UUID, required=True)

    commands.add_parser(
        "calibrate-batches",
        help="Test increasing Ollama batch sizes once and save the largest valid sizes.",
    )
    asset_selection_ab = commands.add_parser(
        "asset-selection-ab",
        help="Run a paired bounded A/B test of two LLM asset-selection methods.",
    )
    asset_selection_ab.add_argument("--source-run-id", type=UUID, required=True)
    asset_selection_ab.add_argument("--limit", type=int, default=1000)
    asset_selection_ab.add_argument("--seed", type=int, default=42)

    resume_asset_selection_ab = commands.add_parser(
        "resume-asset-selection-ab",
        help="Resume an interrupted paired LLM asset-selection A/B test.",
    )
    resume_asset_selection_ab.add_argument("--experiment-id", type=UUID, required=True)
    return parser


async def print_candidates(config: BacktestConfig, limit: int) -> None:
    conn = await connect()
    try:
        events = await candidate_events(
            conn,
            start=config.start,
            end=config.end,
            minimum_days_remaining=config.minimum_days_remaining,
            maximum_days_remaining=config.maximum_days_remaining,
            included_tags=sorted(config.included_tags),
            excluded_tags=sorted(config.excluded_tags),
            limit=limit,
        )
        market_count = 0
        for event in events:
            for market in await event_markets(conn, event):
                duration = (market.end_at - market.created_at).total_seconds() / 86_400
                if not (
                    config.minimum_days_remaining
                    < duration
                    <= config.maximum_days_remaining
                ):
                    continue
                print(
                    f"{market.market_id}\t{event.event_id}\t{market.created_at.date()}\t"
                    f"{market.end_at.date()}\t{market.question}"
                )
                market_count += 1
                if limit and market_count >= limit:
                    break
            if limit and market_count >= limit:
                break
        print(f"[candidate markets] count={market_count}")
    finally:
        await conn.close()


async def regenerate_report(run_id: UUID) -> None:
    conn = await connect()
    try:
        await initialize_historical_schema(conn)
        run = await historical_run(conn, run_id)
        await generate_run_reports(conn, run_id=run_id, run_dir=Path(run["output_dir"]))
    finally:
        await conn.close()


async def main_async() -> None:
    args = build_parser().parse_args()
    config = BacktestConfig()
    if args.command == "calibrate-batches":
        selected = await calibrate_batches(config)
        print(f"[calibration complete] {selected}")
        return
    if args.command == "asset-selection-ab":
        experiment_id = await run_asset_selection_experiment(
            source_run_id=args.source_run_id,
            experiment_id=None,
            query_limit=args.limit,
            sample_seed=args.seed,
        )
        print(f"[asset-selection A/B complete] experiment_id={experiment_id}")
        return
    if args.command == "resume-asset-selection-ab":
        experiment_id = await run_asset_selection_experiment(
            source_run_id=None,
            experiment_id=args.experiment_id,
            query_limit=1,
            sample_seed=0,
        )
        print(f"[asset-selection A/B complete] experiment_id={experiment_id}")
        return
    if args.command == "list-candidates":
        config = replace(config, start=args.start, end=args.end)
        await print_candidates(config, args.limit)
        return
    if args.command == "smoke-test":
        maximum_events = args.max_events
        if maximum_events is None:
            maximum_events = len(args.event_id) if args.event_id else 1
        if maximum_events < 1:
            raise ValueError("Smoke tests require --max-events of at least 1")
        config = replace(
            config,
            start=args.start,
            end=args.end,
            historical_data_cutoff=args.end,
            maximum_events=maximum_events,
            selected_event_ids=tuple(args.event_id),
        )
        engine = HistoricalBacktestEngine(config, stop_after_stage=args.through)
        run_id = await engine.run()
        print(f"[smoke test stopped after {args.through}] run_id={run_id}")
        return
    if args.command == "run":
        config = replace(config, maximum_events=args.max_events)
        engine = HistoricalBacktestEngine(config)
        run_id = await engine.run()
        print(f"[complete] run_id={run_id}")
        return
    if args.command == "resume":
        engine = HistoricalBacktestEngine(
            config,
            run_id=args.run_id,
            stop_after_stage=args.through,
        )
        run_id = await engine.run(resume=True)
        if args.through:
            print(f"[resume stopped after {args.through}] run_id={run_id}")
        else:
            print(f"[complete] run_id={run_id}")
        return
    if args.command == "report":
        await regenerate_report(args.run_id)
        print(f"[report complete] run_id={args.run_id}")
        return
    if args.command == "purge-run":
        await purge_historical_run(args.run_id)
        print(f"[purged] run_id={args.run_id}")


if __name__ == "__main__":
    asyncio.run(main_async())
