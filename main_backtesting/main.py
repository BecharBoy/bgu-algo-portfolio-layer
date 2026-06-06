from __future__ import annotations

import argparse
import asyncio

from main_backtesting.config import BacktestConfig
from main_backtesting.engine import BacktestEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the January 2026 event-driven backtest.")
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Limit candidate events for a small integration run; zero means all.",
    )
    return parser


async def main_async() -> None:
    args = build_parser().parse_args()
    engine = BacktestEngine(BacktestConfig(), max_events=args.max_events)
    run_id = await engine.run()
    print(f"[complete] run_id={run_id}")


if __name__ == "__main__":
    asyncio.run(main_async())
