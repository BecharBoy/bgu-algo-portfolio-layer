from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.ingest_polymarket_1m_to_db import build_parser, run_ingestion


async def main_async() -> None:
    await run_ingestion(build_parser().parse_args())


if __name__ == "__main__":
    asyncio.run(main_async())
