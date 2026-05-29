from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import asyncpg
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
REQUIRED_DB_ENV = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


def db_config() -> dict[str, Any]:
    load_dotenv(ENV_PATH)
    missing = [name for name in REQUIRED_DB_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing database environment variables: {', '.join(missing)}")
    return {
        "host": os.environ["DB_HOST"],
        "port": int(os.environ["DB_PORT"]),
        "database": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


async def connect() -> asyncpg.Connection:
    return await asyncpg.connect(**db_config())


async def create_pool(min_size: int = 1, max_size: int = 10) -> asyncpg.Pool:
    return await asyncpg.create_pool(**db_config(), min_size=min_size, max_size=max_size)
