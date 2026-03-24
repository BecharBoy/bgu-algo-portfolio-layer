from dataclasses import dataclass
from typing import List
import os
import pandas as pd

@dataclass
class Settings:
    ib_host: str
    ib_port: int
    ib_client_id: int
    db_connection_string: str
    gemini_api_key: str
    universe: List[str]
    trade_mode: str = "paper"
    max_concurrent_positions: int = 10



def load_settings() -> Settings:
    return Settings(
        ib_host=os.environ.get("IB_HOST", "127.0.0.1"),
        ib_port=int(os.environ["IB_PORT"]),
        ib_client_id=int(os.environ["IB_CLIENT_ID"]),
        db_connection_string=os.environ["DB_CONNECTION_STRING"],
        gemini_api_key=os.environ["GEMINI_API_KEY"],
        universe=_load_universe(),
    )


def _load_universe(path: str = "tickers.csv") -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found — add your tickers before running")
    with open(path) as f:
        tickers = [line.strip() for line in f if line.strip()]
    if not tickers:
        raise ValueError(f"{path} is empty — populate your trading universe first")
    return tickers
