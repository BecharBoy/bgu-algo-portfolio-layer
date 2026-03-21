from dataclasses import dataclass
from typing import List


@dataclass
class Settings:
    ib_host: str
    ib_port: int
    ib_client_id: int
    db_connection_string: str
    gemini_api_key: str
    trade_mode: str
    universe: List[str]


def load_settings() -> Settings:
    # TODO: Load values from environment variables and/or .env file.
    # TODO: Validate required secrets and fail fast with clear errors.
    # TODO: Parse universe list from config source.
    pass
