import asyncio
import logging
import os
from datetime import date

from config import load_settings
from DB import DB
from Data_Feed import DataFeed
from IB import IB_Connect
from Portfolio import Portfolio
from jobs import bootstrap_history_job, daily_incremental_update_job, daily_trading_job
from strategies.mean_reversion.meanreversion import MeanReversionMomentum
from strategies.cointegration.StatArbStrategy import StatArbStrategy
import csv


def load_tickers(path: str) -> list[str]:
    with open(path, "r") as f:
        return [row[0].strip() for row in csv.reader(f) if row and row[0].strip()]


def setup_logging() -> None:
    """
    Configures logging to write to both stdout and a daily log file.
    Each run produces logs/YYYY-MM-DD.log — a new file per calendar day.
    """
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{date.today().strftime('%Y-%m-%d')}.log"

    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%H:%M:%S"

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(console)

    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
    root.addHandler(file_handler)

    logging.info(f"Logging to {log_filename}")


async def run() -> None:
    setup_logging()

    settings = load_settings()

    db = DB(connection_string=settings.db_connection_string)
    await db.connect()
    await db.init_schema()

    data_feed = DataFeed(settings=settings, db=db)

    ib = IB_Connect(
        host=settings.ib_host,
        port=settings.ib_port,
        client_id=settings.ib_client_id,
    )
    await ib.connect()

    portfolio = Portfolio(
        settings=settings,
        db=db,
        ib=ib,
        datafeed=data_feed,
    )
    portfolio.add_strategy(MeanReversionMomentum(capital_allocation=0.5))
    portfolio.add_strategy(StatArbStrategy(capital_allocation=0.5))

    await bootstrap_history_job(data_feed, db)
    await daily_incremental_update_job(data_feed)
    await daily_trading_job(portfolio)

    ib.disconnect()
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run())
