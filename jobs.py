from Data_Feed import DataFeed
from Portfolio import Portfolio
from DB import DB
import logging
from datetime import date

async def bootstrap_history_job(data_feed: DataFeed, db: DB) -> None:
    already_done = await db.get_system_flag("bootstrap_done")
    if already_done == "true":
        logging.info("bootstrap_history_job: history already bootstrapped, skipping.")
        return
    await data_feed.bootstrap_two_year_history()
    await db.set_system_flag("bootstrap_done", "true")


async def daily_incremental_update_job(data_feed: DataFeed) -> None:
    if date.today().weekday() >= 5:  # 5=Saturday, 6=Sunday
        logging.info("daily_incremental_update_job: weekend — skipping.")
        return
    await data_feed.daily_incremental_update()

async def daily_trading_job(portfolio: Portfolio) -> None:
    if date.today().weekday() >= 5:
        logging.info("daily_trading_job: weekend — skipping.")
        return
    await portfolio.run_cycle()
