from DataFeed import DataFeed
from Portfolio import Portfolio


async def bootstrap_history_job(data_feed: DataFeed) -> None:
    await data_feed.bootstrap_two_year_history()


async def daily_incremental_update_job(data_feed: DataFeed) -> None:
    await data_feed.daily_incremental_update()


async def daily_trading_job(portfolio: Portfolio]) -> None:
    await portfolio.run_cycle()
