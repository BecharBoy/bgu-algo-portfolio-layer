from typing import List


async def bootstrap_history_job(data_feed, db_client, tickers: List[str]) -> None:
    # TODO: Download two-year history for universe and upsert into DB.
    # TODO: Make job idempotent so reruns are safe.
    pass


async def daily_incremental_update_job(data_feed, db_client, tickers: List[str]) -> None:
    # TODO: Fetch latest completed daily bar for each ticker.
    # TODO: Upsert bars and verify expected row counts.
    pass


async def daily_trading_job(portfolio, tickers: List[str]) -> None:
    # TODO: Trigger one portfolio run cycle after data update completes.
    # TODO: Support dry-run mode before enabling broker order placement.
    pass
