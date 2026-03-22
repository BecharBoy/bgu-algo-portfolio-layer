import asyncio
import logging

from config import load_settings
from DB import DB
from Data_Feed import DataFeed
from IB import IB_Connect
from Portfolio import Portfolio
from jobs import bootstrap_history_job, daily_incremental_update_job, daily_trading_job
from strategies.mean_reversion.meanreversion import MeanReversionMomentum
from strategies.cointegration.StatArbStrategy import StatArbStrategy


async def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )

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
    portfolio.add_strategy(MeanReversionMomentum(weight_allocation=0.5))
    portfolio.add_strategy(StatArbStrategy(weight_allocation=0.5))

    await bootstrap_history_job(data_feed, db)
    await daily_incremental_update_job(data_feed)
    await daily_trading_job(portfolio)

    ib.disconnect()
    await db.disconnect()


if __name__ == "__main__":
    asyncio.run(run())
