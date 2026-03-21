import asyncio

from config import load_settings
from DB import DB
from Data_Feed import Data_Feed
from IB import IB_Connect
from Portfolio import Portfolio
from jobs import bootstrap_history_job, daily_incremental_update_job, daily_trading_job
from strategies.mean_reversion.meanreversion import MeanReversionMomentum
from strategies.cointegration.StatArbStrategy import StatArbStrategy


async def run() -> None:
    # TODO: Central app bootstrap:
    # TODO: 1) load config
    # TODO: 2) create clients
    # TODO: 3) register strategies
    # TODO: 4) execute daily workflow
    settings = load_settings()

    db_client = DB(connection_string=settings.db_connection_string)
    data_feed = Data_Feed()
    ib_client = IB_Connect(settings.ib_host, settings.ib_port, settings.ib_client_id)

    portfolio = Portfolio(
        ib_client=ib_client,
        db_client=db_client,
        data_feed=data_feed,
        total_capital=0.0,  # TODO: Pull starting capital from account snapshot/settings.
    )
    portfolio.add_strategy("mean_reversion", MeanReversionMomentum(weight_allocation=0.5))
    portfolio.add_strategy("cointegration_arb", StatArbStrategy(weight_allocation=0.5))

    # TODO: Run bootstrap only when DB history is missing.
    await bootstrap_history_job(data_feed, db_client, settings.universe)
    await daily_incremental_update_job(data_feed, db_client, settings.universe)
    await daily_trading_job(portfolio, settings.universe)


if __name__ == "__main__":
    # TODO: Add structured logging and process-level exception handling.
    asyncio.run(run())
