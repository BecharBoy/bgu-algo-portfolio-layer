import asyncio
from ib_async import IB


async def test():
    ib = IB()
    await ib.connectAsync('127.0.0.1', 4002, clientId=1)
    print("Connected:", ib.isConnected())

    accounts = ib.managedAccounts()
    print("Accounts:", accounts)

    await asyncio.sleep(1)
    ib.disconnect()
    print("Done")


asyncio.run(test())