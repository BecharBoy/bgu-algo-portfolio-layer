import asyncio
from datetime import datetime
from ib_async import IB, Stock


def fmt(v):
    if v is None:
        return None
    try:
        if isinstance(v, float) and (v != v):  # nan
            return None
    except Exception:
        pass
    return v


async def main():
    ib = IB()

    try:
        await ib.connectAsync("127.0.0.1", 4002, clientId=7, timeout=10)
        print("Connected:", ib.isConnected())
        print("Accounts:", ib.managedAccounts())

        # Google (Alphabet Class C)
        contract = Stock("GOOG", "SMART", "USD")

        qualified = await ib.qualifyContractsAsync(contract)
        if not qualified:
            raise RuntimeError("Could not qualify GOOG contract")

        contract = qualified[0]
        print(f"Qualified contract: conId={contract.conId}, symbol={contract.symbol}, exchange={contract.exchange}")

        # Request market data (Level 1)
        ticker = ib.reqMktData(contract, genericTickList="", snapshot=False, regulatorySnapshot=False)

        # Wait for first updates
        for _ in range(10):
            await asyncio.sleep(1)
            if any([
                ticker.bid is not None,
                ticker.ask is not None,
                ticker.last is not None,
                ticker.close is not None
            ]):
                break

        data = {
            "symbol": contract.symbol,
            "conId": contract.conId,
            "time": datetime.now().isoformat(timespec="seconds"),

            "bid": fmt(ticker.bid),
            "bidSize": fmt(ticker.bidSize),
            "ask": fmt(ticker.ask),
            "askSize": fmt(ticker.askSize),

            "last": fmt(ticker.last),
            "lastSize": fmt(ticker.lastSize),
            "prevClose": fmt(ticker.close),

            "open": fmt(ticker.open),
            "high": fmt(ticker.high),
            "low": fmt(ticker.low),
            "volume": fmt(ticker.volume),

            "bidExchange": fmt(getattr(ticker, "bidExchange", None)),
            "askExchange": fmt(getattr(ticker, "askExchange", None)),
            "lastExchange": fmt(getattr(ticker, "lastExchange", None)),

            "halted": fmt(getattr(ticker, "halted", None)),
            "rtTime": str(fmt(getattr(ticker, "rtTime", None))),
            "rtVolume": fmt(getattr(ticker, "rtVolume", None)),
            "vwap": fmt(getattr(ticker, "vwap", None)),
            "marketDataType": fmt(getattr(ticker, "marketDataType", None)),
        }

        print("\n=== GOOG Level 1 ===")
        for k, v in data.items():
            print(f"{k:15}: {v}")

        # Keep listening a bit longer to see updates
        print("\nListening for updates for 15 seconds...")
        for i in range(15):
            await asyncio.sleep(1)
            print(
                f"[{i+1:02d}s] "
                f"bid={fmt(ticker.bid)} x {fmt(ticker.bidSize)} | "
                f"ask={fmt(ticker.ask)} x {fmt(ticker.askSize)} | "
                f"last={fmt(ticker.last)} x {fmt(ticker.lastSize)} | "
                f"vol={fmt(ticker.volume)}"
            )

        ib.cancelMktData(contract)

    finally:
        if ib.isConnected():
            ib.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    asyncio.run(main())

