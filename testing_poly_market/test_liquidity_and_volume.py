from polymarket_apis import PolymarketGammaClient


def check_liquidity_volume():
    token_id = "51508280778202349361616850684455231843716212176724253736363122559269229712002"

    print("=== Analyzing Liquidity & Volume (Gamma) ===")

    with PolymarketGammaClient() as gamma:
        try:
            markets = gamma.get_markets(
                token_ids=[token_id],
                limit=10,
            )

            if not markets:
                print("No Gamma markets found for this token_id.")
                return

            m = markets[0]

            print(f"Question    : {getattr(m, 'question', 'N/A')}")
            print(f"Slug        : {getattr(m, 'slug', 'N/A')}")
            print(f"Market ID   : {getattr(m, 'id', 'N/A')}")
            print(f"Active      : {getattr(m, 'active', 'N/A')}")
            print(f"Closed      : {getattr(m, 'closed', 'N/A')}")
            print(f"Liquidity   : {getattr(m, 'liquidity', 'N/A')}")
            print(f"Volume      : {getattr(m, 'volume', 'N/A')}")
            print(f"Condition ID: {getattr(m, 'conditionId', getattr(m, 'condition_id', 'N/A'))}")

            liquidity = getattr(m, "liquidity", None)
            volume = getattr(m, "volume", None)

            if liquidity is not None and volume is not None:
                liquidity = float(liquidity)
                volume = float(volume)
                ratio = volume / liquidity if liquidity > 0 else None

                print("\n--- Derived Metrics ---")
                print(f"Volume / Liquidity Ratio: {ratio:.4f}" if ratio is not None else "Volume / Liquidity Ratio: N/A")

        except Exception as e:
            print(f"Error executing Gamma liquidity/volume check: {e}")


if __name__ == "__main__":
    check_liquidity_volume()