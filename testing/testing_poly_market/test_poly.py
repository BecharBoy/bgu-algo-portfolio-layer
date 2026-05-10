from polymarket_apis import PolymarketReadOnlyClobClient


def check_order_book_depth():
    token_id = "51508280778202349361616850684455231843716212176724253736363122559269229712002"

    print("=== Analyzing Order Book Depth ===")

    with PolymarketReadOnlyClobClient() as clob:
        try:
            book = clob.get_order_book(token_id)

            bids = list(getattr(book, "bids", []) or [])
            asks = list(getattr(book, "asks", []) or [])

            if not bids and not asks:
                print("Order book is empty.")
                return

            bids = sorted(bids, key=lambda x: float(x.price), reverse=True)
            asks = sorted(asks, key=lambda x: float(x.price))

            best_bid = float(bids[0].price) if bids else None
            best_ask = float(asks[0].price) if asks else None
            spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else None

            top_n = 5
            bid_depth = sum(float(level.size) for level in bids[:top_n])
            ask_depth = sum(float(level.size) for level in asks[:top_n])
            total_depth = bid_depth + ask_depth
            depth_imbalance = ((bid_depth - ask_depth) / total_depth) if total_depth > 0 else 0.0

            print(f"Best Bid          : {best_bid if best_bid is not None else 'N/A'}")
            print(f"Best Ask          : {best_ask if best_ask is not None else 'N/A'}")
            print(f"Spread            : {spread if spread is not None else 'N/A'}")
            print(f"Top-{top_n} Bid Depth : {bid_depth:,.2f}")
            print(f"Top-{top_n} Ask Depth : {ask_depth:,.2f}")
            print(f"Depth Imbalance   : {depth_imbalance:.4f}")
            print()

            print("--- Top Bids ---")
            for level in bids[:top_n]:
                print(f"Price={float(level.price):.4f} | Size={float(level.size):,.2f}")

            print("\n--- Top Asks ---")
            for level in asks[:top_n]:
                print(f"Price={float(level.price):.4f} | Size={float(level.size):,.2f}")

        except Exception as e:
            print(f"Error executing get_order_book: {e}")


if __name__ == "__main__":
    check_order_book_depth()