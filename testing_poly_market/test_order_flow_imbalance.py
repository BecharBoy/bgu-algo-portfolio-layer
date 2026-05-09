from datetime import UTC, datetime, timedelta
from polymarket_apis import PolymarketReadOnlyClobClient, PolymarketDataClient


def _to_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromtimestamp(int(value), tz=UTC)


def check_item_2_final():
    token_id = "51508280778202349361616850684455231843716212176724253736363122559269229712002"

    print("=== Analyzing Trade Tape (Item 2) ===")

    after_time = datetime.now(UTC) - timedelta(hours=12)

    with (
        PolymarketReadOnlyClobClient() as clob,
        PolymarketDataClient() as data_client,
    ):
        try:
            market_ids = clob.get_market_ids_from_token(token_id)
            condition_id = market_ids.condition_id

            market_info = clob.get_clob_market_info(condition_id)

            target_outcome = None
            for tok in market_info.tokens:
                if str(tok.token_id) == token_id:
                    target_outcome = str(tok.outcome).strip().lower()
                    break

            trades = []
            offset = 0
            limit = 500

            while True:
                batch = data_client.get_trades(
                    condition_id=condition_id,
                    taker_only=True,
                    limit=limit,
                    offset=offset,
                )

                if not batch:
                    break

                reached_old_data = False

                for t in batch:
                    ts = _to_datetime(t.timestamp)

                    if ts < after_time:
                        reached_old_data = True
                        continue

                    same_token = str(getattr(t, "token_id", "")) == token_id
                    same_outcome = str(getattr(t, "outcome", "")).strip().lower() == target_outcome

                    if same_token or same_outcome:
                        trades.append(t)

                if len(batch) < limit or reached_old_data:
                    break

                offset += limit

            if not trades:
                print("No trades found in this window.")
                return

            print(f"Retrieved {len(trades)} individual trades.\n")

            whale_threshold = 5000
            buy_vol = 0.0
            sell_vol = 0.0

            for t in trades:
                price = float(t.price)
                size = float(t.size)
                side = str(t.side).upper()
                ts = _to_datetime(t.timestamp)

                if side == "BUY":
                    buy_vol += size
                else:
                    sell_vol += size

                if size >= whale_threshold:
                    print(
                        f"[WHALE DETECTED] Side: {side} | "
                        f"Size: {size:,.0f} | Price: {price:.4f} | "
                        f"Outcome: {t.outcome} | Time: {ts.isoformat()}"
                    )

            ofi = buy_vol - sell_vol

            print(f"\n--- Statistics for OFI Calculation ---")
            print(f"Total Buy Volume : {buy_vol:,.0f}")
            print(f"Total Sell Volume: {sell_vol:,.0f}")
            print(f"Order Flow Imbalance (OFI): {ofi:,.0f}")

        except Exception as e:
            print(f"Error executing trade tape analysis: {e}")


if __name__ == "__main__":
    check_item_2_final()