from datetime import UTC, datetime, timedelta
from polymarket_apis import PolymarketReadOnlyClobClient


FIVE_MIN = 5 * 60


def floor_to_5m(dt: datetime) -> datetime:
    ts = int(dt.timestamp())
    floored = ts - (ts % FIVE_MIN)
    return datetime.fromtimestamp(floored, tz=UTC)


def ceil_to_next_5m(dt: datetime) -> datetime:
    ts = int(dt.timestamp())
    ceiled = ((ts + FIVE_MIN - 1) // FIVE_MIN) * FIVE_MIN
    return datetime.fromtimestamp(ceiled, tz=UTC)


def check_5m_timestamp_sync():
    print("=== Checking 5-Minute Timestamp Synchronization ===")

    with PolymarketReadOnlyClobClient() as clob:
        server_time = clob.get_utc_time()

        bucket_start = floor_to_5m(server_time)
        bucket_end = bucket_start + timedelta(minutes=5)
        next_poll = ceil_to_next_5m(server_time)

        print(f"Polymarket Server UTC : {server_time.isoformat()}")
        print(f"Current 5m Bucket Start: {bucket_start.isoformat()}")
        print(f"Current 5m Bucket End  : {bucket_end.isoformat()}")
        print(f"Next Poll Time         : {next_poll.isoformat()}")

        lag_seconds = int((server_time - bucket_start).total_seconds())
        print(f"Seconds Since Bucket Start: {lag_seconds}")


if __name__ == "__main__":
    check_5m_timestamp_sync()