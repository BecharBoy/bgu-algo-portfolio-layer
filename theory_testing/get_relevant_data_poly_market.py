"""
Polymarket Historical Data Fetcher — Simple & Direct
"""

import csv
from datetime import UTC, datetime, timedelta

from polymarket_apis import (
    PolymarketReadOnlyClobClient,
)
from event_specs import EVENT_SPECS, EVENT_WINDOW_BEFORE, EVENT_WINDOW_AFTER


def normalize_dt(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def save_points_to_csv(points: list, output_file: str) -> None:
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "yes_price"])
        writer.writeheader()
        for point in points:
            ts = normalize_dt(point.timestamp)
            writer.writerow({
                "timestamp": ts.isoformat(),
                "yes_price": float(point.value),
            })


def filter_points_to_window(points: list, start: datetime, end: datetime) -> list:
    start = normalize_dt(start)
    end = normalize_dt(end)
    out = [p for p in points if start <= normalize_dt(p.timestamp) <= end]
    out.sort(key=lambda p: normalize_dt(p.timestamp))
    return out


def dedup_points(points: list) -> list:
    dedup = {}
    for p in points:
        ts = normalize_dt(p.timestamp)
        dedup[ts] = p
    return [dedup[k] for k in sorted(dedup.keys())]


def fetch_prices_history(clob, token_id: str, start: datetime, end: datetime, fidelity: int, output_csv: str) -> list:
    print(f"--- Prices History | {start.date()} → {end.date()} ---")

    start = normalize_dt(start)
    end = normalize_dt(end)
    max_range = timedelta(days=15)
    total_range = end - start

    if total_range <= max_range:
        history = clob.get_history(token_id=token_id, start_time=start, end_time=end, fidelity=fidelity)
        points = list(history.history or [])
    else:
        chunk_points = []
        chunk_start = start
        while chunk_start < end:
            chunk_end = min(chunk_start + max_range, end)
            history = clob.get_history(token_id=token_id, start_time=chunk_start, end_time=chunk_end, fidelity=fidelity)
            chunk_points.extend(list(history.history or []))
            if chunk_end == end:
                break
            chunk_start = chunk_end
        points = dedup_points(chunk_points)

    points = filter_points_to_window(points, start, end)

    if not points:
        all_history = clob.get_all_history(token_id=token_id)
        points = filter_points_to_window(list(all_history.history or []), start, end)

    print(f"  Points received: {len(points)}")

    if not points:
        print("  No history points found")
        return []

    prices = [float(p.value) for p in points]
    times = [normalize_dt(p.timestamp) for p in points]

    print(f"  First: {times[0].isoformat()} → {prices[0]:.4f}")
    print(f"  Last:  {times[-1].isoformat()} → {prices[-1]:.4f}")
    print(f"  Range: {min(prices):.4f} – {max(prices):.4f}")

    save_points_to_csv(points, output_csv)
    print(f"  Saved: {output_csv}\n")

    return points


def fetch_all_windows_for_event(clob, event_key: str):
    spec = EVENT_SPECS[event_key]
    poly = spec.get("poly")

    if not poly:
        print(f"⚠️  {event_key}: no poly data, skipping")
        return

    event_ts = spec["event_ts"]
    condition_id = poly["condition_id"]

    print(f"\n>>> {event_key.upper()} <<<")
    print(f"Condition ID: {condition_id}")

    # Get YES token from condition_id
    yes_token_id = None
    try:
        market_info = clob.get_clob_market_info(condition_id)
        for tok in market_info.tokens:
            if str(tok.outcome).strip().lower() == "yes":
                yes_token_id = str(tok.token_id)
                print(f"YES token: {yes_token_id}\n")
                break
    except Exception as e:
        print(f"❌ Error getting token for {event_key}: {e}")
        return

    if not yes_token_id:
        print(f"❌ No YES token found for {event_key}")
        return

    # Formation period
    formation_start = normalize_dt(poly["formation_start"])
    formation_end = normalize_dt(poly["formation_end"])

    print(f"📊 Formation Period: {formation_start.date()} → {formation_end.date()}")
    fetch_prices_history(
        clob=clob,
        token_id=yes_token_id,
        start=formation_start,
        end=formation_end,
        fidelity=poly["fidelity"],
        output_csv=poly["formation_csv"],
    )

    # Event window
    event_start = normalize_dt(event_ts - EVENT_WINDOW_BEFORE)
    event_end = normalize_dt(event_ts + EVENT_WINDOW_AFTER)

    print(f"⚡ Event Window: {event_start.date()} → {event_end.date()}")
    fetch_prices_history(
        clob=clob,
        token_id=yes_token_id,
        start=event_start,
        end=event_end,
        fidelity=poly["fidelity"],
        output_csv=poly["event_csv"],
    )


def main():
    # All events with poly data (2024 + 2025)
    POLY_EVENT_KEYS = [
        # 2024
        "cpi_hot_april",
        "jobs_aug_unemployment",
        "fomc_cut_sep",
        # 2025
        "fomc_jan_2025",
        "cpi_mar_2025",
        "nfp_may_2025",
        "fomc_jul_2025",
    ]

    with PolymarketReadOnlyClobClient() as clob:
        for event_key in POLY_EVENT_KEYS:
            try:
                fetch_all_windows_for_event(clob, event_key)
            except Exception as e:
                print(f"❌ Failed on {event_key}: {e}\n")
                continue

    print("\n✅ All events processed!")


if __name__ == "__main__":
    main()