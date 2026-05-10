"""
Fetch and prepare Polymarket data for the information-diffusion-lag test.

This script produces the prediction-market side of the research dataset. It keeps
the old simple price CSVs for compatibility, but also writes signal panels that
match the paper's theory:

    delta_tau_t = P_PM(t) - P_PM(t - delta_t)

The resulting CSVs are intended to be merged with intraday equity returns from
`yahoo_fetch.py`.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from polymarket_apis import PolymarketDataClient, PolymarketReadOnlyClobClient

from event_specs import EVENT_SPECS, EVENT_WINDOW_AFTER, EVENT_WINDOW_BEFORE


OUTPUT_DIR = Path(".")

# The current event specs use 60-minute Polymarket history. If a future event has
# 5-minute history available, set its `fidelity` to 5 in event_specs.py and the
# same code will generate 5-minute deltas.
DEFAULT_FIDELITY_MINUTES = 60

# Probability-shock thresholds. These are research defaults, not final strategy
# parameters.
SIGNAL_DELTA_MINUTES = 60
DELTA_WINDOWS_MINUTES = {
    "delta_tau_1h": 60,
    "delta_tau_4h": 240,
    "delta_tau_1d": 1440,
}
GAMMA = 0.05
MIN_VOLUME_USDC = 100.0

# Data API pagination guard. Large markets can have many trades; this prevents an
# accidental infinite pull.
TRADE_PAGE_SIZE = 500
MAX_TRADE_PAGES = 100


def normalize_dt(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def normalize_series_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def event_research_window(event_key: str) -> tuple[datetime, datetime]:
    spec = EVENT_SPECS[event_key]
    event_ts = normalize_dt(spec["event_ts"])
    event_start = event_ts - EVENT_WINDOW_BEFORE
    event_end = event_ts + EVENT_WINDOW_AFTER

    poly = spec.get("poly")
    if not poly:
        return event_start, event_end

    formation_start = normalize_dt(poly["formation_start"])
    formation_end = normalize_dt(poly["formation_end"])
    return min(formation_start, event_start), max(formation_end, event_end)


def get_yes_token_id(clob: PolymarketReadOnlyClobClient, condition_id: str) -> str:
    market_info = clob.get_clob_market_info(condition_id)
    for token in market_info.tokens:
        if str(token.outcome).strip().lower() == "yes":
            return str(token.token_id)
    raise ValueError(f"No YES token found for condition_id={condition_id}")


def dedup_points(points: Iterable[object]) -> list[object]:
    dedup: dict[datetime, object] = {}
    for point in points:
        dedup[normalize_dt(point.timestamp)] = point
    return [dedup[key] for key in sorted(dedup)]


def fetch_price_history(
    clob: PolymarketReadOnlyClobClient,
    token_id: str,
    start: datetime,
    end: datetime,
    fidelity_minutes: int,
) -> pd.DataFrame:
    start = normalize_dt(start)
    end = normalize_dt(end)

    max_range = timedelta(days=15)
    chunk_start = start
    points: list[object] = []

    while chunk_start < end:
        chunk_end = min(chunk_start + max_range, end)
        history = clob.get_history(
            token_id=token_id,
            start_time=chunk_start,
            end_time=chunk_end,
            fidelity=fidelity_minutes,
        )
        points.extend(list(history.history or []))
        if chunk_end == end:
            break
        chunk_start = chunk_end

    if not points:
        all_history = clob.get_all_history(token_id=token_id)
        points = [
            point
            for point in list(all_history.history or [])
            if start <= normalize_dt(point.timestamp) <= end
        ]

    points = dedup_points(points)
    rows = [
        {
            "timestamp": normalize_dt(point.timestamp),
            "yes_price": float(point.value),
        }
        for point in points
        if start <= normalize_dt(point.timestamp) <= end
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "yes_price"])

    return df.sort_values("timestamp").drop_duplicates("timestamp")


def save_legacy_price_csv(df: pd.DataFrame, output_file: str) -> None:
    out = df[["timestamp", "yes_price"]].copy()
    out["timestamp"] = out["timestamp"].map(lambda ts: normalize_dt(ts).isoformat())
    out.to_csv(OUTPUT_DIR / output_file, index=False)


def load_existing_price_history(
    event_key: str,
    poly: dict,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    paths = [
        OUTPUT_DIR / f"polymarket_prices_{event_key}.csv",
        OUTPUT_DIR / poly["formation_csv"],
        OUTPUT_DIR / poly["event_csv"],
    ]
    frames: list[pd.DataFrame] = []

    for path in paths:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if not {"timestamp", "yes_price"}.issubset(df.columns):
            continue
        df = df[["timestamp", "yes_price"]].copy()
        df["timestamp"] = normalize_series_ts(df["timestamp"])
        df["yes_price"] = pd.to_numeric(df["yes_price"], errors="coerce")
        df = df.dropna(subset=["timestamp", "yes_price"])
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["timestamp", "yes_price"])

    start_ts = pd.Timestamp(normalize_dt(start))
    end_ts = pd.Timestamp(normalize_dt(end))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates("timestamp").sort_values("timestamp")
    combined = combined[(combined["timestamp"] >= start_ts) & (combined["timestamp"] <= end_ts)]
    return combined


def fetch_public_trades(
    data_client: PolymarketDataClient,
    condition_id: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    start = normalize_dt(start)
    end = normalize_dt(end)
    rows: list[dict[str, object]] = []

    for page in range(MAX_TRADE_PAGES):
        offset = page * TRADE_PAGE_SIZE
        try:
            trades = data_client.get_trades(
                limit=TRADE_PAGE_SIZE,
                offset=offset,
                taker_only=True,
                condition_id=condition_id,
            )
        except Exception as exc:
            print(f"  trade page failed at offset={offset}, keeping collected trades: {exc}")
            break

        if not trades:
            break

        page_times = [normalize_dt(trade.timestamp) for trade in trades]

        for trade in trades:
            ts = normalize_dt(trade.timestamp)
            if not start <= ts <= end:
                continue

            size = float(trade.size)
            price = float(trade.price)
            rows.append(
                {
                    "timestamp": ts,
                    "condition_id": str(trade.condition_id),
                    "token_id": str(trade.token_id),
                    "outcome": str(trade.outcome),
                    "side": str(trade.side),
                    "size": size,
                    "price": price,
                    "usdc_size": size * price,
                    "transaction_hash": str(trade.transaction_hash),
                }
            )

        # The data API normally returns newest trades first. Once an entire page
        # is older than the window we can stop. If ordering ever changes, the max
        # page guard still protects the script.
        if page_times and max(page_times) < start:
            break
        if len(trades) < TRADE_PAGE_SIZE:
            break

    if not rows:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "condition_id",
                "token_id",
                "outcome",
                "side",
                "size",
                "price",
                "usdc_size",
                "transaction_hash",
            ]
        )

    return pd.DataFrame(rows).sort_values("timestamp")


def aggregate_trade_flow(trades: pd.DataFrame, fidelity_minutes: int) -> pd.DataFrame:
    freq = f"{fidelity_minutes}min"
    columns = [
        "timestamp",
        "trade_count",
        "volume_usdc",
        "yes_buy_usdc",
        "yes_sell_usdc",
        "no_buy_usdc",
        "no_sell_usdc",
        "ofi_proxy_usdc",
    ]

    if trades.empty:
        return pd.DataFrame(columns=columns)

    df = trades.copy()
    df["timestamp"] = normalize_series_ts(df["timestamp"])
    df = df.dropna(subset=["timestamp"])
    df["bucket"] = df["timestamp"].dt.floor(freq)
    outcome = df["outcome"].str.lower().str.strip()
    side = df["side"].str.upper().str.strip()

    df["yes_buy_usdc"] = df["usdc_size"].where((outcome == "yes") & (side == "BUY"), 0.0)
    df["yes_sell_usdc"] = df["usdc_size"].where((outcome == "yes") & (side == "SELL"), 0.0)
    df["no_buy_usdc"] = df["usdc_size"].where((outcome == "no") & (side == "BUY"), 0.0)
    df["no_sell_usdc"] = df["usdc_size"].where((outcome == "no") & (side == "SELL"), 0.0)

    grouped = (
        df.groupby("bucket")
        .agg(
            trade_count=("usdc_size", "size"),
            volume_usdc=("usdc_size", "sum"),
            yes_buy_usdc=("yes_buy_usdc", "sum"),
            yes_sell_usdc=("yes_sell_usdc", "sum"),
            no_buy_usdc=("no_buy_usdc", "sum"),
            no_sell_usdc=("no_sell_usdc", "sum"),
        )
        .reset_index()
        .rename(columns={"bucket": "timestamp"})
    )

    # Proxy for YES-side order flow imbalance. Buying YES and selling NO are
    # bullish for the YES probability; selling YES and buying NO are bearish.
    grouped["ofi_proxy_usdc"] = (
        grouped["yes_buy_usdc"]
        - grouped["yes_sell_usdc"]
        - grouped["no_buy_usdc"]
        + grouped["no_sell_usdc"]
    )
    return grouped[columns]


def build_signal_panel(
    event_key: str,
    condition_id: str,
    yes_token_id: str,
    price_df: pd.DataFrame,
    trade_flow: pd.DataFrame,
    fidelity_minutes: int,
) -> pd.DataFrame:
    spec = EVENT_SPECS[event_key]
    freq = f"{fidelity_minutes}min"

    if price_df.empty:
        raise ValueError(f"No Polymarket price history for {event_key}")

    prices = price_df.copy()
    prices["timestamp"] = normalize_series_ts(prices["timestamp"])
    prices = prices.dropna(subset=["timestamp"]).sort_values("timestamp")
    prices = prices.set_index("timestamp")["yes_price"].astype(float)

    # Regularize the probability series so delta_tau is comparable across events.
    panel = prices.resample(freq).last().ffill().to_frame("prob_yes")
    panel = panel.reset_index()

    for column, minutes in DELTA_WINDOWS_MINUTES.items():
        periods = max(1, round(minutes / fidelity_minutes))
        panel[column] = panel["prob_yes"].diff(periods)

    signal_periods = max(1, round(SIGNAL_DELTA_MINUTES / fidelity_minutes))
    panel["delta_tau_signal"] = panel["prob_yes"].diff(signal_periods)
    panel["abs_delta_tau_signal"] = panel["delta_tau_signal"].abs()
    panel["signal_direction"] = panel["delta_tau_signal"].apply(
        lambda value: 1 if value > 0 else (-1 if value < 0 else 0)
    )
    panel["valid_probability_signal"] = panel["abs_delta_tau_signal"] >= GAMMA

    if trade_flow.empty:
        panel["trade_count"] = 0
        panel["volume_usdc"] = 0.0
        panel["yes_buy_usdc"] = 0.0
        panel["yes_sell_usdc"] = 0.0
        panel["no_buy_usdc"] = 0.0
        panel["no_sell_usdc"] = 0.0
        panel["ofi_proxy_usdc"] = 0.0
        panel["volume_data_available"] = False
        panel["valid_volume_signal"] = False
        panel["valid_signal"] = panel["valid_probability_signal"]
    else:
        flow = trade_flow.copy()
        flow["timestamp"] = normalize_series_ts(flow["timestamp"])
        panel = panel.merge(flow, on="timestamp", how="left")
        flow_cols = [
            "trade_count",
            "volume_usdc",
            "yes_buy_usdc",
            "yes_sell_usdc",
            "no_buy_usdc",
            "no_sell_usdc",
            "ofi_proxy_usdc",
        ]
        panel[flow_cols] = panel[flow_cols].fillna(0.0)
        panel["trade_count"] = panel["trade_count"].astype(int)
        panel["volume_data_available"] = True
        panel["valid_volume_signal"] = panel["volume_usdc"] >= MIN_VOLUME_USDC
        panel["valid_signal"] = panel["valid_probability_signal"] & panel["valid_volume_signal"]

    panel.insert(0, "event", event_key)
    panel.insert(1, "event_ts", normalize_dt(spec["event_ts"]).isoformat())
    panel.insert(2, "condition_id", condition_id)
    panel.insert(3, "yes_token_id", yes_token_id)
    panel.insert(4, "fidelity_minutes", fidelity_minutes)
    panel["gamma"] = GAMMA
    panel["min_volume_usdc"] = MIN_VOLUME_USDC
    panel["timestamp"] = panel["timestamp"].map(lambda ts: normalize_dt(ts).isoformat())
    return panel


def fetch_event(
    clob: PolymarketReadOnlyClobClient,
    data_client: PolymarketDataClient,
    event_key: str,
) -> pd.DataFrame | None:
    spec = EVENT_SPECS[event_key]
    poly = spec.get("poly")
    if not poly:
        print(f"{event_key}: no Polymarket config, skipping")
        return None

    condition_id = poly["condition_id"]
    fidelity_minutes = int(poly.get("fidelity", DEFAULT_FIDELITY_MINUTES))
    research_start, research_end = event_research_window(event_key)

    print(f"\n[{event_key}]")
    print(f"  condition_id: {condition_id}")
    print(f"  research window: {research_start.isoformat()} -> {research_end.isoformat()}")
    print(f"  fidelity: {fidelity_minutes} minutes")

    try:
        yes_token_id = get_yes_token_id(clob, condition_id)
        print(f"  yes_token_id: {yes_token_id}")
    except Exception as exc:
        yes_token_id = ""
        print(f"  CLOB market lookup failed, trying existing CSVs: {exc}")

    if yes_token_id:
        full_prices = fetch_price_history(
            clob=clob,
            token_id=yes_token_id,
            start=research_start,
            end=research_end,
            fidelity_minutes=fidelity_minutes,
        )
    else:
        full_prices = pd.DataFrame(columns=["timestamp", "yes_price"])

    if full_prices.empty:
        fallback_prices = load_existing_price_history(event_key, poly, research_start, research_end)
        if not fallback_prices.empty:
            full_prices = fallback_prices
            print(f"  loaded existing price rows from CSV fallback: {len(full_prices)}")

    print(f"  price rows: {len(full_prices)}")

    if full_prices.empty:
        print("  no price history found")
        return None

    # Keep the original output names used by older scripts.
    formation_prices = full_prices[
        (full_prices["timestamp"] >= normalize_dt(poly["formation_start"]))
        & (full_prices["timestamp"] <= normalize_dt(poly["formation_end"]))
    ]
    event_start = normalize_dt(spec["event_ts"]) - EVENT_WINDOW_BEFORE
    event_end = normalize_dt(spec["event_ts"]) + EVENT_WINDOW_AFTER
    event_prices = full_prices[
        (full_prices["timestamp"] >= event_start)
        & (full_prices["timestamp"] <= event_end)
    ]
    save_legacy_price_csv(formation_prices, poly["formation_csv"])
    save_legacy_price_csv(event_prices, poly["event_csv"])

    raw_price_path = OUTPUT_DIR / f"polymarket_prices_{event_key}.csv"
    save_legacy_price_csv(full_prices, str(raw_price_path))

    try:
        trades = fetch_public_trades(data_client, condition_id, research_start, research_end)
    except Exception as exc:
        print(f"  trade fetch failed, continuing without volume data: {exc}")
        trades = pd.DataFrame()

    if not trades.empty:
        trades_out = trades.copy()
        trades_out["timestamp"] = trades_out["timestamp"].map(lambda ts: normalize_dt(ts).isoformat())
        trades_out.to_csv(OUTPUT_DIR / f"polymarket_trades_{event_key}.csv", index=False)
    print(f"  trade rows in window: {len(trades)}")

    trade_flow = aggregate_trade_flow(trades, fidelity_minutes)
    signal_panel = build_signal_panel(
        event_key=event_key,
        condition_id=condition_id,
        yes_token_id=yes_token_id,
        price_df=full_prices,
        trade_flow=trade_flow,
        fidelity_minutes=fidelity_minutes,
    )
    signal_panel.to_csv(OUTPUT_DIR / f"polymarket_signal_{event_key}.csv", index=False)
    print(f"  signal rows: {len(signal_panel)}")

    return signal_panel


def main() -> None:
    event_keys = [event_key for event_key, spec in EVENT_SPECS.items() if spec.get("poly")]
    panels: list[pd.DataFrame] = []

    data_client = PolymarketDataClient()
    try:
        with PolymarketReadOnlyClobClient() as clob:
            for event_key in event_keys:
                try:
                    panel = fetch_event(clob, data_client, event_key)
                except Exception as exc:
                    print(f"{event_key}: failed: {exc}")
                    continue
                if panel is not None and not panel.empty:
                    panels.append(panel)
    finally:
        data_client.client.close()

    if panels:
        combined = pd.concat(panels, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "polymarket_signal_panel.csv", index=False)
        print(f"\nSaved combined signal panel: polymarket_signal_panel.csv ({len(combined)} rows)")
    else:
        print("\nNo Polymarket panels were produced")


if __name__ == "__main__":
    main()
