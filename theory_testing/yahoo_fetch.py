"""
yahoo_fetch.py — Rate-Sensitive Assets around Macro Events
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from event_specs import EVENT_SPECS, event_window


LONG_CANDIDATES = ["DHI", "LEN", "ENPH", "RUN", "CVNA", "TSLA", "UPST", "IWM"]
SHORT_CANDIDATES = ["BAC", "JPM", "WFC", "KRE", "PGR", "ALL", "KO", "PG", "PEP"]
BENCHMARKS = ["SPY", "QQQ", "TLT"]
ALL_TICKERS = LONG_CANDIDATES + SHORT_CANDIDATES + BENCHMARKS

WINDOW_START = "2024-03-01"
WINDOW_END = "2025-09-01"

EVENT_KEYS = [
    # 2024
    "cpi_hot_april",
    "jobs_aug_unemployment",
    "fomc_cut_sep",
    "cpi_cool_july",

    # 2025
    "fomc_jan_2025",
    "cpi_mar_2025",
    "nfp_may_2025",
    "fomc_jul_2025",
]


def normalize_idx(idx):
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def get_price_at(df, target_dt, direction="before"):
    if direction == "before":
        sub = df[df.index <= target_dt]
        return sub.iloc[-1] if not sub.empty else None
    sub = df[df.index > target_dt]
    return sub.iloc[0] if not sub.empty else None


def hourly_refetch_supported(event_dt: pd.Timestamp) -> bool:
    now = pd.Timestamp.utcnow().tz_localize(None)
    oldest_supported = now - pd.Timedelta(days=729)
    return event_dt >= oldest_supported


def fetch_or_load_hourly(event_key: str) -> pd.DataFrame:
    spec = EVENT_SPECS[event_key]
    event_dt = pd.Timestamp(spec["event_ts"].replace(tzinfo=None))
    csv_path = Path(spec["yahoo"]["hourly_csv"])

    if csv_path.exists():
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df.index = normalize_idx(df.index)
        return df.sort_index()

    if not hourly_refetch_supported(event_dt):
        raise ValueError(f"Hourly refetch not supported for {event_key} ({event_dt}). Missing: {csv_path}")

    win_start, win_end = event_window(spec["event_ts"])
    download_start = win_start.strftime("%Y-%m-%d")
    download_end = (win_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(
        tickers=ALL_TICKERS,
        start=download_start,
        end=download_end,
        interval="1h",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError(f"Yahoo returned no hourly data for {event_key}")

    if "Close" not in df:
        raise ValueError(f"Yahoo hourly data missing Close for {event_key}")

    close = df["Close"].copy()
    close.index = normalize_idx(close.index)

    if close.empty:
        raise ValueError(f"Yahoo Close frame empty for {event_key}")

    close.to_csv(csv_path)
    return close.sort_index()


def main():
    print("Fetching daily data...")
    daily = yf.download(
        ALL_TICKERS,
        start=WINDOW_START,
        end=WINDOW_END,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
    )

    if daily.empty or "Close" not in daily:
        raise ValueError("Yahoo daily download failed or returned no Close data")

    close_daily = daily["Close"].copy()
    close_daily.index = pd.to_datetime(close_daily.index).tz_localize(None)
    normalized = (close_daily / close_daily.iloc[0]) * 100
    normalized.to_csv("daily_normalized.csv")
    print(f"Daily: {len(close_daily)} rows × {len(close_daily.columns)} tickers\n")

    results = []
    skipped = []

    for event_key in EVENT_KEYS:
        spec = EVENT_SPECS[event_key]
        event_dt = pd.Timestamp(spec["event_ts"].replace(tzinfo=None))
        event_type = spec["event_type"]

        try:
            close = fetch_or_load_hourly(event_key)
        except Exception as e:
            print(f"⚠️ Skipping {event_key}: {e}\n")
            skipped.append((event_key, str(e)))
            continue

        p0_row = get_price_at(close, event_dt, "before")
        p1d_row = get_price_at(close, event_dt + pd.Timedelta(days=1), "after")
        p3d_row = get_price_at(close, event_dt + pd.Timedelta(days=3), "after")

        print(f"[{event_key}] {event_dt} ({event_type})")
        print(f"{'Ticker':<8} {'P0':>8} {'P+1d':>9} {'Ret1d':>7} {'P+3d':>9} {'Ret3d':>7}")
        print("-" * 52)

        for ticker in ALL_TICKERS:
            if ticker not in close.columns or p0_row is None or pd.isna(p0_row[ticker]):
                continue

            p0 = float(p0_row[ticker])
            p1 = float(p1d_row[ticker]) if p1d_row is not None and pd.notna(p1d_row[ticker]) else None
            p3 = float(p3d_row[ticker]) if p3d_row is not None and pd.notna(p3d_row[ticker]) else None

            if p3d_row is not None and p1d_row is not None and p3d_row.name == p1d_row.name:
                p3 = None

            r1 = f"{(p1 / p0 - 1) * 100:+.2f}%" if p1 is not None else "N/A"
            r3 = f"{(p3 / p0 - 1) * 100:+.2f}%" if p3 is not None else "N/A"
            p1s = f"{p1:.2f}" if p1 is not None else "N/A"
            p3s = f"{p3:.2f}" if p3 is not None else "N/A"

            print(f"{ticker:<8} {p0:>8.2f} {p1s:>9} {r1:>7} {p3s:>9} {r3:>7}")

            results.append({
                "event": event_key,
                "event_type": event_type,
                "date": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "p0": p0,
                "p1d": p1,
                "p3d": p3,
                "ret_1d": (p1 / p0 - 1) * 100 if p1 is not None else None,
                "ret_3d": (p3 / p0 - 1) * 100 if p3 is not None else None,
            })

        print()

    summary = pd.DataFrame(results)
    summary.to_csv("event_returns_summary.csv", index=False)

    print("Done. Files saved:")
    print("  daily_normalized.csv")
    print("  event_returns_summary.csv")
    print("  hourly_<event>.csv")

    if skipped:
        print("\nSkipped events:")
        for event_key, reason in skipped:
            print(f"  - {event_key}: {reason}")


if __name__ == "__main__":
    main()