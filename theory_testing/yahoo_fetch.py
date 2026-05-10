"""
Fetch and prepare Yahoo Finance data for the information-diffusion-lag test.

The old version of this script only created event-window prices and static
1-day/3-day returns. The paper needs a richer dataset:

1. Intraday equity returns over the prediction-market research window.
2. Benchmark returns used to estimate market beta.
3. Optional alignment with Polymarket probability shocks.
4. Preliminary beta_M and beta_tau estimates for each stock.

Run `get_relevant_data_poly_market.py` first if you want the aligned macro-equity
panels and beta_tau estimates.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from event_specs import EVENT_SPECS, EVENT_WINDOW_AFTER, EVENT_WINDOW_BEFORE, event_window


OUTPUT_DIR = Path(".")

# Yahoo has strict historical limits on intraday bars. For old 2024/2025 events,
# 1h is the practical default. You can try YAHOO_INTERVAL=5m for recent events.
REQUESTED_INTERVAL = os.getenv("YAHOO_INTERVAL", "1h")
FALLBACK_INTERVALS = ["1h", "1d"]

MARKET_BENCHMARK = "SPY"
BENCHMARKS = ["SPY", "QQQ", "IWM", "TLT"]

# Paper-style rate-shock candidates. During a rate hike, the first group should
# be structurally hurt; during a rate cut, it should benefit. The second group is
# expected to be more resilient or positively exposed to higher rates.
RATE_HIKE_SHORT_CANDIDATES = ["RGTI", "TSLA", "UPST", "CVNA", "RUN", "ENPH", "DHI", "LEN"]
RATE_HIKE_LONG_CANDIDATES = ["GOOG", "GOOGL", "MSFT", "AAPL", "JPM", "BAC", "WFC", "PGR", "ALL", "KO", "PG", "PEP"]

TRADABLE_UNIVERSE = sorted(set(RATE_HIKE_SHORT_CANDIDATES + RATE_HIKE_LONG_CANDIDATES))
ALL_TICKERS = sorted(set(TRADABLE_UNIVERSE + BENCHMARKS))

EVENT_KEYS = list(EVENT_SPECS.keys())
MIN_REGRESSION_OBS = 20


def to_naive_utc(dt: datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC").tz_localize(None)
    return ts


def datetime64_ns(series: pd.Series, utc: bool = False) -> pd.Series:
    parsed = pd.to_datetime(series, utc=utc, errors="coerce")
    if utc:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    return parsed.astype("datetime64[ns]")


def normalize_idx(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.to_datetime(idx)
    if getattr(out, "tz", None) is not None:
        out = out.tz_convert("UTC").tz_localize(None)
    return pd.DatetimeIndex(out).astype("datetime64[ns]")


def interval_minutes(interval: str) -> int:
    value = int(interval[:-1])
    unit = interval[-1].lower()
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported interval: {interval}")


def bars_per_trading_day(interval: str) -> int:
    minutes = interval_minutes(interval)
    if minutes >= 1440:
        return 1
    return max(1, round(390 / minutes))


def candidate_intervals() -> list[str]:
    intervals = [REQUESTED_INTERVAL]
    for interval in FALLBACK_INTERVALS:
        if interval not in intervals:
            intervals.append(interval)
    return intervals


def event_research_window(event_key: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    spec = EVENT_SPECS[event_key]
    event_ts = to_naive_utc(spec["event_ts"])
    starts = [event_ts - pd.Timedelta(EVENT_WINDOW_BEFORE)]
    ends = [event_ts + pd.Timedelta(EVENT_WINDOW_AFTER)]

    poly = spec.get("poly")
    if poly:
        starts.append(to_naive_utc(poly["formation_start"]))
        ends.append(to_naive_utc(poly["formation_end"]))

    return min(starts), max(ends)


def get_price_at(df: pd.DataFrame, target_dt: pd.Timestamp, direction: str = "before") -> pd.Series | None:
    if direction == "before":
        sub = df[df.index <= target_dt]
        return sub.iloc[-1] if not sub.empty else None
    sub = df[df.index > target_dt]
    return sub.iloc[0] if not sub.empty else None


def load_cached_prices(path: Path, required_tickers: Iterable[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None

    if df.empty:
        return None

    df.index = normalize_idx(df.index)
    required = set(required_tickers)
    if not required.issubset(set(df.columns)):
        return None

    return df.sort_index()


def extract_close(downloaded: pd.DataFrame) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            return pd.DataFrame()
        close = downloaded["Close"].copy()
    elif "Close" in downloaded.columns:
        close = downloaded[["Close"]].copy()
        close.columns = [ALL_TICKERS[0]]
    else:
        return pd.DataFrame()

    close.index = normalize_idx(close.index)
    close = close.sort_index()
    close = close.dropna(how="all")
    close = close.loc[:, ~close.columns.duplicated()]
    return close


def download_prices(
    event_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    interval: str,
) -> pd.DataFrame:
    # yfinance end dates are exclusive. Add one calendar day so the final event
    # day is included.
    download_start = start.strftime("%Y-%m-%d")
    download_end = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"  downloading {interval}: {download_start} -> {download_end}")
    downloaded = yf.download(
        tickers=ALL_TICKERS,
        start=download_start,
        end=download_end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = extract_close(downloaded)
    if close.empty:
        raise ValueError(f"Yahoo returned no Close data for {event_key} at interval={interval}")

    # Keep only the requested research window. Yahoo may return full trading days.
    close = close[(close.index >= start) & (close.index <= end)]
    if close.empty:
        raise ValueError(f"Yahoo Close data did not overlap {event_key} research window")

    return close


def fetch_or_load_prices(event_key: str) -> tuple[pd.DataFrame, str]:
    start, end = event_research_window(event_key)
    preferred_path = OUTPUT_DIR / f"equity_prices_{event_key}_{REQUESTED_INTERVAL}.csv"
    cached = load_cached_prices(preferred_path, ALL_TICKERS)
    if cached is not None:
        return cached, REQUESTED_INTERVAL

    last_error: Exception | None = None
    for interval in candidate_intervals():
        cache_path = OUTPUT_DIR / f"equity_prices_{event_key}_{interval}.csv"
        cached = load_cached_prices(cache_path, ALL_TICKERS)
        if cached is not None:
            return cached, interval

        try:
            close = download_prices(event_key, start, end, interval)
        except Exception as exc:
            last_error = exc
            print(f"  {interval} failed: {exc}")
            continue

        close.to_csv(cache_path)

        # Preserve the legacy hourly file names when the chosen interval is 1h.
        if interval == "1h":
            legacy_path = OUTPUT_DIR / EVENT_SPECS[event_key]["yahoo"]["hourly_csv"]
            close.to_csv(legacy_path)

        return close, interval

    raise ValueError(f"Could not fetch Yahoo data for {event_key}: {last_error}")


def ticker_role(ticker: str) -> str:
    if ticker in RATE_HIKE_SHORT_CANDIDATES:
        return "rate_hike_short__rate_cut_long"
    if ticker in RATE_HIKE_LONG_CANDIDATES:
        return "rate_hike_long__rate_cut_short"
    if ticker in BENCHMARKS:
        return "benchmark"
    return "unknown"


def build_return_panels(close: pd.DataFrame, event_key: str, interval: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = close.pct_change()
    returns.to_csv(OUTPUT_DIR / f"equity_returns_{event_key}_{interval}.csv")

    bpd = bars_per_trading_day(interval)
    windows = {
        "1bar": 1,
        "3bar": 3,
        "6bar": 6,
        "1d": bpd,
        "3d": bpd * 3,
    }

    long_frames: list[pd.DataFrame] = []
    market_ret_1bar = returns.get(MARKET_BENCHMARK)
    qqq_ret_1bar = returns.get("QQQ")
    iwm_ret_1bar = returns.get("IWM")
    tlt_ret_1bar = returns.get("TLT")

    for ticker in close.columns:
        frame = pd.DataFrame(
            {
                "event": event_key,
                "timestamp": close.index,
                "ticker": ticker,
                "role": ticker_role(ticker),
                "close": close[ticker].values,
                "ret_1bar": returns[ticker].values,
            }
        )

        for label, periods in windows.items():
            frame[f"ret_{label}"] = close[ticker].pct_change(periods).values
            frame[f"fwd_ret_{label}"] = (close[ticker].shift(-periods) / close[ticker] - 1).values

        if market_ret_1bar is not None:
            frame["market_ret_1bar"] = market_ret_1bar.values
            frame["excess_ret_1bar"] = frame["ret_1bar"] - frame["market_ret_1bar"]
        if qqq_ret_1bar is not None:
            frame["qqq_ret_1bar"] = qqq_ret_1bar.values
        if iwm_ret_1bar is not None:
            frame["iwm_ret_1bar"] = iwm_ret_1bar.values
        if tlt_ret_1bar is not None:
            frame["tlt_ret_1bar"] = tlt_ret_1bar.values

        long_frames.append(frame)

    long_features = pd.concat(long_frames, ignore_index=True)
    long_features.to_csv(OUTPUT_DIR / f"equity_features_{event_key}_{interval}.csv", index=False)

    benchmark = pd.DataFrame({"event": event_key, "timestamp": close.index})
    for ticker in BENCHMARKS:
        if ticker not in close.columns:
            continue
        for label, periods in windows.items():
            benchmark[f"{ticker}_ret_{label}"] = close[ticker].pct_change(periods).values
            benchmark[f"{ticker}_fwd_ret_{label}"] = (close[ticker].shift(-periods) / close[ticker] - 1).values
    benchmark.to_csv(OUTPUT_DIR / f"benchmark_features_{event_key}_{interval}.csv", index=False)

    return long_features, benchmark


def load_pm_signal(event_key: str) -> pd.DataFrame | None:
    signal_path = OUTPUT_DIR / f"polymarket_signal_{event_key}.csv"
    if signal_path.exists():
        pm = pd.read_csv(signal_path)
        pm["timestamp"] = datetime64_ns(pm["timestamp"], utc=True)
        return pm.dropna(subset=["timestamp"]).sort_values("timestamp")

    spec = EVENT_SPECS[event_key]
    poly = spec.get("poly")
    if not poly:
        return None

    fallback_paths = [
        OUTPUT_DIR / f"polymarket_prices_{event_key}.csv",
        OUTPUT_DIR / poly["formation_csv"],
        OUTPUT_DIR / poly["event_csv"],
    ]
    for path in fallback_paths:
        if not path.exists():
            continue
        pm = pd.read_csv(path)
        if not {"timestamp", "yes_price"}.issubset(pm.columns):
            continue
        pm["timestamp"] = datetime64_ns(pm["timestamp"], utc=True)
        pm = pm.dropna(subset=["timestamp"]).sort_values("timestamp")
        pm["prob_yes"] = pd.to_numeric(pm["yes_price"], errors="coerce")
        pm["delta_tau_signal"] = pm["prob_yes"].diff()
        pm["abs_delta_tau_signal"] = pm["delta_tau_signal"].abs()
        pm["valid_probability_signal"] = pm["abs_delta_tau_signal"] >= 0.05
        pm["valid_signal"] = pm["valid_probability_signal"]
        return pm

    return None


def align_macro_equity(
    event_key: str,
    equity_features: pd.DataFrame,
    interval: str,
) -> pd.DataFrame | None:
    pm = load_pm_signal(event_key)
    if pm is None or pm.empty:
        return None

    equity = equity_features.copy()
    equity["timestamp"] = datetime64_ns(equity["timestamp"])
    equity = equity.dropna(subset=["timestamp"]).sort_values("timestamp")

    pm_cols = [
        col
        for col in [
            "timestamp",
            "prob_yes",
            "delta_tau_signal",
            "abs_delta_tau_signal",
            "delta_tau_1h",
            "delta_tau_4h",
            "delta_tau_1d",
            "signal_direction",
            "valid_probability_signal",
            "valid_volume_signal",
            "valid_signal",
            "volume_usdc",
            "ofi_proxy_usdc",
        ]
        if col in pm.columns
    ]
    pm = pm[pm_cols].copy()
    pm["timestamp"] = datetime64_ns(pm["timestamp"])
    pm = pm.dropna(subset=["timestamp"]).sort_values("timestamp")

    tolerance = pd.Timedelta(minutes=max(interval_minutes(interval), 60))
    aligned = pd.merge_asof(
        equity,
        pm,
        on="timestamp",
        direction="backward",
        tolerance=tolerance,
    )
    aligned.to_csv(OUTPUT_DIR / f"aligned_macro_equity_{event_key}_{interval}.csv", index=False)
    return aligned


def estimate_macro_betas(aligned: pd.DataFrame, event_key: str) -> pd.DataFrame:
    required = {"ticker", "ret_1bar", "market_ret_1bar", "delta_tau_signal"}
    if not required.issubset(aligned.columns):
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for ticker, group in aligned.groupby("ticker"):
        if ticker in BENCHMARKS:
            continue

        reg = group[["ret_1bar", "market_ret_1bar", "delta_tau_signal"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(reg) < MIN_REGRESSION_OBS:
            continue
        if reg["market_ret_1bar"].var() == 0 or reg["delta_tau_signal"].var() == 0:
            continue

        y = reg["ret_1bar"].to_numpy(dtype=float)
        x = np.column_stack(
            [
                np.ones(len(reg)),
                reg["market_ret_1bar"].to_numpy(dtype=float),
                reg["delta_tau_signal"].to_numpy(dtype=float),
            ]
        )

        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        fitted = x @ coef
        resid = y - fitted
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        rows.append(
            {
                "event": event_key,
                "ticker": ticker,
                "role": ticker_role(ticker),
                "n_obs": len(reg),
                "alpha": coef[0],
                "beta_market": coef[1],
                "beta_tau": coef[2],
                "r2": r2,
            }
        )

    betas = pd.DataFrame(rows)
    if not betas.empty:
        betas = betas.sort_values(["beta_tau", "ticker"], ascending=[False, True])
    return betas


def build_event_return_summary(event_key: str, close: pd.DataFrame) -> list[dict[str, object]]:
    spec = EVENT_SPECS[event_key]
    event_dt = to_naive_utc(spec["event_ts"])
    event_type = spec["event_type"]

    p0_row = get_price_at(close, event_dt, "before")
    p1d_row = get_price_at(close, event_dt + pd.Timedelta(days=1), "after")
    p3d_row = get_price_at(close, event_dt + pd.Timedelta(days=3), "after")

    rows: list[dict[str, object]] = []
    for ticker in close.columns:
        if p0_row is None or pd.isna(p0_row[ticker]):
            continue

        p0 = float(p0_row[ticker])
        p1 = float(p1d_row[ticker]) if p1d_row is not None and pd.notna(p1d_row[ticker]) else None
        p3 = float(p3d_row[ticker]) if p3d_row is not None and pd.notna(p3d_row[ticker]) else None

        rows.append(
            {
                "event": event_key,
                "event_type": event_type,
                "date": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "role": ticker_role(ticker),
                "p0": p0,
                "p1d": p1,
                "p3d": p3,
                "ret_1d": (p1 / p0 - 1) * 100 if p1 is not None else None,
                "ret_3d": (p3 / p0 - 1) * 100 if p3 is not None else None,
            }
        )
    return rows


def build_daily_normalized() -> None:
    start = min(to_naive_utc(spec["event_ts"]) for spec in EVENT_SPECS.values()) - pd.Timedelta(days=90)
    end = max(to_naive_utc(spec["event_ts"]) for spec in EVENT_SPECS.values()) + pd.Timedelta(days=30)

    print("Fetching daily normalized panel")
    downloaded = yf.download(
        tickers=ALL_TICKERS,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    close = extract_close(downloaded)
    if close.empty:
        print("  daily download failed or returned no Close data")
        return

    normalized = (close / close.iloc[0]) * 100
    normalized.to_csv(OUTPUT_DIR / "daily_normalized.csv")
    print(f"  daily_normalized.csv rows={len(normalized)} tickers={len(normalized.columns)}")


def main() -> None:
    print(f"Ticker universe ({len(ALL_TICKERS)}): {', '.join(ALL_TICKERS)}")
    print(f"Requested interval: {REQUESTED_INTERVAL}")

    build_daily_normalized()

    event_return_rows: list[dict[str, object]] = []
    beta_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, object]] = []

    for event_key in EVENT_KEYS:
        print(f"\n[{event_key}]")
        requested_start, requested_end = event_research_window(event_key)

        try:
            close, interval = fetch_or_load_prices(event_key)
        except Exception as exc:
            print(f"  skipped: {exc}")
            coverage_rows.append(
                {
                    "event": event_key,
                    "status": "failed",
                    "interval": None,
                    "requested_start": requested_start,
                    "requested_end": requested_end,
                    "actual_start": None,
                    "actual_end": None,
                    "rows": 0,
                    "missing_tickers": ",".join(ALL_TICKERS),
                    "error": str(exc),
                }
            )
            continue

        missing_tickers = sorted(set(ALL_TICKERS) - set(close.columns))
        print(f"  interval used: {interval}")
        print(f"  rows: {len(close)} | {close.index.min()} -> {close.index.max()}")
        if missing_tickers:
            print(f"  missing tickers: {', '.join(missing_tickers)}")

        equity_features, _ = build_return_panels(close, event_key, interval)
        aligned = align_macro_equity(event_key, equity_features, interval)
        if aligned is not None:
            betas = estimate_macro_betas(aligned, event_key)
            if not betas.empty:
                betas.to_csv(OUTPUT_DIR / f"macro_beta_estimates_{event_key}_{interval}.csv", index=False)
                beta_frames.append(betas)
                print(f"  beta estimates: {len(betas)}")
            else:
                print("  beta estimates: none, not enough aligned signal variation")
        else:
            print("  aligned panel: skipped, no Polymarket signal file")

        event_return_rows.extend(build_event_return_summary(event_key, close))
        coverage_rows.append(
            {
                "event": event_key,
                "status": "ok",
                "interval": interval,
                "requested_start": requested_start,
                "requested_end": requested_end,
                "actual_start": close.index.min(),
                "actual_end": close.index.max(),
                "rows": len(close),
                "missing_tickers": ",".join(missing_tickers),
                "error": None,
            }
        )

    if event_return_rows:
        pd.DataFrame(event_return_rows).to_csv(OUTPUT_DIR / "event_returns_summary.csv", index=False)
        print("\nSaved event_returns_summary.csv")

    if beta_frames:
        all_betas = pd.concat(beta_frames, ignore_index=True)
        all_betas.to_csv(OUTPUT_DIR / "macro_beta_estimates.csv", index=False)
        print("Saved macro_beta_estimates.csv")

    if coverage_rows:
        pd.DataFrame(coverage_rows).to_csv(OUTPUT_DIR / "data_coverage_report.csv", index=False)
        print("Saved data_coverage_report.csv")


if __name__ == "__main__":
    main()
