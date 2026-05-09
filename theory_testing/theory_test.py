import pandas as pd
import numpy as np
from pathlib import Path


FED_POLY_CSV = "polymarket_fed_sep2024.csv"
JOBS_POLY_CSV = "polymarket_jobs_august.csv"
EVENT_RETURNS_CSV = "event_returns_summary.csv"

FOMC_EVENT_TS = pd.Timestamp("2024-09-18 14:00:00")
JOBS_POLY_EVENT_TS = pd.Timestamp("2024-09-06 08:30:00")

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA"]

RATE_SENSITIVE = ["DHI", "LEN", "ENPH", "RUN", "CVNA", "TSLA", "UPST", "IWM"]
RATE_RESILIENT = ["BAC", "JPM", "WFC", "KRE", "PGR", "ALL", "KO", "PG", "PEP"]

BACKTEST_EVENTS = [
    # 2024
    {"name": "cpi_hot_april", "date": "2024-05-15 08:30:00", "type": "hawkish"},
    {"name": "cpi_cool_july", "date": "2024-07-11 08:30:00", "type": "dovish"},
    {"name": "jobs_aug_unemployment", "date": "2024-09-06 08:30:00", "type": "dovish"},
    {"name": "fomc_cut_sep", "date": "2024-09-18 14:00:00", "type": "dovish"},

    # 2025
    {"name": "fomc_jan_2025", "date": "2025-01-29 14:00:00", "type": "neutral"},
    {"name": "cpi_mar_2025", "date": "2025-04-10 08:30:00", "type": "dovish"},
    {"name": "nfp_may_2025", "date": "2025-06-06 08:30:00", "type": "neutral"},
    {"name": "fomc_jul_2025", "date": "2025-07-30 14:00:00", "type": "neutral"},
]


def require_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return p


def normalize_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    idx = pd.to_datetime(idx)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx


def load_polymarket(filepath: str) -> pd.DataFrame:
    path = require_file(filepath)
    df = pd.read_csv(path)

    required_cols = {"timestamp", "yes_price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{filepath} missing required columns: {sorted(missing)}")

    df["datetime"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    df["prob"] = pd.to_numeric(df["yes_price"], errors="raise")
    df = df[["datetime", "prob"]].dropna().drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df = df.set_index("datetime")

    if df.empty:
        raise ValueError(f"{filepath} contains no valid rows")

    if (df["prob"] < 0).any() or (df["prob"] > 1).any():
        raise ValueError(f"{filepath} contains prob outside [0,1]")

    return df


def load_event_returns(filepath: str) -> pd.DataFrame:
    path = require_file(filepath)
    df = pd.read_csv(path)

    required_cols = {"event", "ticker", "ret_1d", "ret_3d"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{filepath} missing required columns: {sorted(missing)}")

    return df


def load_hourly_event_csv(event_name: str) -> pd.DataFrame:
    filepath = f"hourly_{event_name}.csv"
    path = require_file(filepath)

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = normalize_datetime_index(df.index)

    if df.empty:
        raise ValueError(f"{filepath} is empty")

    return df.sort_index()


def compute_threshold_lead(poly_df: pd.DataFrame, event_ts: pd.Timestamp, threshold: float) -> dict:
    pre = poly_df[poly_df.index < event_ts]

    if pre.empty:
        return {
            "threshold": threshold,
            "first_cross_ts": None,
            "lead_days": -1.0,
            "left_censored": False,
        }

    above = pre[pre["prob"] >= threshold]
    if above.empty:
        return {
            "threshold": threshold,
            "first_cross_ts": None,
            "lead_days": -1.0,
            "left_censored": False,
        }

    first_cross_ts = above.index[0]
    lead_days = (event_ts - first_cross_ts).total_seconds() / 86400.0
    left_censored = bool(pre["prob"].iloc[0] >= threshold)

    return {
        "threshold": threshold,
        "first_cross_ts": first_cross_ts,
        "lead_days": lead_days,
        "left_censored": left_censored,
    }


def compute_daily_spikes(poly_df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    daily = poly_df.resample("D").last().ffill()
    daily["delta"] = daily["prob"].diff()
    return daily[daily["delta"].abs() >= threshold].copy()


def compute_jobs_repricing(
    jobs_poly_df: pd.DataFrame,
    release_ts: pd.Timestamp,
    spike_threshold: float = 0.05,
) -> dict:
    daily = jobs_poly_df.resample("D").last().ffill()
    daily["delta"] = daily["prob"].diff()

    window_start = release_ts.normalize() - pd.Timedelta(days=3)
    window_end = release_ts.normalize() + pd.Timedelta(days=1)
    window = daily[(daily.index >= window_start) & (daily.index <= window_end)].copy()

    if window.empty:
        return {
            "window_start": window_start,
            "window_end": window_end,
            "max_abs_delta": 0.0,
            "spike_dates": [],
            "spike_values": [],
        }

    spikes = window[window["delta"].abs() >= spike_threshold]
    max_abs_delta = float(window["delta"].abs().max()) if window["delta"].notna().any() else 0.0

    return {
        "window_start": window_start,
        "window_end": window_end,
        "max_abs_delta": max_abs_delta,
        "spike_dates": spikes.index.strftime("%Y-%m-%d").tolist(),
        "spike_values": [float(x) for x in spikes["delta"].tolist()],
    }


def compute_event_returns(events_df: pd.DataFrame) -> dict:
    results = {}

    fomc_aliases = {"fomc_cut_sep", "fomc_decision"}
    jobs_aliases = {"jobs_aug_unemployment", "jobs_weak_aug", "jobs_weak"}

    fomc_df = events_df[events_df["event"].isin(fomc_aliases)]
    if not fomc_df.empty:
        fomc_mag7 = fomc_df[fomc_df["ticker"].isin(MAG7)]
        results["Mag7_FOMC_Ret_1d"] = float(fomc_mag7["ret_1d"].mean()) if not fomc_mag7.empty else np.nan

        fomc_tlt = fomc_df[fomc_df["ticker"] == "TLT"]
        results["TLT_FOMC_Ret_1d"] = float(fomc_tlt["ret_1d"].iloc[0]) if not fomc_tlt.empty else np.nan

    jobs_df = events_df[events_df["event"].isin(jobs_aliases)]
    if not jobs_df.empty:
        results["All_Jobs_Ret_1d"] = float(jobs_df["ret_1d"].mean())

    return results


def build_theory_report(
    fed_poly_df: pd.DataFrame,
    jobs_poly_df: pd.DataFrame,
    event_returns: dict,
) -> pd.DataFrame:
    lead_75 = compute_threshold_lead(fed_poly_df, FOMC_EVENT_TS, 0.75)
    lead_90 = compute_threshold_lead(fed_poly_df, FOMC_EVENT_TS, 0.90)
    fed_spikes = compute_daily_spikes(fed_poly_df, threshold=0.05)
    jobs_repricing = compute_jobs_repricing(jobs_poly_df, JOBS_POLY_EVENT_TS, spike_threshold=0.05)

    mag7_ret = event_returns.get("Mag7_FOMC_Ret_1d", np.nan)
    tlt_ret = event_returns.get("TLT_FOMC_Ret_1d", np.nan)
    jobs_ret = event_returns.get("All_Jobs_Ret_1d", np.nan)

    q1_score = "STRONG" if lead_75["lead_days"] >= 14 else ("MODERATE" if lead_75["lead_days"] >= 7 else "WEAK")
    q2_score = "STRONG" if lead_90["lead_days"] >= 7 else ("MODERATE" if lead_90["lead_days"] >= 3 else "WEAK")
    q3_score = "STRONG" if pd.notna(mag7_ret) and mag7_ret > 1.0 else ("MODERATE" if pd.notna(mag7_ret) and mag7_ret > 0 else "WEAK")
    q4_score = "STRONG" if pd.notna(tlt_ret) and tlt_ret < -1.0 else ("MODERATE" if pd.notna(tlt_ret) and tlt_ret < 0 else "WEAK")
    q5_score = "STRONG" if jobs_repricing["max_abs_delta"] >= 0.10 else ("MODERATE" if jobs_repricing["max_abs_delta"] >= 0.05 else "WEAK")

    rows = [
        {
            "question": "Fed market above 75% at least 14 days before FOMC",
            "score": q1_score,
            "value": round(lead_75["lead_days"], 2),
            "extra": f"first_cross={lead_75['first_cross_ts']}, left_censored={lead_75['left_censored']}",
        },
        {
            "question": "Fed market above 90% sufficiently ahead of FOMC",
            "score": q2_score,
            "value": round(lead_90["lead_days"], 2),
            "extra": f"first_cross={lead_90['first_cross_ts']}, left_censored={lead_90['left_censored']}",
        },
        {
            "question": "Mag7 average return on FOMC event day",
            "score": q3_score,
            "value": None if pd.isna(mag7_ret) else round(float(mag7_ret), 4),
            "extra": "metric=Mag7_FOMC_Ret_1d",
        },
        {
            "question": "TLT sell-the-news reaction on FOMC event day",
            "score": q4_score,
            "value": None if pd.isna(tlt_ret) else round(float(tlt_ret), 4),
            "extra": "metric=TLT_FOMC_Ret_1d",
        },
        {
            "question": "August unemployment Polymarket repriced around Sep 6 release",
            "score": q5_score,
            "value": round(float(jobs_repricing["max_abs_delta"]), 4),
            "extra": f"spike_dates={jobs_repricing['spike_dates']}, spike_values={[round(x, 4) for x in jobs_repricing['spike_values']]}",
        },
        {
            "question": "Fed daily spike count",
            "score": "INFO",
            "value": int(len(fed_spikes)),
            "extra": f"dates={fed_spikes.index.strftime('%Y-%m-%d').tolist()}",
        },
        {
            "question": "Jobs basket average return on jobs event day",
            "score": "INFO",
            "value": None if pd.isna(jobs_ret) else round(float(jobs_ret), 4),
            "extra": "metric=All_Jobs_Ret_1d",
        },
        {
            "question": "Fed Polymarket coverage window",
            "score": "INFO",
            "value": int(len(fed_poly_df)),
            "extra": f"{fed_poly_df.index.min()} -> {fed_poly_df.index.max()}",
        },
        {
            "question": "Jobs Polymarket coverage window",
            "score": "INFO",
            "value": int(len(jobs_poly_df)),
            "extra": f"{jobs_poly_df.index.min()} -> {jobs_poly_df.index.max()}",
        },
    ]

    return pd.DataFrame(rows)


def simulate_advanced_stat_arb(
    hourly_df: pd.DataFrame,
    event_name: str,
    event_date_str: str,
    event_type: str,
    take_profit: float = 0.03,
    initial_capital: float = 100000.0,
) -> dict | None:
    if hourly_df is None or hourly_df.empty:
        return None

    if event_type == "dovish":
        long_tickers = RATE_SENSITIVE
        short_tickers = RATE_RESILIENT
    else:
        long_tickers = RATE_RESILIENT
        short_tickers = RATE_SENSITIVE

    event_dt = pd.Timestamp(event_date_str)
    before_mask = hourly_df.index <= event_dt

    if not before_mask.any():
        fallback_dt = pd.Timestamp(event_date_str.split()[0] + " 23:59:59")
        before_mask = hourly_df.index <= fallback_dt
        if not before_mask.any():
            return None

    t0_idx = hourly_df.index[before_mask][-1]
    future_data = hourly_df[hourly_df.index > t0_idx]
    if future_data.empty:
        return None

    entry_idx = future_data.index[0]
    entry_row = hourly_df.loc[entry_idx]

    valid_longs = [t for t in long_tickers if t in entry_row.index and pd.notna(entry_row[t]) and float(entry_row[t]) > 0]
    valid_shorts = [t for t in short_tickers if t in entry_row.index and pd.notna(entry_row[t]) and float(entry_row[t]) > 0]

    if not valid_longs or not valid_shorts:
        return None

    long_alloc = (initial_capital / 2.0) / len(valid_longs)
    short_alloc = (initial_capital / 2.0) / len(valid_shorts)

    peak_portfolio_value = initial_capital
    max_drawdown = 0.0

    exit_time = None
    exit_portfolio_value = None
    exit_spread = None
    hit_target = False

    for current_time, prices_current in future_data.loc[entry_idx:].iterrows():
        long_rets = []
        for t in valid_longs:
            px0 = float(entry_row[t])
            pxt = float(prices_current[t])
            if pd.notna(pxt) and pxt > 0:
                long_rets.append((pxt / px0) - 1)

        short_rets = []
        for t in valid_shorts:
            px0 = float(entry_row[t])
            pxt = float(prices_current[t])
            if pd.notna(pxt) and pxt > 0:
                short_rets.append((pxt / px0) - 1)

        if not long_rets or not short_rets:
            continue

        avg_long = float(np.mean(long_rets))
        avg_short = float(np.mean(short_rets))
        spread = avg_long - avg_short

        long_value = sum(
            long_alloc * (float(prices_current[t]) / float(entry_row[t]))
            for t in valid_longs
            if pd.notna(prices_current[t]) and float(prices_current[t]) > 0
        )
        short_value = sum(
            short_alloc * (2.0 - (float(prices_current[t]) / float(entry_row[t])))
            for t in valid_shorts
            if pd.notna(prices_current[t]) and float(prices_current[t]) > 0
        )
        portfolio_value = long_value + short_value

        if portfolio_value > peak_portfolio_value:
            peak_portfolio_value = portfolio_value

        drawdown = (portfolio_value - peak_portfolio_value) / peak_portfolio_value
        max_drawdown = min(max_drawdown, drawdown)

        exit_time = current_time
        exit_portfolio_value = portfolio_value
        exit_spread = spread

        if spread >= take_profit:
            hit_target = True
            break

    if exit_time is None or exit_portfolio_value is None or exit_spread is None:
        return None

    profit = exit_portfolio_value - initial_capital
    hours_held = (exit_time - entry_idx).total_seconds() / 3600.0

    return {
        "event": event_name,
        "event_type": event_type,
        "event_ts": event_date_str,
        "entry_ts": entry_idx,
        "exit_ts": exit_time,
        "hours_held": round(hours_held, 2),
        "valid_longs": len(valid_longs),
        "valid_shorts": len(valid_shorts),
        "spread_exit": round(exit_spread, 6),
        "profit": round(profit, 2),
        "return_pct": round((profit / initial_capital) * 100.0, 4),
        "max_drawdown_pct": round(max_drawdown * 100.0, 4),
        "hit_take_profit": hit_target,
    }


def available_backtest_events() -> list[dict]:
    available = []
    for event in BACKTEST_EVENTS:
        filepath = Path(f"hourly_{event['name']}.csv")
        if filepath.exists():
            available.append(event)
        else:
            print(f"⚠️ Missing hourly file for {event['name']}: {filepath}")
    return available


def run_stat_arb_backtest() -> pd.DataFrame:
    results = []

    for event in available_backtest_events():
        hourly_df = load_hourly_event_csv(event["name"])
        result = simulate_advanced_stat_arb(
            hourly_df=hourly_df,
            event_name=event["name"],
            event_date_str=event["date"],
            event_type=event["type"],
            take_profit=0.03,
        )
        if result is not None:
            results.append(result)

    if not results:
        raise ValueError("No stat-arb results were produced from available hourly CSV files")

    return pd.DataFrame(results)


def print_theory_report(report_df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("THEORY REPORT")
    print("=" * 72)
    for _, row in report_df.iterrows():
        print(f"{row['score']:<8} | {row['question']}")
        print(f"         value={row['value']} | {row['extra']}")


def print_stat_arb_summary(bt_df: pd.DataFrame) -> None:
    total_profit = float(bt_df["profit"].sum())
    wins = int((bt_df["profit"] > 0).sum())
    tested = int(len(bt_df))
    hit_rate = float(bt_df["hit_take_profit"].mean() * 100.0)
    avg_profit = float(bt_df["profit"].mean())
    avg_mdd = float(bt_df["max_drawdown_pct"].mean())

    print("\n" + "=" * 72)
    print("STAT ARB SUMMARY")
    print("=" * 72)
    print(
        bt_df[
            [
                "event",
                "event_type",
                "entry_ts",
                "exit_ts",
                "hours_held",
                "profit",
                "return_pct",
                "max_drawdown_pct",
                "hit_take_profit",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 72)
    print("AGGREGATE")
    print("=" * 72)
    print(f"Events tested        : {tested}")
    print(f"Winning events       : {wins}")
    print(f"Win rate             : {(wins / tested) * 100:.2f}%")
    print(f"Hit take-profit rate : {hit_rate:.2f}%")
    print(f"Average profit       : ${avg_profit:,.2f}")
    print(f"Total net profit     : ${total_profit:,.2f}")
    print(f"Average max drawdown : {avg_mdd:.2f}%")


def main() -> None:
    fed_poly_df = load_polymarket(FED_POLY_CSV)
    jobs_poly_df = load_polymarket(JOBS_POLY_CSV)
    event_returns_df = load_event_returns(EVENT_RETURNS_CSV)
    event_returns = compute_event_returns(event_returns_df)

    theory_report = build_theory_report(
        fed_poly_df=fed_poly_df,
        jobs_poly_df=jobs_poly_df,
        event_returns=event_returns,
    )
    theory_report.to_csv("theory_report.csv", index=False)
    print_theory_report(theory_report)

    stat_arb_df = run_stat_arb_backtest()
    stat_arb_df.to_csv("stat_arb_summary.csv", index=False)
    print_stat_arb_summary(stat_arb_df)


if __name__ == "__main__":
    main()