from __future__ import annotations

import json
import re
import time
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yfinance as yf


CLOB_API = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

EVENT = ("2003524", "GOOGL", "GOOGL Earnings Beat", "2026-04-17", "2026-04-30")

OUTPUT_DIR = Path(__file__).resolve().parent / "one_event_data"
OUTPUT_DIR.mkdir(exist_ok=True)

REQUESTED_INTERVAL = "1m"
POLY_FIDELITY_CANDIDATES = [1, 5, 15, 60, 360, 1440]
YAHOO_INTERVAL_CANDIDATES = ["1m", "2m", "5m", "15m", "30m", "1h", "1d"]


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def event_bounds(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    start = datetime.fromisoformat(start_date).replace(tzinfo=UTC)
    end_exclusive = datetime.fromisoformat(end_date).replace(tzinfo=UTC) + timedelta(days=1)
    return start, end_exclusive


def to_unix(dt: datetime) -> int:
    return int(dt.timestamp())


def normalize_ts(ts: Any) -> pd.Timestamp:
    return pd.Timestamp(ts).tz_convert("UTC") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize("UTC")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return path.name


def summarize_frame(source: str, interval: str, df: pd.DataFrame, status: str, error: str = "") -> dict[str, Any]:
    if df.empty:
        return {
            "source": source,
            "requested_interval": REQUESTED_INTERVAL,
            "interval": interval,
            "status": status,
            "rows": 0,
            "first_timestamp_utc": "",
            "last_timestamp_utc": "",
            "error": error,
        }

    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    timestamps = pd.to_datetime(df[ts_col], utc=True, errors="coerce").dropna()
    return {
        "source": source,
        "requested_interval": REQUESTED_INTERVAL,
        "interval": interval,
        "status": status,
        "rows": len(df),
        "first_timestamp_utc": timestamps.min().isoformat() if not timestamps.empty else "",
        "last_timestamp_utc": timestamps.max().isoformat() if not timestamps.empty else "",
        "error": error,
    }


def get_tokens(client: httpx.Client, market_id: str) -> tuple[dict[str, str], dict[str, Any]]:
    print(f"[meta] market={market_id}")
    response = client.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    response.raise_for_status()
    market = response.json()

    token_map: dict[str, str] = {}
    outcomes = parse_jsonish(market.get("outcomes")) or []
    token_ids = parse_jsonish(market.get("clobTokenIds")) or []
    if isinstance(outcomes, list) and isinstance(token_ids, list) and len(outcomes) == len(token_ids):
        token_map = {str(outcome): str(token_id) for outcome, token_id in zip(outcomes, token_ids)}

    condition_id = market.get("conditionId")
    if condition_id and "Yes" not in token_map:
        clob_response = client.get(f"{CLOB_API}/markets/{condition_id}", timeout=30)
        if clob_response.status_code == 200:
            try:
                clob_market = clob_response.json()
            except json.JSONDecodeError:
                clob_market = {}

            for token in clob_market.get("tokens", []) if isinstance(clob_market, dict) else []:
                outcome = token.get("outcome")
                token_id = token.get("token_id") or token.get("tokenId")
                if outcome and token_id:
                    token_map[str(outcome)] = str(token_id)

            if "Yes" not in token_map:
                outcomes_found = re.findall(r'"outcome"\s*:\s*"([^"]+)"', clob_response.text)
                tokens_found = re.findall(r'"token_id"\s*:\s*"([^"]+)"', clob_response.text)
                if outcomes_found and tokens_found:
                    token_map.update(dict(zip(outcomes_found, tokens_found)))

    market_meta = {
        "market_id": market.get("id") or market_id,
        "condition_id": condition_id,
        "question": market.get("question"),
        "slug": market.get("slug"),
        "outcomes": json.dumps(outcomes) if isinstance(outcomes, list) else str(outcomes),
        "token_ids": json.dumps(token_map),
    }
    return token_map, market_meta


def fetch_poly_history(
    client: httpx.Client,
    token_id: str,
    start: datetime,
    end_exclusive: datetime,
    fidelity_minutes: int,
) -> tuple[pd.DataFrame, str]:
    start_ts = to_unix(start)
    end_ts = to_unix(end_exclusive) - 1
    chunk_seconds = 7 * 86400 if fidelity_minutes <= 5 else 15 * 86400
    cursor = start_ts
    rows: list[dict[str, Any]] = []

    while cursor <= end_ts:
        chunk_end = min(cursor + chunk_seconds - 1, end_ts)
        response = client.get(
            f"{CLOB_API}/prices-history",
            params={
                "market": token_id,
                "startTs": cursor,
                "endTs": chunk_end,
                "fidelity": fidelity_minutes,
            },
            timeout=30,
        )
        if response.status_code != 200:
            return pd.DataFrame(), f"HTTP {response.status_code}: {response.text[:300]}"

        history = response.json().get("history", [])
        rows.extend(history)
        cursor = chunk_end + 1
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame(), "no history rows returned"

    df = pd.DataFrame(rows)
    if not {"t", "p"}.issubset(df.columns):
        return pd.DataFrame(), f"unexpected history columns: {list(df.columns)}"

    out = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(df["t"].astype(float), unit="s", utc=True),
            "polymarket_yes_price": pd.to_numeric(df["p"], errors="coerce"),
            "polymarket_fidelity_minutes": fidelity_minutes,
        }
    )
    out = out.dropna(subset=["timestamp_utc", "polymarket_yes_price"])
    out = out.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    return out, ""


def fetch_best_poly_history(
    client: httpx.Client,
    yes_token_id: str,
    start: datetime,
    end_exclusive: datetime,
) -> tuple[pd.DataFrame, int | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []

    for fidelity in POLY_FIDELITY_CANDIDATES:
        print(f"[poly] trying fidelity={fidelity} minute(s)")
        try:
            df, error = fetch_poly_history(client, yes_token_id, start, end_exclusive, fidelity)
        except Exception as exc:
            df = pd.DataFrame()
            error = str(exc)

        if df.empty:
            attempts.append(summarize_frame("polymarket_yes", f"{fidelity}m", df, "empty", error))
            continue

        attempts.append(summarize_frame("polymarket_yes", f"{fidelity}m", df, "selected"))
        return df, fidelity, attempts

    return pd.DataFrame(), None, attempts


def normalize_yahoo_history(raw: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    keep = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in df.columns]
    if not keep:
        return pd.DataFrame()

    df = df[keep].copy()
    index = pd.to_datetime(df.index)
    if getattr(index, "tz", None) is None:
        index = index.tz_localize("UTC")
    else:
        index = index.tz_convert("UTC")

    df.insert(0, "timestamp_utc", index)
    rename_map = {
        "Open": "stock_open",
        "High": "stock_high",
        "Low": "stock_low",
        "Close": "stock_close",
        "Volume": "stock_volume",
    }
    df = df.rename(columns=rename_map)
    df["ticker"] = ticker
    df["stock_interval"] = interval
    return df.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")


def fetch_yahoo_history(
    ticker: str,
    start: datetime,
    end_exclusive: datetime,
    interval: str,
) -> tuple[pd.DataFrame, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            raw = yf.Ticker(ticker).history(
                start=start.strftime("%Y-%m-%d"),
                end=end_exclusive.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=False,
                actions=False,
                prepost=True,
                raise_errors=True,
            )
    except Exception as exc:
        return pd.DataFrame(), str(exc)

    df = normalize_yahoo_history(raw, ticker, interval)
    if df.empty:
        return df, "no rows returned"
    return df, ""


def fetch_best_yahoo_history(
    ticker: str,
    start: datetime,
    end_exclusive: datetime,
) -> tuple[pd.DataFrame, str | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []

    for interval in YAHOO_INTERVAL_CANDIDATES:
        print(f"[yahoo] trying interval={interval}")
        df, error = fetch_yahoo_history(ticker, start, end_exclusive, interval)
        if df.empty:
            attempts.append(summarize_frame(f"yahoo_{ticker}", interval, df, "empty", error))
            continue

        attempts.append(summarize_frame(f"yahoo_{ticker}", interval, df, "selected"))
        return df, interval, attempts

    return pd.DataFrame(), None, attempts


def merge_event_data(
    event: tuple[str, str, str, str, str],
    market_meta: dict[str, Any],
    poly_df: pd.DataFrame,
    stock_df: pd.DataFrame,
) -> pd.DataFrame:
    market_id, ticker, event_name, start_date, end_date = event

    if poly_df.empty and stock_df.empty:
        return pd.DataFrame()
    if poly_df.empty:
        merged = stock_df.copy()
    elif stock_df.empty:
        merged = poly_df.copy()
    else:
        merged = pd.merge(poly_df, stock_df, on="timestamp_utc", how="outer").sort_values("timestamp_utc")

    merged.insert(0, "event_id", market_id)
    merged.insert(1, "event_name", event_name)
    merged.insert(2, "event_start_date", start_date)
    merged.insert(3, "event_end_date", end_date)
    merged.insert(4, "market_question", market_meta.get("question"))
    if "ticker" not in merged.columns:
        merged["ticker"] = ticker
    return merged


def write_availability_report(attempts: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    report = pd.DataFrame(attempts)
    report.to_csv(path, index=False)
    return report


def main() -> None:
    market_id, ticker, event_name, start_date, end_date = EVENT
    start, end_exclusive = event_bounds(start_date, end_date)

    print(f"Event: {event_name} | market={market_id} | ticker={ticker}")
    print(f"Window: {start.isoformat()} -> {end_exclusive.isoformat()} (end exclusive)")

    attempts: list[dict[str, Any]] = []
    with httpx.Client(headers={"User-Agent": "one-event-data-fetch/1.0"}) as client:
        token_map, market_meta = get_tokens(client, market_id)
        yes_token_id = token_map.get("Yes") or token_map.get("YES") or token_map.get("yes")
        if not yes_token_id:
            raise ValueError(f"No YES token found. Token map: {token_map}")

        print(f"[meta] yes_token_id={yes_token_id[:24]}...")
        poly_df, poly_fidelity, poly_attempts = fetch_best_poly_history(client, yes_token_id, start, end_exclusive)
        attempts.extend(poly_attempts)

    stock_df, stock_interval, stock_attempts = fetch_best_yahoo_history(ticker, start, end_exclusive)
    attempts.extend(stock_attempts)

    merged = merge_event_data(EVENT, market_meta, poly_df, stock_df)
    data_path = OUTPUT_DIR / f"{market_id}_{ticker}_one_event_data.csv"
    availability_path = OUTPUT_DIR / f"{market_id}_{ticker}_availability.csv"
    meta_path = OUTPUT_DIR / f"{market_id}_{ticker}_market_meta.json"

    if not merged.empty:
        merged.to_csv(data_path, index=False)
        print(f"[saved] data: {display_path(data_path)} ({len(merged)} rows)")
    else:
        print("[saved] data: no CSV written because both data sources were empty")

    report = write_availability_report(attempts, availability_path)
    meta = {
        **market_meta,
        "event": {
            "market_id": market_id,
            "ticker": ticker,
            "name": event_name,
            "start_date": start_date,
            "end_date": end_date,
            "start_utc": start.isoformat(),
            "end_exclusive_utc": end_exclusive.isoformat(),
        },
        "selected": {
            "polymarket_fidelity_minutes": poly_fidelity,
            "stock_interval": stock_interval,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] availability: {display_path(availability_path)}")
    print(f"[saved] meta: {display_path(meta_path)}")

    print("\nAvailability:")
    cols = ["source", "interval", "status", "rows", "first_timestamp_utc", "last_timestamp_utc", "error"]
    print(report[cols].to_string(index=False))


if __name__ == "__main__":
    main()
