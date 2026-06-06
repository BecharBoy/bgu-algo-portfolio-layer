"""
Event-Asset Price vs Polymarket Probability Analyzer
=====================================================
Reads recently analyzed events from the DB, fetches daily price data
from yfinance and probability history from the Polymarket CLOB API,
then generates premium dark-themed comparison plots.

Usage:
    python -m database.event_asset_price_vs_probability
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import httpx
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import yfinance as yf

sys.stdout.reconfigure(errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from database.db_connection import connect

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCHEMA_NAME = "checking_relevant_events"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com/prices-history"
CHUNK_SEC = 10 * 86400
MAX_EVENTS = 5
OUTPUT_DIR = REPO_ROOT / "database" / "plots"

# ---------------------------------------------------------------------------
# Dark-theme colour palette
# ---------------------------------------------------------------------------
C = {
    "bg":           "#0d1117",
    "panel":        "#161b22",
    "grid":         "#21262d",
    "border":       "#30363d",
    "txt":          "#e6edf3",
    "txt2":         "#8b949e",
    "price":        "#58a6ff",
    "price_fill":   "#58a6ff",
    "prob":         "#f0883e",
    "prob_fill":    "#f0883e",
    "pos":          "#3fb950",
    "neg":          "#f85149",
    "accent":       "#bc8cff",
}

ASSET_COLORS = [
    "#58a6ff", "#f0883e", "#3fb950", "#bc8cff",
    "#f85149", "#79c0ff", "#d2a8ff", "#56d364",
    "#ffa657", "#ff7b72", "#a5d6ff", "#ffd8b5",
]


# ===================================================================
# 1.  DB helpers
# ===================================================================

async def load_events_from_db(limit: int = MAX_EVENTS) -> list[dict[str, Any]]:
    """Return the *limit* most-recently analysed events from the LLM review table."""
    conn = await connect()
    try:
        rows = await conn.fetch(
            f"""
            SELECT event_id, event_title, llm_input, llm_output
            FROM {SCHEMA_NAME}.event_asset_llm_review
            ORDER BY analyzed_at DESC
            LIMIT $1
            """,
            limit,
        )
    finally:
        await conn.close()

    events: list[dict[str, Any]] = []
    for r in rows:
        llm_in = json.loads(r["llm_input"]) if isinstance(r["llm_input"], str) else r["llm_input"]
        llm_out = json.loads(r["llm_output"]) if isinstance(r["llm_output"], str) else r["llm_output"]
        events.append({
            "event_id":    r["event_id"],
            "event_title": r["event_title"],
            "created_at":  _parse_dt(llm_in.get("created_at")),
            "end_at":      _parse_dt(llm_in.get("end_at")),
            "market_id":   llm_in.get("market_id"),
            "market_question": llm_in.get("market_question"),
            "tags":        llm_in.get("tags", []),
            "assets":      llm_out.get("assets", []),
        })
    return events


def _parse_dt(val: Any) -> datetime | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.astimezone(timezone.utc) if val.tzinfo else val.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return None


# ===================================================================
# 2.  Polymarket helpers
# ===================================================================

def fetch_event_markets(event_id: str) -> list[dict[str, Any]]:
    resp = httpx.get(f"{GAMMA_API_BASE}/events/{event_id}", timeout=30)
    resp.raise_for_status()
    raw = resp.json().get("markets") or []
    markets: list[dict[str, Any]] = []
    for m in raw:
        outcomes = m.get("outcomes") or []
        if isinstance(outcomes, str):
            outcomes = json.loads(outcomes)
        op = m.get("outcomePrices") or []
        if isinstance(op, str):
            op = json.loads(op)
        tokens = m.get("clobTokenIds") or []
        if isinstance(tokens, str):
            tokens = json.loads(tokens)
        markets.append({
            "market_id":       str(m.get("id") or ""),
            "question":        m.get("question") or m.get("groupItemTitle") or "",
            "outcomes":        outcomes,
            "outcome_prices":  [float(p) for p in op if p not in (None, "")],
            "clob_token_ids":  tokens,
            "closed":          bool(m.get("closed")),
        })
    return markets


def find_correct_market(markets: list[dict[str, Any]]) -> dict[str, Any] | None:
    # Pass 1 - resolved YES
    for m in markets:
        oc = [o.lower() for o in m["outcomes"]]
        pr = m["outcome_prices"]
        if "yes" in oc and len(pr) == len(oc) and pr[oc.index("yes")] >= 0.95:
            return m
    # Pass 2 - highest YES probability
    best, best_p = None, -1.0
    for m in markets:
        oc = [o.lower() for o in m["outcomes"]]
        pr = m["outcome_prices"]
        if "yes" in oc and len(pr) == len(oc):
            p = pr[oc.index("yes")]
            if p > best_p:
                best, best_p = m, p
    return best


def yes_token_id(market: dict[str, Any]) -> str | None:
    oc = [o.lower() for o in market["outcomes"]]
    tk = market["clob_token_ids"]
    if len(oc) == len(tk) and "yes" in oc:
        return tk[oc.index("yes")]
    return None


def resolve_market(market: dict[str, Any]) -> tuple[str | None, bool | None]:
    oc = [o.lower() for o in market["outcomes"]]
    pr = market["outcome_prices"]
    if "yes" not in oc or len(pr) != len(oc):
        return None, None
    yp = pr[oc.index("yes")]
    if market["closed"]:
        if yp >= 0.95:
            return "YES", True
        if yp <= 0.05:
            return "NO", False
    return None, None


def fetch_prob_history(token: str, start: datetime, end: datetime) -> pd.DataFrame:
    rows: list[tuple[datetime, float]] = []
    cur = int(start.timestamp())
    end_ts = int(end.timestamp())
    while cur <= end_ts:
        ce = min(cur + CHUNK_SEC - 1, end_ts)
        r = httpx.get(CLOB_API, params={"market": token, "startTs": cur, "endTs": ce, "fidelity": 60}, timeout=30)
        r.raise_for_status()
        for h in r.json().get("history") or []:
            rows.append((datetime.fromtimestamp(float(h["t"]), tz=timezone.utc), max(0.0, min(1.0, float(h["p"])))))
        cur = ce + 1
    if not rows:
        return pd.DataFrame(columns=["datetime", "probability"])
    df = pd.DataFrame(rows, columns=["datetime", "probability"])
    df = df.drop_duplicates("datetime").sort_values("datetime").set_index("datetime")
    return df.resample("1D").last().dropna().reset_index()


# ===================================================================
# 3.  yfinance helper
# ===================================================================

def fetch_prices(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    t = yf.Ticker(symbol)
    h = t.history(start=start.strftime("%Y-%m-%d"), end=(end + timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d")
    if h.empty:
        return pd.DataFrame(columns=["datetime", "close"])
    df = h[["Close"]].reset_index()
    df.columns = ["datetime", "close"]
    if df["datetime"].dt.tz is not None:
        df["datetime"] = df["datetime"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


# ===================================================================
# 4.  PLOTTING  -- premium dark-theme visuals
# ===================================================================

def _setup_style():
    plt.rcParams.update({
        "figure.facecolor":   C["bg"],
        "axes.facecolor":     C["panel"],
        "axes.edgecolor":     C["border"],
        "axes.labelcolor":    C["txt"],
        "axes.grid":          True,
        "grid.color":         C["grid"],
        "grid.alpha":         0.4,
        "grid.linewidth":     0.5,
        "text.color":         C["txt"],
        "xtick.color":        C["txt2"],
        "ytick.color":        C["txt2"],
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.facecolor":   C["panel"],
        "legend.edgecolor":   C["border"],
        "legend.fontsize":    7,
        "font.family":        "sans-serif",
        "font.size":          9,
        "savefig.facecolor":  C["bg"],
        "savefig.edgecolor":  C["bg"],
    })


def _corr(price_df: pd.DataFrame, prob_df: pd.DataFrame) -> float | None:
    """Pearson correlation between daily close price and probability."""
    if price_df.empty or prob_df.empty:
        return None
    p1 = price_df.set_index("datetime")["close"]
    p2 = prob_df.copy()
    if p2["datetime"].dt.tz is not None:
        p2["datetime"] = p2["datetime"].dt.tz_localize(None)
    p2 = p2.set_index("datetime")["probability"]
    merged = pd.concat([p1, p2], axis=1, join="inner").dropna()
    if len(merged) < 3:
        return None
    return float(merged.corr().iloc[0, 1])


def _pct_change(df: pd.DataFrame) -> float | None:
    if df.empty or len(df) < 2:
        return None
    return (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100


# ---------- Plot A: per-asset grid with dual axes --------------------

def plot_asset_grid(
    event: dict[str, Any],
    price_data: dict[str, pd.DataFrame],
    prob_df: pd.DataFrame,
    market_q: str,
    outcome: str | None,
    did_happen: bool | None,
) -> Path:
    assets = event["assets"]
    n = len(assets)
    cols = 2
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes_flat = [axes] if n == 1 else axes.flatten()

    # Suptitle block
    status = "YES" if did_happen is True else ("NO" if did_happen is False else "PENDING")
    status_color = C["pos"] if did_happen is True else (C["neg"] if did_happen is False else C["txt2"])
    fig.suptitle(
        f'{event["event_title"]}',
        fontsize=16, fontweight="bold", color=C["txt"], y=0.98,
    )
    fig.text(
        0.5, 0.955,
        f'Market: {market_q}   |   Outcome: {status}',
        ha="center", fontsize=11, color=status_color, fontstyle="italic",
    )

    for i, asset in enumerate(assets):
        ax = axes_flat[i]
        sym = asset["symbol"]
        pdf = price_data.get(sym, pd.DataFrame(columns=["datetime", "close"]))
        corr = _corr(pdf, prob_df)
        pct = _pct_change(pdf)

        # --- left axis: price ---
        if not pdf.empty:
            ax.plot(pdf["datetime"], pdf["close"], color=C["price"], linewidth=2, zorder=3)
            ax.fill_between(pdf["datetime"], pdf["close"], pdf["close"].min() * 0.998,
                            color=C["price_fill"], alpha=0.10, zorder=2)
            ax.set_ylabel(f"Price ($)", color=C["price"], fontsize=9)
            ax.tick_params(axis="y", colors=C["price"])
        else:
            ax.set_ylabel("(no data)", fontsize=9)

        # --- right axis: probability ---
        ax2 = ax.twinx()
        if not prob_df.empty:
            pd_dates = prob_df["datetime"].copy()
            if pd_dates.dt.tz is not None:
                pd_dates = pd_dates.dt.tz_localize(None)
            probs = prob_df["probability"] * 100
            ax2.plot(pd_dates, probs, color=C["prob"], linewidth=2, linestyle="--", zorder=3)
            ax2.fill_between(pd_dates, probs, 0, color=C["prob_fill"], alpha=0.08, zorder=2)
            ax2.set_ylabel("Probability (%)", color=C["prob"], fontsize=9)
            ax2.tick_params(axis="y", colors=C["prob"])
            ax2.set_ylim(0, 105)

        # --- title / annotations ---
        badge = ""
        if corr is not None:
            badge += f"  r={corr:+.2f}"
        if pct is not None:
            col = C["pos"] if pct >= 0 else C["neg"]
            badge += f"  {pct:+.1f}%"
        ax.set_title(f'{sym} - {asset.get("asset_name", "")}', fontsize=10,
                     fontweight="bold", color=C["txt"], pad=10)
        if badge.strip():
            ax.text(0.98, 0.94, badge.strip(), transform=ax.transAxes, ha="right",
                    fontsize=8, color=C["accent"],
                    bbox=dict(boxstyle="round,pad=0.3", fc=C["panel"], ec=C["border"], alpha=0.9))

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", rotation=30)

        # legend
        lines1, lab1 = ax.get_legend_handles_labels()
        lines2, lab2 = ax2.get_legend_handles_labels()
        if lines1 or lines2:
            ax.legend(lines1 + lines2, lab1 + lab2, loc="upper left")

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUTPUT_DIR / f'event_{event["event_id"]}_grid.png'
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------- Plot B: all-assets overlay (normalized % change) ----------

def plot_overlay(
    event: dict[str, Any],
    price_data: dict[str, pd.DataFrame],
    prob_df: pd.DataFrame,
    market_q: str,
    outcome: str | None,
    did_happen: bool | None,
) -> Path:
    fig, ax = plt.subplots(figsize=(16, 7))

    status = "YES" if did_happen is True else ("NO" if did_happen is False else "PENDING")
    status_color = C["pos"] if did_happen is True else (C["neg"] if did_happen is False else C["txt2"])

    fig.suptitle(
        f'{event["event_title"]}',
        fontsize=16, fontweight="bold", color=C["txt"], y=0.98,
    )
    fig.text(
        0.5, 0.94,
        f'Normalized Price Change (%) vs Probability   |   Outcome: {status}',
        ha="center", fontsize=11, color=status_color, fontstyle="italic",
    )

    # Plot each asset as normalised % change
    for idx, asset in enumerate(event["assets"]):
        sym = asset["symbol"]
        pdf = price_data.get(sym, pd.DataFrame(columns=["datetime", "close"]))
        if pdf.empty or len(pdf) < 2:
            continue
        pct = (pdf["close"] / pdf["close"].iloc[0] - 1) * 100
        color = ASSET_COLORS[idx % len(ASSET_COLORS)]
        ax.plot(pdf["datetime"], pct, color=color, linewidth=1.6, alpha=0.85, label=sym)

    ax.set_ylabel("Price Change (%)", color=C["txt"], fontsize=11)
    ax.axhline(0, color=C["grid"], linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.tick_params(axis="x", rotation=30)

    # Probability on right axis
    ax2 = ax.twinx()
    if not prob_df.empty:
        pd_dates = prob_df["datetime"].copy()
        if pd_dates.dt.tz is not None:
            pd_dates = pd_dates.dt.tz_localize(None)
        probs = prob_df["probability"] * 100
        ax2.plot(pd_dates, probs, color=C["prob"], linewidth=2.5, linestyle="-",
                 label="Probability", zorder=5)
        ax2.fill_between(pd_dates, probs, 0, color=C["prob_fill"], alpha=0.10, zorder=4)
        ax2.set_ylabel("Probability (%)", color=C["prob"], fontsize=11)
        ax2.tick_params(axis="y", colors=C["prob"])
        ax2.set_ylim(0, 105)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=3, framealpha=0.9)

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    out = OUTPUT_DIR / f'event_{event["event_id"]}_overlay.png'
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------- Plot C: correlation bar chart per event -------------------

def plot_correlation_bars(
    event: dict[str, Any],
    price_data: dict[str, pd.DataFrame],
    prob_df: pd.DataFrame,
    market_q: str,
) -> Path:
    labels, vals, colors = [], [], []
    for asset in event["assets"]:
        sym = asset["symbol"]
        pdf = price_data.get(sym, pd.DataFrame(columns=["datetime", "close"]))
        r = _corr(pdf, prob_df)
        if r is not None:
            labels.append(sym)
            vals.append(r)
            colors.append(C["pos"] if r >= 0 else C["neg"])

    if not labels:
        return OUTPUT_DIR / "empty.png"

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, vals, color=colors, edgecolor=C["border"], linewidth=0.6, width=0.6, zorder=3)
    ax.axhline(0, color=C["txt2"], linewidth=0.8)
    ax.set_ylabel("Correlation (r)", fontsize=11, color=C["txt"])
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(
        f'Price-Probability Correlation  |  {event["event_title"]}',
        fontsize=13, fontweight="bold", color=C["txt"], pad=12,
    )
    # Value labels on bars
    for bar, v in zip(bars, vals):
        y = v + 0.04 if v >= 0 else v - 0.08
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{v:+.2f}",
                ha="center", fontsize=9, color=C["txt"], fontweight="bold")

    fig.tight_layout()
    out = OUTPUT_DIR / f'event_{event["event_id"]}_corr.png'
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


# ===================================================================
# 5.  MAIN
# ===================================================================

def process_event(event: dict[str, Any], idx: int, total: int) -> list[Path]:
    eid = event["event_id"]
    title = event["event_title"]
    start = event["created_at"]
    end = event["end_at"]

    print(f"\n{'='*80}")
    print(f"EVENT {idx}/{total}: {title}  (id={eid})")
    print(f"  Period: {start.date() if start else '?'} -> {end.date() if end else '?'}")
    print(f"  Assets: {len(event['assets'])}")

    if start is None or end is None:
        print("  SKIP - missing dates")
        return []

    # -- Polymarket markets -----------------------------------------------
    print("  Fetching Polymarket markets...")
    try:
        markets = fetch_event_markets(eid)
    except Exception as e:
        print(f"  SKIP - Polymarket API error: {e}")
        return []

    correct = find_correct_market(markets)
    if correct is None:
        print("  SKIP - no correct market found")
        return []

    mkt_q = correct["question"]
    outcome, did_happen = resolve_market(correct)
    token = yes_token_id(correct)
    status = "YES" if did_happen is True else ("NO" if did_happen is False else "PENDING")

    print(f"  Market:  {mkt_q[:70]}")
    print(f"  Outcome: {status}")

    if token is None:
        print("  SKIP - no YES token")
        return []

    # -- Probability history ----------------------------------------------
    print("  Fetching probability history...")
    prob_df = fetch_prob_history(token, start, end)
    print(f"  -> {len(prob_df)} daily probability points")

    # -- Asset prices -----------------------------------------------------
    print("  Fetching asset prices...")
    price_data: dict[str, pd.DataFrame] = {}
    for asset in event["assets"]:
        sym = asset["symbol"]
        pdf = fetch_prices(sym, start, end)
        price_data[sym] = pdf
        n_bars = len(pdf)
        if n_bars > 0:
            pct = (pdf["close"].iloc[-1] / pdf["close"].iloc[0] - 1) * 100
            print(f"    {sym:>6s}: {n_bars:3d} bars  {pct:+.1f}%")
        else:
            print(f"    {sym:>6s}:  no data")

    # -- Generate plots ---------------------------------------------------
    print("  Generating plots...")
    plots: list[Path] = []

    p1 = plot_asset_grid(event, price_data, prob_df, mkt_q, outcome, did_happen)
    plots.append(p1)
    print(f"    Grid:    {p1.name}")

    p2 = plot_overlay(event, price_data, prob_df, mkt_q, outcome, did_happen)
    plots.append(p2)
    print(f"    Overlay: {p2.name}")

    p3 = plot_correlation_bars(event, price_data, prob_df, mkt_q)
    plots.append(p3)
    print(f"    Corr:    {p3.name}")

    return plots


def main() -> None:
    _setup_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  Event-Asset Price vs Polymarket Probability Analyzer")
    print("  Loading events from DB...")
    print("=" * 80)

    events = asyncio.run(load_events_from_db())
    if not events:
        print("No events found in event_asset_llm_review. Run build_relevant_event_asset_groups.py first.")
        sys.exit(1)

    print(f"Found {len(events)} events to analyse")

    all_plots: list[Path] = []
    for idx, ev in enumerate(events, 1):
        plots = process_event(ev, idx, len(events))
        all_plots.extend(plots)

    # -- Final summary ----------------------------------------------------
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"  Events processed: {len(events)}")
    print(f"  Plots generated:  {len(all_plots)}")
    for p in all_plots:
        print(f"    {p.name}")
    print(f"  Output dir: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
