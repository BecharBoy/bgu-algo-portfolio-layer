
import httpx
import re
import json
import time
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from pathlib import Path

CLOB_API  = "https://clob.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"

EVENTS_TO_BACKTEST = [
    ("2003524",  "GOOGL", "GOOGL Earnings Beat",           "2026-04-17", "2026-04-30"),
    ("2003532",  "META",  "META Earnings Beat",            "2026-04-17", "2026-04-30"),
    ("1819243",  "LNG",   "QatarEnergy LNG Production",    "2026-04-02", "2026-05-05"),
    ("1540766",  "USO",   "Strait of Hormuz Normal",       "2026-03-10", "2026-05-01"),
    ("2036399",  "USO",   "Iran Ceasefire Extended Apr22", "2026-04-08", "2026-04-23"),
    ("1696325",  "USO",   "US Seizes Oil Tanker Apr15",    "2026-03-15", "2026-04-16"),
]

OUTPUT_DIR = Path("backtest_data")
OUTPUT_DIR.mkdir(exist_ok=True)
PLOTS_DIR = Path("backtest_data/plots")
PLOTS_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1: Get full-precision token IDs
# ─────────────────────────────────────────────────────────────
def get_tokens(market_id: str) -> dict:
    print(f"  [tokens] market {market_id} ...", end=" ")
    r = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=30)
    if r.status_code != 200:
        print(f"FAILED (gamma {r.status_code})")
        return {}
    data = r.json()
    condition_id = data.get("conditionId")
    raw_tokens = data.get("clobTokenIds")
    if raw_tokens:
        try:
            token_list = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
            if isinstance(token_list, list) and len(token_list) >= 2:
                print(f"OK via Gamma ({len(token_list)} tokens)")
                r2 = httpx.get(f"{CLOB_API}/markets/{condition_id}", timeout=30)
                if r2.status_code == 200:
                    outcomes = re.findall(r'"outcome"\s*:\s*"([^"]+)"', r2.text)
                    clob_tokens = re.findall(r'"token_id"\s*:\s*"([^"]+)"', r2.text)
                    if outcomes and clob_tokens:
                        return dict(zip(outcomes, clob_tokens))
                return {"Yes": token_list[0], "No": token_list[1]}
        except Exception as e:
            print(f"parse error: {e}")
    if condition_id:
        r2 = httpx.get(f"{CLOB_API}/markets/{condition_id}", timeout=30)
        if r2.status_code == 200:
            outcomes = re.findall(r'"outcome"\s*:\s*"([^"]+)"', r2.text)
            clob_tokens = re.findall(r'"token_id"\s*:\s*"([^"]+)"', r2.text)
            if outcomes and clob_tokens:
                print(f"OK via CLOB")
                return dict(zip(outcomes, clob_tokens))
    print(f"FAILED (no tokens found)")
    return {}


# ─────────────────────────────────────────────────────────────
# STEP 2: Pull Polymarket probability timeseries
# ─────────────────────────────────────────────────────────────
def get_poly_history(token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    print(f"  [poly] token={token_id[:20]}...")
    all_history = []
    CHUNK = 10 * 86400
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(cursor + CHUNK, end_ts)
        for fidelity in [60, 360, 1440]:
            r = httpx.get(
                f"{CLOB_API}/prices-history",
                params={"market": token_id, "startTs": cursor,
                        "endTs": chunk_end, "fidelity": fidelity},
                timeout=30,
            )
            if r.status_code == 200:
                chunk = r.json().get("history", [])
                if chunk:
                    all_history.extend(chunk)
                    break
            time.sleep(0.15)
        cursor = chunk_end + 1

    if all_history:
        df = pd.DataFrame(all_history)
        df["dt"]   = pd.to_datetime(df["t"].astype(float), unit="s", utc=True)
        df["prob"] = df["p"].astype(float) * 100
        df = df[["dt", "prob"]].set_index("dt").sort_index()
        df = df[~df.index.duplicated()]
        print(f"    poly: {len(df)} pts via chunked startTs/endTs")
        return df

    r = httpx.get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id, "interval": "max"},
        timeout=30,
    )
    if r.status_code == 200:
        history = r.json().get("history", [])
        if history:
            df = pd.DataFrame(history)
            df["dt"]   = pd.to_datetime(df["t"].astype(float), unit="s", utc=True)
            df["prob"] = df["p"].astype(float) * 100
            df = df[["dt", "prob"]].set_index("dt").sort_index()
            start_dt = pd.Timestamp(start_ts, unit="s", tz="UTC")
            end_dt   = pd.Timestamp(end_ts,   unit="s", tz="UTC")
            df = df[(df.index >= start_dt) & (df.index <= end_dt)]
            print(f"    poly: {len(df)} pts via interval=max (WARNING: may miss early days)")
            return df

    print(f"    poly: EMPTY (all methods failed)")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# STEP 3: Pull stock/ETF price — extended +7 days beyond event end
# ─────────────────────────────────────────────────────────────
def get_asset_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetches price data up to end + 7 calendar days to capture post-event drift."""
    end_extended = (datetime.fromisoformat(end) + timedelta(days=7)).strftime("%Y-%m-%d")
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=start, end=end_extended, interval="1h")
        if not df.empty:
            df.index = df.index.tz_convert("UTC")
            df = df[["Close"]].rename(columns={"Close": "price"})
            print(f"    asset ({ticker}): {len(df)} hourly pts (end extended to {end_extended})")
            return df
        df = tk.history(start=start, end=end_extended, interval="1d")
        if not df.empty:
            df.index = df.index.tz_convert("UTC")
            df = df[["Close"]].rename(columns={"Close": "price"})
            print(f"    asset ({ticker}): {len(df)} daily pts fallback (end extended to {end_extended})")
            return df
    except Exception as e:
        print(f"    asset ({ticker}): ERROR {e}")
    return pd.DataFrame()


# ─────────────────────────────────────────────────────────────
# STEP 4: Merge and compute key stats
# ─────────────────────────────────────────────────────────────
def compute_stats(poly_df: pd.DataFrame, asset_df: pd.DataFrame,
                  name: str, event_end: str):
    if poly_df.empty or asset_df.empty:
        return {}

    poly_h  = poly_df.resample("1h").last().ffill()
    asset_h = asset_df.resample("1h").last().ffill()

    event_end_dt = pd.Timestamp(event_end + "T23:59:59", tz="UTC")
    post_end_dt  = event_end_dt + timedelta(days=7)

    # Inner join for the event window only
    merged_event = poly_h.join(asset_h, how="inner").dropna()
    if merged_event.empty:
        return {}

    # Post-event stock data (no polymarket data here — fill prob as NaN)
    asset_post = asset_h[(asset_h.index > event_end_dt) & (asset_h.index <= post_end_dt)]
    if not asset_post.empty:
        post_df = asset_post.copy()
        post_df["prob"] = float("nan")   # no market data after resolution
        merged = pd.concat([merged_event, post_df[["prob", "price"]]])
    else:
        merged = merged_event

    merged = merged.sort_index()

    # Stats (computed on event window only)
    crossed_80 = merged_event[merged_event["prob"] >= 80]
    t_cross_80 = crossed_80.index[0] if not crossed_80.empty else None

    price_at_cross     = merged_event.loc[t_cross_80, "price"] if t_cross_80 else None
    price_at_event_end = merged_event["price"].iloc[-1]
    price_at_end       = merged["price"].iloc[-1]   # last point = +7d

    pct_move_after = None
    if price_at_cross and price_at_cross > 0:
        pct_move_after = round(
            (price_at_event_end - price_at_cross) / price_at_cross * 100, 2
        )

    pct_move_7d = None
    if price_at_event_end > 0:
        pct_move_7d = round(
            (price_at_end - price_at_event_end) / price_at_event_end * 100, 2
        )

    open_prob   = merged_event["prob"].iloc[0]
    hours_to_80 = None
    if t_cross_80:
        hours_to_80 = round(
            (t_cross_80 - merged_event.index[0]).total_seconds() / 3600, 1
        )

    stats = {
        "name":                    name,
        "open_prob":               round(open_prob, 1),
        "close_prob":              round(merged_event["prob"].iloc[-1], 1),
        "t_cross_80":              str(t_cross_80) if t_cross_80 else "never",
        "hours_to_80":             hours_to_80,
        "price_at_open":           round(merged_event["price"].iloc[0], 2),
        "price_at_cross80":        round(price_at_cross, 2) if price_at_cross else None,
        "price_at_event_end":      round(price_at_event_end, 2),
        "price_at_end_7d":         round(price_at_end, 2),
        "pct_move_after_80_cross": pct_move_after,
        "pct_move_7d_post_event":  pct_move_7d,
        "n_poly_pts":              len(poly_df),
        "n_asset_pts":             len(asset_df),
    }
    return stats, merged, event_end_dt


# ─────────────────────────────────────────────────────────────
# STEP 5: Plot — Plotly PNG (via kaleido) with matplotlib fallback
# ─────────────────────────────────────────────────────────────
def _plot_matplotlib(merged: pd.DataFrame, stats: dict, name: str,
                     ticker: str, event_end_dt: pd.Timestamp, out_path: Path):
    """Pure matplotlib — no extra deps beyond the standard stack."""
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    fig.patch.set_facecolor("#f7f6f2")
    ax1.set_facecolor("#f9f8f5")

    # Polymarket probability (left axis)
    ax1.fill_between(merged.index, merged["prob"], alpha=0.12, color="#01696f")
    ax1.plot(merged.index, merged["prob"], color="#01696f", linewidth=2.2,
             label="YES Probability (%)")
    ax1.set_ylabel("YES Probability (%)", color="#01696f", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#01696f")
    ax1.set_ylim(0, 108)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    # Stock price (right axis) — split into event window vs post-event
    ax2 = ax1.twinx()
    mask_event = merged.index <= event_end_dt
    mask_post  = merged.index >  event_end_dt
    if mask_event.any():
        ax2.plot(merged[mask_event].index, merged[mask_event]["price"],
                 color="#e07b00", linewidth=2, label=f"{ticker} (event window)")
    if mask_post.any():
        ax2.plot(merged[mask_post].index, merged[mask_post]["price"],
                 color="#a12c7b", linewidth=2, linestyle="--",
                 label=f"{ticker} (+7d post-event)")
    ax2.set_ylabel(f"{ticker} Price (USD)", color="#964219", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#964219")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.2f}"))

    # 80% cross vertical line
    if stats["t_cross_80"] != "never":
        t80 = pd.Timestamp(stats["t_cross_80"])
        ax1.axvline(t80, color="#006494", linewidth=1.4, linestyle="--")
        ax1.text(t80, 103, "80% cross", color="#006494", fontsize=9,
                 ha="center", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="#f9f8f5",
                           ec="#006494", alpha=0.85))

    # Event end vertical line
    ax1.axvline(event_end_dt, color="#964219", linewidth=1.4, linestyle="--")
    ax1.text(event_end_dt, 96, "Event end", color="#964219", fontsize=9,
             ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.2", fc="#f9f8f5",
                       ec="#964219", alpha=0.85))

    # 7-day shaded region
    ax1.axvspan(event_end_dt, event_end_dt + timedelta(days=7),
                alpha=0.07, color="#a12c7b")

    # Stats annotation box
    stats_text = (
        f"Open→Close prob: {stats['open_prob']}% → {stats['close_prob']}%\n"
        f"Hours to 80%: {stats['hours_to_80']}h\n"
        f"Δ after 80% cross: {stats['pct_move_after_80_cross']}%\n"
        f"7d post-event drift: {stats['pct_move_7d_post_event']}%"
    )
    ax1.text(0.01, 0.97, stats_text, transform=ax1.transAxes, fontsize=9,
             verticalalignment="top",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f9f8f5",
                       edgecolor="#dcd9d5", alpha=0.92))

    # X axis formatting
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30, ha="right")
    ax1.set_xlabel("Date (UTC)", fontsize=10)
    ax1.grid(axis="y", color="#dcd9d5", linewidth=0.6, linestyle="--")
    ax1.grid(axis="x", color="#dcd9d5", linewidth=0.4, linestyle=":")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower left",
               fontsize=9, framealpha=0.9, edgecolor="#dcd9d5")

    fig.suptitle(
        f"{name}  ·  {ticker} vs Polymarket YES Probability",
        fontsize=13, fontweight="bold", color="#28251d", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)


def _plot_plotly(merged: pd.DataFrame, stats: dict, name: str,
                 ticker: str, event_end_dt: pd.Timestamp, out_path: Path):
    """Plotly → PNG via kaleido (richer dual-axis styling)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=merged.index, y=merged["prob"],
            name="YES Probability (%)",
            line=dict(color="#01696f", width=2.5),
            fill="tozeroy", fillcolor="rgba(1,105,111,0.08)",
        ),
        secondary_y=False,
    )

    mask_event = merged.index <= event_end_dt
    mask_post  = merged.index >  event_end_dt
    if mask_event.any():
        fig.add_trace(
            go.Scatter(x=merged[mask_event].index, y=merged[mask_event]["price"],
                       name=f"{ticker} (event window)",
                       line=dict(color="#e07b00", width=2)),
            secondary_y=True,
        )
    if mask_post.any():
        fig.add_trace(
            go.Scatter(x=merged[mask_post].index, y=merged[mask_post]["price"],
                       name=f"{ticker} (+7d post-event)",
                       line=dict(color="#a12c7b", width=2, dash="dot")),
            secondary_y=True,
        )

    if stats["t_cross_80"] != "never":
        t80 = pd.Timestamp(stats["t_cross_80"])
        fig.add_vline(x=t80.value / 1e6, line_width=1.5, line_dash="dash",
                      line_color="#006494", annotation_text="80% cross",
                      annotation_position="top right",
                      annotation_font_size=11, annotation_font_color="#006494")

    fig.add_vline(x=event_end_dt.value / 1e6, line_width=1.5, line_dash="dash",
                  line_color="#964219", annotation_text="Event end",
                  annotation_position="top left",
                  annotation_font_size=11, annotation_font_color="#964219")

    fig.add_vrect(x0=event_end_dt, x1=event_end_dt + timedelta(days=7),
                  fillcolor="rgba(161,44,123,0.06)", layer="below", line_width=0,
                  annotation_text="7-day drift", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#a12c7b")

    stats_text = (
        f"Open→Close prob: {stats['open_prob']}% → {stats['close_prob']}%<br>"
        f"Hours to 80%: {stats['hours_to_80']}h<br>"
        f"Δ after 80% cross: {stats['pct_move_after_80_cross']}%<br>"
        f"7d post-event drift: {stats['pct_move_7d_post_event']}%"
    )
    fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99,
                       text=stats_text, showarrow=False, align="left",
                       font=dict(size=11, color="#28251d"),
                       bgcolor="rgba(249,248,245,0.85)",
                       bordercolor="#dcd9d5", borderwidth=1, borderpad=8)

    fig.update_layout(
        title=dict(
            text=f"<b>{name}</b>  ·  {ticker} vs Polymarket YES probability",
            font=dict(size=15),
        ),
        plot_bgcolor="#f9f8f5", paper_bgcolor="#f7f6f2",
        font=dict(family="'Inter','Helvetica Neue',sans-serif",
                  color="#28251d", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(249,248,245,0.9)",
                    bordercolor="#dcd9d5", borderwidth=1),
        hovermode="x unified",
        margin=dict(l=60, r=60, t=80, b=60),
        width=1100, height=520,
        xaxis=dict(title="Date (UTC)", gridcolor="#dcd9d5",
                   tickformat="%b %d", zeroline=False),
    )
    fig.update_yaxes(title_text="YES Probability (%)", secondary_y=False,
                     range=[0, 105], gridcolor="#dcd9d5",
                     zeroline=False, ticksuffix="%")
    fig.update_yaxes(title_text=f"{ticker} Price (USD)", secondary_y=True,
                     gridcolor="rgba(0,0,0,0)", zeroline=False, tickprefix="$")

    fig.write_image(str(out_path), scale=2)  # PNG @ 2× resolution


def plot_event(merged: pd.DataFrame, stats: dict, name: str, ticker: str,
               event_end_dt: pd.Timestamp, market_id: str) -> str:
    """
    Saves a dual-axis PNG (Polymarket prob + stock price).
    Primary: Plotly + kaleido  →  falls back to matplotlib if kaleido missing.
    """
    out_path = PLOTS_DIR / f"{market_id}_{ticker}_chart.png"
    try:
        import kaleido  # noqa: F401
        _plot_plotly(merged, stats, name, ticker, event_end_dt, out_path)
        print(f"    plot saved (Plotly/kaleido) → {out_path}")
    except ImportError:
        print("    kaleido not found — using matplotlib instead")
        _plot_matplotlib(merged, stats, name, ticker, event_end_dt, out_path)
        print(f"    plot saved (matplotlib) → {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def to_unix(d: str) -> int:
    return int(datetime.fromisoformat(d + "T00:00:00+00:00").timestamp())


all_stats  = []
plot_paths = []

for market_id, ticker, name, start, end in EVENTS_TO_BACKTEST:
    print(f"\n{'='*60}")
    print(f"Event: {name}  |  market={market_id}  |  asset={ticker}")
    print(f"{'='*60}")

    tokens = get_tokens(market_id)
    if not tokens:
        print("  SKIP — no tokens")
        continue

    yes_token = tokens.get("Yes")
    if not yes_token:
        print(f"  SKIP — no Yes token (found: {list(tokens.keys())})")
        continue

    poly_df  = get_poly_history(yes_token, to_unix(start), to_unix(end) + 86400)
    asset_df = get_asset_price(ticker, start, end)   # internally +7 days

    if not poly_df.empty:
        poly_df.to_csv(OUTPUT_DIR / f"{market_id}_poly.csv")
    if not asset_df.empty:
        asset_df.to_csv(OUTPUT_DIR / f"{market_id}_asset_{ticker}.csv")

    result = compute_stats(poly_df, asset_df, name, end)
    if result:
        stats, merged, event_end_dt = result
        all_stats.append(stats)
        merged.to_csv(OUTPUT_DIR / f"{market_id}_merged.csv")

        print(f"  open_prob={stats['open_prob']}%  ->  close={stats['close_prob']}%")
        print(f"  hours_to_80={stats['hours_to_80']}h")
        print(f"  price_at_open={stats['price_at_open']}  cross80={stats['price_at_cross80']}"
              f"  event_end={stats['price_at_event_end']}  +7d={stats['price_at_end_7d']}")
        print(f"  pct_move_after_80_cross={stats['pct_move_after_80_cross']}%"
              f"   7d_post_event={stats['pct_move_7d_post_event']}%")

        p = plot_event(merged, stats, name, ticker, event_end_dt, market_id)
        plot_paths.append(p)
    else:
        print("  SKIP — could not compute stats")

    time.sleep(0.5)


# ─────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
if all_stats:
    summary = pd.DataFrame(all_stats)
    print(summary.to_string(index=False))
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"\nSaved to: {OUTPUT_DIR}/summary.csv")
    print(f"\nPlots ({len(plot_paths)} PNG files):")
    for p in plot_paths:
        print(f"  {p}")
else:
    print("No results.")
