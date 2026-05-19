import httpx
import json
from datetime import datetime, timezone

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
GAMMA_API  = "https://gamma-api.polymarket.com/events/keyset"
START_DATE = "2026-01-01T00:00:00Z"   # only 2026 events
MIN_DURATION_DAYS = 5

# Tag slugs to fetch — one request per tag, merged & deduped by event ID
# Slugs are lowercase-hyphenated versions of the tag labels
TARGET_TAG_SLUGS = [
    "equities",
    "earnings",
    "kpis",
    "economy",
    "macro-indicators",
    "business",
    "monthly",
    "hit-price",
    "finance-updown",
    "pyth-finance",
    "stocks",
    "geopolitics",
    "oil",
    "iran",
    "us-x-iran",
    "strait-of-hormuz",
    "ai",
    "big-tech",
    "tech",
    "privates",
]

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def duration_days(event: dict) -> float | None:
    created = parse_dt(event.get("createdAt"))
    end     = parse_dt(event.get("endDate"))
    if created and end:
        return (end - created).total_seconds() / 86400
    return None


def extract_markets(raw_markets: list) -> list[dict]:
    """Pull only the fields we need from each sub-market."""
    out = []
    for m in raw_markets or []:
        out.append({
            "market_id":         m.get("id"),
            "question":          m.get("question"),
            "outcomes":          m.get("outcomes"),          # JSON string e.g. '["Yes","No"]'
            "outcomePrices":     m.get("outcomePrices"),     # JSON string e.g. '[0.82,0.18]'
            "clobTokenIds":      m.get("clobTokenIds"),      # token IDs for price history API
            "volume":            m.get("volumeNum"),
            "volume24hr":        m.get("volume24hr"),
            "volume1wk":         m.get("volume1wk"),
            "liquidityAmm":      m.get("liquidityAmm"),
            "liquidityClob":     m.get("liquidityClob"),
            "active":            m.get("active"),
            "closed":            m.get("closed"),
            "closedTime":        m.get("closedTime"),
            "startDate":         m.get("startDate"),
            "endDate":           m.get("endDate"),
            "lastTradePrice":    m.get("lastTradePrice"),
            "bestBid":           m.get("bestBid"),
            "bestAsk":           m.get("bestAsk"),
            "oneDayPriceChange": m.get("oneDayPriceChange"),
            "oneWeekPriceChange":m.get("oneWeekPriceChange"),
            "umaResolutionStatus": m.get("umaResolutionStatus"),
            "resolvedBy":        m.get("resolvedBy"),
            "groupItemTitle":    m.get("groupItemTitle"),    # e.g. "0-5 posts", "Beat", "Miss"
            "groupItemThreshold":m.get("groupItemThreshold"),
            "formatType":        m.get("formatType"),
            "marketType":        m.get("marketType"),
        })
    return out


def extract_tags(event: dict) -> str:
    tags = event.get("tags") or []
    return "|".join(t.get("label", "") for t in tags if t.get("label"))


def fetch_tag(client: httpx.Client, tag_slug: str) -> list[dict]:
    """Paginate through all events for a given tag slug."""
    events = []
    cursor = None
    page   = 0

    while True:
        params = {
            "limit":         500,
            "tag_slug":      tag_slug,
            "start_date_min": START_DATE,
            "order":         "createdAt",
            "ascending":     "false",
        }
        if cursor:
            params["after_cursor"] = cursor

        try:
            r = client.get(GAMMA_API, params=params, timeout=30)
            r.raise_for_status()
        except httpx.HTTPStatusError as e:
            print(f"  HTTP {e.response.status_code} on tag={tag_slug} page={page}")
            break

        body   = r.json()
        batch  = body.get("events", [])
        cursor = body.get("next_cursor")
        page  += 1

        for event in batch:
            dur = duration_days(event)
            if dur is None or dur <= MIN_DURATION_DAYS:
                continue

            events.append({
                # ── Event-level fields ──────────────────────────────
                "event_id":       event.get("id"),
                "title":          event.get("title"),
                "slug":           event.get("slug"),
                "description":    event.get("description"),
                "category":       event.get("category"),
                "tags":           extract_tags(event),

                # ── Dates ────────────────────────────────────────────
                "createdAt":      event.get("createdAt"),
                "startDate":      event.get("startDate"),
                "endDate":        event.get("endDate"),
                "closedTime":     event.get("closedTime"),
                "duration_days":  round(dur, 1),

                # ── Status ───────────────────────────────────────────
                "active":         event.get("active"),
                "closed":         event.get("closed"),
                "archived":       event.get("archived"),
                "competitive":    event.get("competitive"),

                # ── Volume & Liquidity (event level) ─────────────────
                "volume":         event.get("volume"),
                "volume24hr":     event.get("volume24hr"),
                "volume1wk":      event.get("volume1wk"),
                "volume1mo":      event.get("volume1mo"),
                "volume1yr":      event.get("volume1yr"),
                "openInterest":   event.get("openInterest"),
                "liquidityAmm":   event.get("liquidityAmm"),
                "liquidityClob":  event.get("liquidityClob"),

                # ── Sub-markets (full detail) ─────────────────────────
                "markets":        extract_markets(event.get("markets", [])),
            })

        print(f"  [{tag_slug}] page {page}: +{len(batch)} events fetched (kept after filter: running)")

        if not cursor or not batch:
            break

    return events


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def fetch_and_filter_markets() -> list[dict]:
    all_events: dict[str, dict] = {}   # keyed by event_id to deduplicate

    with httpx.Client() as client:
        for tag_slug in TARGET_TAG_SLUGS:
            print(f"\nFetching tag: {tag_slug}")
            tag_events = fetch_tag(client, tag_slug)
            for ev in tag_events:
                all_events[ev["event_id"]] = ev  # last write wins (same event, doesn't matter)
            print(f"  → {len(tag_events)} qualifying events for tag '{tag_slug}'")

    result = list(all_events.values())
    print(f"\n{'='*60}")
    print(f"TOTAL unique events (duration > {MIN_DURATION_DAYS}d): {len(result)}")
    print(f"{'='*60}")
    return result


if __name__ == "__main__":
    events = fetch_and_filter_markets()

    # ── Save full JSON (preserves nested markets array) ──────
    with open("polymarket_oracle.json", "w", encoding="utf-8") as f:
        json.dump(events, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(events)} events → polymarket_oracle.json")

    # ── Print quick summary ──────────────────────────────────
    for ev in sorted(events, key=lambda x: x["volume"] or 0, reverse=True)[:30]:
        n_markets = len(ev["markets"])
        print(
            f"[{ev['duration_days']:.0f}d | vol={ev['volume']:,.0f} | "
            f"{n_markets} sub-markets] {ev['title']}"
        )