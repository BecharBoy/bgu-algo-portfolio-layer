import json
import csv
from datetime import datetime, timezone

def parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except:
        return None

def duration_days(created, end):
    c = parse_dt(created)
    e = parse_dt(end)
    if not c or not e:
        return None
    return round((e - c).total_seconds() / 86400, 1)

with open("polymarket_oracle.json", "r", encoding="utf-8") as f:
    events = json.load(f)

# ── Two outputs ──────────────────────────────────────────────────────────────
# 1. events.csv     — one row per event (high-level overview)
# 2. markets.csv    — one row per sub-market (outcomes, probabilities, volume)

EVENT_FIELDS = [
    "event_id", "title", "category", "tags",
    "createdAt", "startDate", "endDate", "closedTime", "duration_days",
    "active", "closed", "archived", "competitive",
    "volume", "volume24hr", "volume1wk", "volume1mo", "volume1yr",
    "openInterest", "liquidityAmm", "liquidityClob",
    "n_markets",
]

MARKET_FIELDS = [
    # event context
    "event_id", "event_title", "category", "tags",
    "event_createdAt", "event_endDate", "event_duration_days",
    "event_active", "event_closed",
    "event_volume", "event_volume24hr", "event_liquidityAmm", "event_liquidityClob",
    # sub-market fields
    "market_id", "question", "groupItemTitle", "groupItemThreshold", "formatType", "marketType",
    "outcomes", "outcomePrices",           # JSON strings — parse when needed
    "clobTokenIds",                         # use these for price history API
    "market_volume", "market_volume24hr", "market_volume1wk",
    "market_liquidityAmm", "market_liquidityClob",
    "active", "closed", "closedTime",
    "startDate", "endDate",
    "lastTradePrice", "bestBid", "bestAsk",
    "oneDayPriceChange", "oneWeekPriceChange",
    "umaResolutionStatus", "resolvedBy",
]

with open("oracle_events.csv", "w", newline="", encoding="utf-8") as ef, \
     open("oracle_markets.csv", "w", newline="", encoding="utf-8") as mf:

    e_writer = csv.DictWriter(ef, fieldnames=EVENT_FIELDS, extrasaction="ignore")
    m_writer = csv.DictWriter(mf, fieldnames=MARKET_FIELDS, extrasaction="ignore")
    e_writer.writeheader()
    m_writer.writeheader()

    for ev in events:
        tags = ev.get("tags") or []
        # tags can be list of dicts {"label":...} or already a plain string
        if isinstance(tags, list):
            tag_str = "|".join(t.get("label", "") for t in tags if isinstance(t, dict))
        else:
            tag_str = str(tags)

        markets = ev.get("markets") or []
        dur = duration_days(ev.get("createdAt"), ev.get("endDate"))

        # ── Event row ────────────────────────────────────────────
        e_writer.writerow({
            "event_id":     ev.get("event_id") or ev.get("id"),
            "title":        ev.get("title"),
            "category":     ev.get("category"),
            "tags":         tag_str,
            "createdAt":    ev.get("createdAt"),
            "startDate":    ev.get("startDate"),
            "endDate":      ev.get("endDate"),
            "closedTime":   ev.get("closedTime"),
            "duration_days": dur,
            "active":       ev.get("active"),
            "closed":       ev.get("closed"),
            "archived":     ev.get("archived"),
            "competitive":  ev.get("competitive"),
            "volume":       ev.get("volume"),
            "volume24hr":   ev.get("volume24hr"),
            "volume1wk":    ev.get("volume1wk"),
            "volume1mo":    ev.get("volume1mo"),
            "volume1yr":    ev.get("volume1yr"),
            "openInterest": ev.get("openInterest"),
            "liquidityAmm": ev.get("liquidityAmm"),
            "liquidityClob":ev.get("liquidityClob"),
            "n_markets":    len(markets),
        })

        # ── Market rows (one per sub-market) ─────────────────────
        for m in markets:
            m_writer.writerow({
                # event context repeated on every row for easy filtering
                "event_id":             ev.get("event_id") or ev.get("id"),
                "event_title":          ev.get("title"),
                "category":             ev.get("category"),
                "tags":                 tag_str,
                "event_createdAt":      ev.get("createdAt"),
                "event_endDate":        ev.get("endDate"),
                "event_duration_days":  dur,
                "event_active":         ev.get("active"),
                "event_closed":         ev.get("closed"),
                "event_volume":         ev.get("volume"),
                "event_volume24hr":     ev.get("volume24hr"),
                "event_liquidityAmm":   ev.get("liquidityAmm"),
                "event_liquidityClob":  ev.get("liquidityClob"),
                # sub-market
                "market_id":            m.get("market_id"),
                "question":             m.get("question"),
                "groupItemTitle":       m.get("groupItemTitle"),
                "groupItemThreshold":   m.get("groupItemThreshold"),
                "formatType":           m.get("formatType"),
                "marketType":           m.get("marketType"),
                "outcomes":             m.get("outcomes"),
                "outcomePrices":        m.get("outcomePrices"),
                "clobTokenIds":         m.get("clobTokenIds"),
                "market_volume":        m.get("volume"),
                "market_volume24hr":    m.get("volume24hr"),
                "market_volume1wk":     m.get("volume1wk"),
                "market_liquidityAmm":  m.get("liquidityAmm"),
                "market_liquidityClob": m.get("liquidityClob"),
                "active":               m.get("active"),
                "closed":               m.get("closed"),
                "closedTime":           m.get("closedTime"),
                "startDate":            m.get("startDate"),
                "endDate":              m.get("endDate"),
                "lastTradePrice":       m.get("lastTradePrice"),
                "bestBid":              m.get("bestBid"),
                "bestAsk":              m.get("bestAsk"),
                "oneDayPriceChange":    m.get("oneDayPriceChange"),
                "oneWeekPriceChange":   m.get("oneWeekPriceChange"),
                "umaResolutionStatus":  m.get("umaResolutionStatus"),
                "resolvedBy":           m.get("resolvedBy"),
            })

print(f"Done.")
print(f"  oracle_events.csv  — {len(events)} events")
print(f"  oracle_markets.csv — one row per sub-market outcome")