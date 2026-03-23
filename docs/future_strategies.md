# 04_FUTURE_STRATEGIES.md

## What This Document Is

A deep-dive into every new strategy idea proposed for
my_traders. For each strategy: the idea explained simply,
the evidence it can work, the evidence it cannot, and the
best implementation approach. Written for someone with zero
background.

---

## 1. Sentiment Strategy — The Core Idea

### What Is It?

Markets often react to news before prices fully reflect it.
If you can read the emotional tone of breaking news, social
media, or political statements faster than most market
participants can act, you have a temporary edge.

A sentiment strategy works in 4 steps:
1. Collect text: tweets, Reddit posts, news headlines
2. Score each text piece: positive (+1.0) to negative (-1.0)
3. Aggregate scores by relevant asset (e.g., Trump tweets
   about oil → CVX, XOM get a sentiment score)
4. When aggregate score crosses a threshold → trade

### Why It Can Work

Social media sentiment demonstrably leads price movements
by 1-2 days for certain assets. Twitter/X multilingual
geopolitical sentiment Granger-causes crude oil prices at
the 5% significance level in published research. Studies
confirm that BERT-based sentiment scoring on Twitter improved
WTI crude oil price direction prediction from 57% accuracy
(price-only model) to 71% (with sentiment added).

**Research:**
Bollen, Mao & Zeng (2011), "Twitter mood predicts the
stock market", Journal of Computational Science 2(1).
https://doi.org/10.1016/j.jocs.2010.12.007

Jsomer et al. (2025), "Multilingual X/Twitter sentiment
analysis of geopolitical risk and crude oil markets."
https://jsomer.org/index.php/pub/article/view/23

### Why It Cannot Work

1. **Signal decay:** Sentiment-based alpha from 2011 was
   much stronger than today. Once widely used, the edge
   disappears as everyone trades on the same signal.

2. **Data quality:** Bots, sarcasm, and coordinated
   manipulation all corrupt the signal. "Oil is absolutely
   destroying portfolios" — is that positive or negative
   for oil?

3. **Latency:** By the time you scrape, score, and
   generate a signal, professional traders with direct
   data feeds have already moved. This works for daily
   strategies (overnight news → next-day trade) but
   NOT for intraday.

4. **Regime sensitivity:** Sentiment only drives prices
   during active geopolitical events. In calm periods the
   relationship is weak. You need a regime detector.

### Best Implementation

Keep sentiment as an **overlay** — it amplifies or
suppresses signals from other strategies, not replaces them:

```
[Scrapers] → [Scorer] → [signal_overrides table in DB]
                                ↓
              Portfolio.run_cycle reads overrides
              before placing orders
```

---

## 2. Crude Oil / Geopolitical Cointegration

### The Idea

This is a **text-price cointegration** model. The question:
does a daily aggregate of geopolitical sentiment (from
Trump posts, Middle East news, OPEC announcements) share
a stationary long-run relationship with WTI crude oil prices?

Let:
- Pₜ = WTI crude price at time t
- Sₜ = daily aggregate sentiment score on: Iran, OPEC,
       oil sanctions, Middle East conflict

Test: Is [Pₜ - β·Sₜ - α] stationary? If yes, they are
cointegrated and the spread between them is tradeable.

### The Required First Step: Granger Causality

Before assuming cointegration, test if Sₜ Granger-causes
Pₜ. The Granger test fits:

    Pₜ = Σⱼ aⱼ Pₜ₋ⱼ + Σⱼ bⱼ Sₜ₋ⱼ + εₜ

And tests H₀: b₁ = b₂ = ... = bₖ = 0 (sentiment adds
nothing beyond past prices). Reject H₀ → sentiment leads
price. Published result: daily Twitter sentiment on
geopolitical risk Granger-causes crude oil at lag 1-2
days, F-statistic significant at p < 0.05.

### Asset Mapping (No Futures Account Needed)

Direct crude trading requires a futures account (CL=F).
Equity proxies that move with crude:
- **Upstream producers:** CVX, XOM, COP — move 0.7-0.9
  correlation with WTI
- **Energy ETF:** XLE — diversified exposure, less volatile
- **Inverse:** When crude falls, airlines rise (UAL, DAL)

**Research:**
Jsomer et al. (2025), ibid.
Kilian, L. (2009), "Not All Oil Price Shocks Are Alike",
American Economic Review 99(3).
https://doi.org/10.1257/aer.99.3.1053

---

## 3. Data Sources — What to Scrape and How

### 3.1 Reddit — Start Here (Free, Stable API)

Reddit has an official Python library (`praw`) with a
generous free tier. No risk of being blocked for normal
research use.

**Best subreddits for signals:**
- `r/wallstreetbets` — retail momentum, meme stocks
- `r/investing` — longer-horizon discussion
- `r/geopolitics` — world events with market impact
- `r/energy` — oil, gas, commodity discussion

```python
import praw

reddit = praw.Reddit(
    client_id="YOUR_ID",
    client_secret="YOUR_SECRET",
    user_agent="my_traders_sentiment/1.0"
)

def fetch_reddit_posts(subreddit: str, limit: int = 100) -> list[dict]:
    sub = reddit.subreddit(subreddit)
    return [
        {
            "title":   post.title,
            "body":    post.selftext,
            "score":   post.score,
            "created": post.created_utc,
        }
        for post in sub.new(limit=limit)
    ]
```

**Signal quality:** WSB sentiment has statistically
significant predictive power for meme/volatile stocks
(GME, AMC, MEME ETF) but weak signal for S&P 500 large
caps. Strongest signal for mid-cap volatile names.

**Research:** Bollen et al. (2011), ibid. Shen, Shafiq &
Mian (2022), "Short-term stock market price trend prediction
using a comprehensive deep learning system", Journal of
Big Data.
https://doi.org/10.1186/s40537-022-00677-x

### 3.2 News RSS Feeds — Most Reliable Signal

RSS feeds are free, structured, timestamped, and require
zero authentication. Articles are more deliberate than
tweets — higher signal quality per item.

```python
import feedparser

FEEDS = {
    "reuters":  "https://feeds.reuters.com/reuters/businessNews",
    "ap":       "https://feeds.finance.yahoo.com/rss/2.0/headline",
}

def fetch_news(url: str) -> list[dict]:
    feed = feedparser.parse(url)
    return [
        {
            "title":     e.title,
            "summary":   e.summary,
            "published": e.published,
        }
        for e in feed.entries
    ]
```

### 3.3 X/Twitter — Hardest, Most Valuable for Specific Accounts

**Reality in 2026:** Free API tier is heavily restricted.
Full firehose requires enterprise pricing ($42,000/month).

**Practical approach:** Monitor ONLY low-volume,
high-impact accounts:
- @realDonaldTrump (Truth Social cross-posts)
- @WhiteHouse
- @OPECSecretariat
- @SecEnergy

These accounts post dozens of times per day at most —
easily within the free Basic tier (10,000 tweets/month).
A single Trump tweet about oil sanctions can move CVX ±3%
within hours.

### 3.4 Truth Social — Trump Specifically

No official API exists. Three options in 2026:

**Option 1 — RSS feed (simplest, fragile):**
```python
import feedparser

def fetch_truth_social(handle: str) -> list[dict]:
    url = f"https://truthsocial.com/@{handle}/feed.rss"
    feed = feedparser.parse(url)
    return [
        {"text": e.title, "published": e.published}
        for e in feed.entries
    ]
```

**Option 2 — Playwright headless browser:**
Automate login and timeline scraping. Requires a real
Truth Social account. More reliable but more maintenance.

**Option 3 — truth_social_api Python package:**
Unofficial wrapper around public endpoints. Works until
Truth Social changes their API structure.

**Recommendation:** Start with Option 1 (RSS). Low volume
(~5-15 posts/day from Trump) means RSS is sufficient.
If RSS breaks, upgrade to Playwright.

### 3.5 Polymarket — Highest Signal-to-Effort Ratio

Polymarket is a prediction market where users bet real money
on whether real-world events will happen. The price of a
contract (0-100%) represents the crowd's probability
estimate — aggregating information from sophisticated
global traders into a single clean number.

This is more information-dense than 10,000 tweets.
A contract "Iran nuclear deal by Q3 2026" at 8% is a clear,
liquid, crowd-verified signal.

**API:** Free, no authentication required.

```python
import httpx

async def get_polymarket_markets(keyword: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://gamma-api.polymarket.com/markets",
            params={"search": keyword, "active": True}
        )
    data = resp.json()
    return [
        {
            "question":    m["question"],
            "probability": float(m["outcomePrices"]),
            "volume":      m["volume"],
            "end_date":    m["endDateIso"],
        }
        for m in data
    ]
```

**Markets to monitor for oil strategy:**
- "Iran nuclear deal" → inverse oil signal
- "OPEC production cut" → direct oil signal
- "US recession" → inverse energy signal

### 3.6 Kalshi — US-Regulated Prediction Market

Same concept as Polymarket but CFTC-regulated. Legally
cleaner for US residents. Requires free account for read
access.

```python
async def get_kalshi_markets(keyword: str) -> list[dict]:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://trading-api.kalshi.com/trade-api/v2/markets",
            params={"status": "open", "search": keyword},
            headers={"Accept": "application/json"}
        )
    return resp.json().get("markets", [])
```

**Why Kalshi over Polymarket:** More regulated, US-focused
markets (Fed rate decisions, inflation, elections), often
with deeper liquidity on macro topics directly relevant to
equity strategies.

---

## 4. Kalman Filter Pairs Strategy

### The Idea

Instead of using static OLS to compute hedge ratios in
StatArbStrategy (which fixes β for the entire 60-day
window), maintain a Kalman-estimated β that updates every
single bar. The signal logic (z-score ±2.0) stays exactly
the same — only the β computation changes from static to
dynamic.

### Why It Works Better

Static OLS assumes β is constant for 60 days. If AAPL-MSFT
had β=1.2 in January and a sector rotation changes it to
β=0.9 in March, you are trading the wrong hedge ratio for
up to 60 days. Every trade during that period is slightly
wrong — generating consistent small losses that compound.

Kalman adapts β in ~3-5 bars after the relationship changes.
OLS with a 60-day window takes up to 60 bars.

**Published result:** Switching from rolling OLS to Kalman
filter on equity pairs reduced false signals by ~32% and
improved annualized Sharpe ratio from 0.8
