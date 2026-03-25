# SIGNAL_DISCOVERY_ALTERNATIVE_DATA.md

## What This Document Is

A complete research-based guide to discovering trading
opportunities before the market prices them in. Covers
four layers: alternative data, news and event-based signals,
quantitative screeners, and systematic pipeline architecture.
Every signal is backed by academic evidence with effect sizes.

---

## PART 1: WHY TRADITIONAL DATA IS NOT ENOUGH

Standard financial data (price, volume, fundamentals) is
available to every participant simultaneously. The moment
a 10-K is filed, thousands of analysts process it. The
information half-life of public filings is measured in
seconds for large-cap stocks.

The only durable edge comes from:
1. Processing public information FASTER (HFT territory)
2. Processing public information MORE ACCURATELY (NLP, GNN)
3. Having access to data OTHER participants do not process (alternative data)
4. Asking different QUESTIONS about the same data (novel signal construction)

This document focuses on 2, 3, and 4.

---

## PART 2: FORM 4 — INSIDER TRADING SIGNALS

### 2.1 What Form 4 Is

Under Section 16 of the Securities Exchange Act (1934),
corporate insiders (officers, directors, >10% shareholders)
must report any transaction in their company's securities
within 2 business days. Form 4 filings are public on SEC EDGAR:
https://www.sec.gov/cgi-bin/browse-edgar

Each filing contains:
- Insider name and title
- Transaction date
- Number of shares bought/sold
- Price per share
- Transaction type (open market vs. pre-planned 10b5-1 plan)

### 2.2 What the Research Shows

**Insider BUYS are highly informative. Insider SELLS are not.**

- Insider open-market purchases predict positive abnormal
  returns of 4-7% over the next 6-12 months
- Insider sales contain near-zero predictive information
  because insiders sell for many reasons (diversification,
  tax, divorce) unrelated to firm prospects

**Critical distinction — Open Market vs. 10b5-1 Plan:**
- Open market purchase: insider decided today to buy.
  Strong signal — no legal way to plan this in advance.
- 10b5-1 plan sale: automatic schedule set up months ago.
  Weak signal — predates any current knowledge.

**Alpha decay timeline (Oenschlaeger & Mollenhoff, 2025):**
For LARGE-CAP stocks (S&P 500): alpha is positive for
1-5 days after filing then vanishes — too fast to trade
profitably after costs.

For SMALL-CAP stocks (Russell 2000): alpha persists for
20-60 days after filing. Monthly CAPM alpha of 39 bps
(annualized 4.7%) and Carhart 4-factor alpha of 37 bps
(annualized 4.5%) are statistically significant at 1%
level even after transaction costs.

**Why small-cap only works:**
In large caps, thousands of quant funds scrape EDGAR in
real time. The signal is arb'd away within hours. In small
caps, fewer eyes watch each filing — information propagates
slowly.

**The "Not-Sold" signal (Cziraki et al., 2021):**
When a portfolio insider sells some holdings but
deliberately keeps others, the kept stocks earn abnormal
returns. The act of NOT selling is itself a signal.
Strategy: when an insider files a Form 4 sale, identify
which holdings they KEPT. Those are high-conviction holds.
Annualized alpha 4.8% after costs, 12-month hold.

### 2.3 Filter Rules — Only Trade Strong Signals

Apply ALL filters before acting:

| Filter            | Threshold              | Reason                        |
|-------------------|------------------------|-------------------------------|
| Market cap        | < $5B                  | Alpha only in small/mid cap   |
| Transaction type  | Open market only       | Excludes 10b5-1 plans         |
| Insider title     | C-suite or Director    | Excludes low-info insiders    |
| Dollar value      | > $100,000             | Meaningful personal conviction|
| Cluster buying    | >= 2 insiders / 30 days| Stronger consensus signal     |
| Not in blackout   | 30d post-earnings      | Insiders cannot trade legally |

**Cluster buying** — multiple insiders buying the same week —
is the strongest form of the signal. One insider may have
personal reasons. Three insiders buying simultaneously is
near-impossible to explain without genuine conviction.

### 2.4 Implementation

```python
def is_strong_buy_signal(filing: dict) -> bool:
    """
    Returns True if filing meets strong-buy criteria:
    1. Transaction type = open market purchase (code 'P')
    2. NOT a 10b5-1 plan
    3. Insider is CEO, CFO, or board director
    4. Value > $100,000
    """
    return (
        filing['transaction_code'] == 'P' and
        not filing['is_10b5_plan'] and
        filing['title'] in ['CEO', 'CFO', 'Director', 'President'] and
        filing['shares'] * filing['price'] > 100_000
    )
```

**Research:**
Oenschlaeger, E. & Mollenhoff, S. (2025). "Insider filings
as trading signals — Does it pay to be fast?"
Finance Research Letters 72.
https://doi.org/10.1016/j.frl.2024.015435

---

## PART 3: SHORT INTEREST — SQUEEZE PREDICTION

### 3.1 What Short Interest Measures

Short interest = shares sold short / total float.

A stock with 25% short interest means 1 in 4 shares in
circulation has been borrowed and sold. All short sellers
must eventually buy back (cover). If the stock rises, they
face margin calls that FORCE covering — creating a feedback
loop called a short squeeze.

### 3.2 Directional Signal

High short interest negatively predicts returns on aggregate:
short sellers are predominantly institutional and better
informed. High aggregate short interest predicts negative
future returns in 24 of 32 countries studied (Gorbenko, 2023).

BUT: for individual stocks with high short interest +
catalyst, expected return can FLIP POSITIVE via squeeze.

### 3.3 Short Squeeze Prediction Model

Mendelu Working Paper (2025) studied squeeze determinants
using rare-event logistic regression:

Key findings:
- 1 percentage point increase in short interest combined
  with elevated investor attention => 5 percentage point
  increase in P(squeeze)
- High institutional ownership is STABILIZING (easier
  to borrow shares => harder to squeeze)
- Short interest CHANGE (7-17% increase) is more
  predictive than level alone. A stock rising from 10%
  to 25% SI is more squeeze-prone than one stable at 25%.
- Crowding: when many short sellers hold the same position,
  simultaneous covering cascade risk increases super-linearly.

### 3.4 Squeeze Score Construction

```python
def compute_squeeze_score(ticker_data: dict) -> float:
    """
    Returns squeeze probability score 0-1.
    Higher = more likely to squeeze.
    """
    si            = ticker_data['short_interest_pct']   # e.g. 0.25
    si_change     = ticker_data['si_30d_change']        # e.g. 0.10
    days_to_cover = ticker_data['days_to_cover']        # SI / avg_vol
    inst_own      = ticker_data['inst_ownership']       # e.g. 0.70
    atv           = ticker_data['volume_ratio']         # today/avg

    score = 0.0
    if si > 0.20:          score += 0.25
    if si_change > 0.07:   score += 0.20
    if days_to_cover > 5:  score += 0.20
    if inst_own < 0.50:    score += 0.15
    if atv > 2.0:          score += 0.20
    return min(score, 1.0)

# Entry: score > 0.70 AND catalyst present (earnings, news, FDA)
# Exit:  trailing stop 15% OR short interest drops below 10%
```

**Days-to-cover (DTC) = short_interest_shares / avg_daily_volume**

DTC > 5: if all short sellers cover simultaneously, it
takes 5 full trading days of volume. The covering rally
can last 5+ days — enough time for a momentum strategy
to profit.

**Data sources:**
- FINRA Short Sale Data: bi-monthly, free.
  https://www.finra.org/investors/learn-to-invest/advanced-investing/short-selling
- IEX Cloud: daily short volume, ~$9/month
- Quandl/Nasdaq: historical short interest since 2010

**Research:**
Gorbenko, A. (2023). "Short Interest and Aggregate Stock Returns."
Review of Asset Pricing Studies 13(4), 691-729.
https://academic.oup.com/raps/article/13/4/691/7127046

Mendelu WP 104/2025. "The Effects of Short Interest on the
Likelihood of Short Squeeze."
http://ftp.mendelu.cz/RePec/men/wpaper/104_2025.pdf

---

## PART 4: VOLUME ANOMALY — ABNORMAL TRADING SIGNAL

### 4.1 Definition

Abnormal Trading Volume (ATV):

    ATV_t^i = V_t^i / (1/21 * sum_{k=1}^{21} V_{t-k}^i)

ATV > 3.0 means today's volume is 3x the 21-day average.

### 4.2 Academic Foundation

Research findings (SMU 2008, EFMA 2016):
- Stocks with top-decile ATV over a week outperform
  over the following 1-5 weeks
- Return forecasting power of ATV is strongly positive
  up to five weeks ahead
- Effect is driven by INVESTOR ATTENTION, not information:
  the volume spike gets picked up by screeners, Reddit,
  Twitter — creating a buying pressure cascade

**Winners with high ATV:** Continue up for 1-5 weeks,
then reverse at 3-6 month horizon.
**Losers with high ATV:** Recover faster than losers
without ATV — attention accelerates price discovery.

**Trading implication:**
- ATV spike = ENTRY signal (buy the breakout)
- ATV returning to normal = EXIT signal (attention gone)
- Do NOT hold past 6 weeks based on ATV alone

### 4.3 Persistent ATV (PATV) — Stronger Signal

Rolling z-score of volume:

    PATV_t^i = (1/5) * sum_{k=0}^{4} z-score(V_{t-k}^i)

Portfolios with high PATV (sustained abnormal volume
multiple days in a row) drift in the direction of recent
returns short-term. Single-day spikes are noise.
Multi-day sustained ATV is the real signal.

### 4.4 Volume-Momentum Composite

The strongest combination:

    CompositeSignal_t^i = TSMOM_t^i * ATV_t^i

Research result: High-ATV winners (top quintile ATV +
top quintile momentum) outperform low-ATV winners by
3.2% over the following month. Momentum gives direction,
ATV confirms the move will spread via attention cascade.

```python
def composite_signal(returns_252d: float,
                     atv: float,
                     threshold_mom: float = 0.0,
                     threshold_atv: float = 2.0) -> int:
    """Returns: +1 (long), -1 (short), 0 (no trade)"""
    if returns_252d > threshold_mom and atv > threshold_atv:
        return +1
    if returns_252d < threshold_mom and atv > threshold_atv:
        return -1
    return 0
```

**Research:**
Lee, C. & Swaminathan, B. (2000). "Price Momentum and
Trading Volume." Journal of Finance 55(5), 2017-2069.

EFMA (2016). "Abnormal Trading Volume and the Cross-Section
of Stock Returns."

---

## PART 5: PEAD SETUP — POST-EARNINGS ANNOUNCEMENT DRIFT

### 5.1 The Anomaly

One of the most replicated anomalies in finance.

When a company reports earnings beating consensus, the
stock continues drifting upward for 30-60 days.
Drift is proportional to surprise size.

**Standardized Unexpected Earnings (SUE):**

    SUE_t^i = (EPS_actual - EPS_consensus) / sigma(forecast_error)

where sigma = historical std of forecast errors for that stock.

**Effect size (Bernard & Thomas, 1989):**
- Top SUE quintile vs. bottom: 6% excess return over 60 days
- Effect is monotone: Q5 > Q4 > Q3 > Q2 > Q1
- Survives Fama-French 3 and 5 factor risk adjustment
- Persists today (weaker than 1989, but still significant)

### 5.2 The Strongest PEAD Setup

    SUE > +15% AND price reaction on earnings day < +3%

Market acknowledged the beat but did not fully absorb it.
Expected drift: 3-6% over next 30 days.

Why the muted reaction occurs:
1. Report released after-hours — full analysis takes days
2. Management gave cautious guidance despite the beat
3. Beat was in a non-core metric — market uncertain
   if it is sustainable

### 5.3 Implementation

```python
import numpy as np

def pead_entry_signal(eps_actual: float,
                      eps_consensus: float,
                      price_change_day0: float,
                      historical_forecast_errors: list) -> bool:
    sigma = np.std(historical_forecast_errors)
    if sigma == 0:
        return False
    sue = (eps_actual - eps_consensus) / sigma
    # Positive surprise, muted price reaction
    return sue > 1.5 and 0.0 < price_change_day0 < 0.03

# Entry:  day after earnings (avoid overnight gap risk)
# Size:   1/2 normal (elevated uncertainty post-earnings)
# Exit:   30 days OR trailing stop 8% OR next earnings
```

**Research:**
Ball, R. & Brown, P. (1968). "An Empirical Evaluation of
Accounting Income Numbers." Journal of Accounting Research.

Bernard, V. & Thomas, J. (1989). "Post-Earnings-Announcement
Drift: Delayed Price Response or Risk Premium?"
Journal of Accounting and Economics 13, 305-340.

---

## PART 6: ALTERNATIVE DATA — WHAT HEDGE FUNDS USE

### 6.1 Satellite and Geospatial Data

Providers: Orbital Insight, Descartes Labs, SpaceKnow.

What they provide:
- Parking lot fullness at retailers => same-store sales proxy
- Oil storage tank shadow measurement => commodity inventory
- Construction activity => housing/industrial output
- Shipping container traffic => trade flow prediction

Evidence: Satellite parking lot data predicts same-store
sales 15% more accurately than analyst consensus.
Information arrives 2-3 weeks before the earnings report.

Cost: $50,000-$500,000/year (institutional).
Individual trader proxy: Google Maps street-level
busyness data (free, but not systematic).

### 6.2 Credit Card Transaction Data

Providers: Second Measure (Bloomberg), Earnest Research,
YipitData.

What it measures: Anonymized aggregated spending at
specific merchants. If McDonald's credit card transactions
are up 12% YoY in Q3 => Q3 revenue will beat consensus.

Edge: Largest for consumer-facing companies (retail,
restaurants, travel) in mid-to-small cap where analyst
coverage is sparse.

Free approximation:
- Google Trends for brand search volume (free API)
- App download trends: Sensor Tower (free tier)
- Yelp/TripAdvisor review volume as foot traffic proxy

### 6.3 Google Trends as Free Alternative Data

Academic evidence (Da, Engelberg & Gao, 2011):
Search volume for a stock ticker predicts price changes
at weekly horizon. Sudden search volume spikes => excess
returns of 3.5% over the following 2 weeks as retail
attention drives buying pressure => then partial reversal.

    SVI_t = Search Volume Index (Google Trends, 0-100)
    Abnormal_SVI_t = SVI_t - median(SVI_{t-8weeks:t-1})

    if Abnormal_SVI_t > 2 * std(SVI): attention spike

```python
from pytrends.request import TrendReq

def get_google_attention_signal(ticker: str) -> float:
    """
    Returns normalized attention signal.
    > 2.0 = significant retail attention spike.
    """
    pytrends = TrendReq(hl='en-US', tz=360)
    pytrends.build_payload([f"{ticker} stock"], timeframe='today 3-m')
    df = pytrends.interest_over_time()
    series = df[f"{ticker} stock"]
    current         = series.iloc[-1]
    historical_mean = series.iloc[:-4].mean()
    historical_std  = series.iloc[:-4].std()
    return (current - historical_mean) / (historical_std + 1e-8)
```

**Research:**
Da, Z., Engelberg, J. & Gao, P. (2011). "In Search of
Attention." Journal of Finance 66(5), 1461-1499.

### 6.4 SEC EDGAR — Free Institutional-Grade Data

**8-K filings (material events):**
Filed within 4 days of any material event: leadership
change, major contract, M&A announcement, regulatory
action. An 8-K at 8:30 AM before market open is a
tradeable event.

**13F filings (institutional holdings):**
Every fund >$100M AUM files quarterly. See exactly what
Berkshire, Bridgewater, Renaissance hold and what changed.
Lag: 45 days after quarter-end. Still useful for
understanding smart money positioning.

**13D/G (large position disclosure):**
Any investor accumulating >5% of a company's float must
file 13D (activist) or 13G (passive) within 10 days.
An activist 13D filing typically causes +10-15% immediate
move and is often a multi-month catalyst.

```python
import requests
import pandas as pd

def monitor_edgar_8k(ticker: str, days: int = 30) -> list:
    """Returns list of 8-K filings in the last N days."""
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    url = (f"https://efts.sec.gov/LATEST/search-index?"
           f"q=%22{ticker}%22&forms=8-K&dateRange=custom&startdt={start}")
    response = requests.get(url, timeout=10).json()
    return response.get('hits', {}).get('hits', [])
```

---

## PART 7: SUPPLY CHAIN COINTEGRATION DISCOVERY

### 7.1 Finding Pairs from Fundamentals First

Don't mine for pairs statistically first.
Start with economic relationships, then confirm statistically.

**Method 1 — SEC 10-K Revenue Concentration:**
Companies must disclose when a single customer accounts
for >10% of revenue. EDGAR full-text search:

    "accounted for approximately" "% of our net revenue"

This gives documented supplier-customer pairs where the
economic link is confirmed in legal filings.

**Method 2 — Sector Proximity Screening:**
Same GICS sub-industry (6-digit code)
+ rolling correlation > 0.6
+ market cap ratio between 0.1 and 10 (similar sensitivity)
=> candidate for cointegration test

**Method 3 — Competitors from 10-K "Competition" Section:**
Every 10-K includes a "Competition" section naming direct
competitors. Companies that explicitly name each other as
competitors share market, customers, and pricing power =>
strong cointegration candidates.

---

## PART 8: MORNING SCREENING PIPELINE

Run this sequence every morning before market open:

```
07:00  1. Pull overnight 8-K filings from EDGAR
           → any material events in our universe?

07:15  2. Pull Form 4 filings from last 2 days
           → any cluster buys in small/mid caps?

07:30  3. ATV screener on full universe
           → any stocks with volume_ratio > 2.5 sustained?

07:45  4. Short squeeze scores for this week's earnings stocks
           → any score > 0.65 with catalyst?

08:00  5. Google Trends scan for attention spikes
           → any abnormal SVI in our watchlist?

08:15  6. Check PEAD setups from last 5 days of earnings
           → any SUE > 1.5 with muted price reaction?

08:30  [Market opens] → Execute with limit orders only
```

---

## PART 9: SIGNAL COMBINATION — CONVICTION STACK

The strongest trades occur when multiple independent
signals converge on the same stock simultaneously.

| # Signals Aligned | Approx Win Rate | Position Size |
|---|---|---|
| 1 | ~53% | 0.5x normal |
| 2 | ~58% | 1.0x normal |
| 3 | ~64% | 1.5x normal |
| 4 | ~68% | 2.0x normal |

**Example — maximum conviction setup:**

```
STOCK: Small-cap semiconductor supplier

Signal 1 (Form 4):    CEO bought $600k open market, 3 days ago
Signal 2 (ATV):       Volume = 3.8x normal for 3 consecutive days
Signal 3 (Momentum):  12-month return = +24% (top quartile)
Signal 4 (Squeeze):   Short interest 19%, DTC = 6.5 days

=> All four signals align
=> Position: 2x normal size
=> Stop: 3x ATR(14) trailing
=> Target hold: 20-40 days
```

**Mathematical justification:**
If each signal has win rate p=0.54 and signals capture
independent information sources (price, filings,
fundamentals, attention), the combined win rate is
empirically 0.64-0.68 — documented in factor combination
studies. The key: combine signals from DIFFERENT DATA
SOURCES for maximum independence benefit.

**Research:**
Da, Z., Engelberg, J. & Gao, P. (2011). ibid.
Gorbenko, A. (2023). ibid.
Mendelu WP 104/2025. ibid.
