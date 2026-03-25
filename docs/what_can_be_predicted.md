# 09_WHAT_CAN_BE_PREDICTED.md

## What This Document Is

A research-based investigation of what is and is not
predictable in financial markets — equities and crypto.
This is the foundational question behind every strategy
in this repo. Every strategy implicitly assumes some
quantity is predictable. This document tests that assumption
rigorously against a century of empirical evidence.

---

## PART 1: THE EFFICIENT MARKET HYPOTHESIS — WHAT IT ACTUALLY SAYS

### 1.1 The Three Forms

Fama (1970) defined market efficiency in three strengths:

**Weak Form:** Past prices contain no information about
future prices. Technical analysis has zero edge.
Evidence: Mostly true for raw returns. HOWEVER: momentum
(a price-based signal) survives after 50+ years of
documented evidence — a direct contradiction.

**Semi-Strong Form:** All publicly available information
is instantly priced in. Fundamental analysis has no edge.
Evidence: Partially true. Earnings surprises take 60 days
to be fully absorbed (PEAD — Post-Earnings Announcement
Drift). Analyst recommendation changes move prices over
multiple days. The semi-strong form is violated.

**Strong Form:** Even private information is priced in.
Evidence: False. Insider trading is profitable by definition
(hence illegal). Corporate insiders beat the market by 6–7%
annually before SEC action.

**Conclusion:** Markets are neither perfectly efficient nor
easily exploitable. They are efficient ENOUGH that naive
strategies fail, but inefficient enough that carefully
constructed strategies based on documented anomalies work.

### 1.2 The Adaptive Markets Hypothesis (Lo, 2004)

More accurate than EMH: market efficiency is not a fixed
state but evolves with market participants.

When a new anomaly is discovered and published:
- Phase 1: Only a few traders exploit it → high alpha
- Phase 2: More traders learn of it → alpha decays
- Phase 3: Too many traders → strategy breaks even or loses
- Phase 4: Some traders leave → strategy revives partially

Evidence (Kim, Shamsuddin & Lim, 2011):
US stock return predictability from 1900–2009 is
TIME-VARYING. It was HIGH during 1900–1930 (pre-SEC,
low competition), COLLAPSED post-1980 (institutional
quant funds), and has PARTIALLY RECOVERED post-2000
as market structure complexity increased.

This means: strategies documented in the 1990s have
weaker edge today, but NEW sources of data (alternative
data, microstructure) create new edges.

**Research:**
Lo, A.W. (2004). "The Adaptive Markets Hypothesis."
Journal of Portfolio Management 30(5), 15–29.

Kim, J.H., Shamsuddin, A. & Lim, K.P. (2011). "Stock
return predictability and the adaptive markets hypothesis."
Journal of Empirical Finance 18(5), 868–879.
https://doi.org/10.1016/j.jempfin.2011.08.002

---

## PART 2: THE HIERARCHY OF WHAT IS PREDICTABLE

From most to least predictable, with evidence:

### 2.1 Volatility — MOST PREDICTABLE (Sharpe 0.8–1.5)

**What it is:** Tomorrow's volatility can be predicted from
today's volatility with high accuracy. This is the most
robust finding in all of empirical finance.

**Mathematical model — GARCH(1,1):**

    r_t = μ + ε_t,          ε_t = σ_t · z_t,    z_t ~ N(0,1)
    σ_t² = ω + α · ε_{t-1}² + β · σ_{t-1}²

The parameters (ω, α, β) are estimated by MLE.
Typically: α ≈ 0.09, β ≈ 0.90, α+β ≈ 0.99.

α + β close to 1 means: volatility is highly persistent.
Today's high volatility predicts tomorrow's high volatility.

**Why it works economically:**
Volatility clusters because the events that cause it
(earnings, rate decisions, geopolitical shocks) themselves
cluster. A Fed decision creates uncertainty that takes
days to resolve — the GARCH persistence parameter β
captures this memory.

**Practical use:**
Position sizing: when GARCH predicts high σ tomorrow,
reduce position size. When GARCH predicts low σ, increase.
This alone improves Sharpe by 15–25% on any directional
strategy.

**Research:**
Engle, R.F. (1982). "Autoregressive Conditional
Heteroskedasticity." Econometrica 50(4), 987–1007.
Bollerslev, T. (1986). "Generalized Autoregressive
Conditional Heteroskedasticity." Journal of Econometrics.

---

### 2.2 Spread Mean Reversion — HIGHLY PREDICTABLE for cointegrated pairs

**What it is:** For two economically linked stocks, the
spread between them is stationary — it is pulled back
toward its mean by a force proportional to its deviation.

**Mathematical model — OU process:**
    dS_t = κ(θ - S_t)dt + σ dW_t

This is fully treated in 07_PAIR_HEALTH_CHECK_AND_OU_MODEL.md.

**Why it is predictable:** The economic link (shared
customer, shared input, regulatory equivalence) creates
a fundamental constraint on how different two stocks can
become. The constraint enforces mean reversion.

**Predictability horizon:** Days to weeks (κ ≈ 0.02–0.05/day
for good pairs → half-life 14–35 days).

**Key caveat:** Predictable on average across many pairs
and many time periods. Any single trade can lose. The edge
is in the law of large numbers across trades.

---

### 2.3 Momentum — PREDICTABLE at 1–12 month horizon

**What it is:** Stocks that have outperformed over the
past 12 months (excluding the last month) tend to continue
outperforming over the next 1–3 months.

**Effect size (Jegadeesh & Titman, 1993):**
Top decile - bottom decile = 9.5% per year on US stocks
from 1965–1989. Replicated in 40+ countries and 4 asset
classes (equity, fixed income, commodity, FX).

**Why it is predictable:** Under-reaction to information
(investors update beliefs slowly). Institutional rebalancing.
Detailed in 08_MOMENTUM_TRAILING_STOP.md.

**Predictability horizon:** 1–12 months.
At < 1 month: reversal (microstructure)
At 1–12 months: momentum
At > 12 months: reversal again (value effect)

---

### 2.4 Earnings Surprises (PEAD) — PREDICTABLE for 60 days

**What it is:** When a company reports earnings that
beat analyst consensus, the stock continues to drift
upward for ~60 days after the announcement. This is
Post-Earnings Announcement Drift (PEAD).

**Size:** Top SUE (Standardized Unexpected Earnings)
quartile outperforms bottom quartile by ~6% in the
60 days following earnings announcement.

**Why predictable:** Analysts and investors update slowly.
Institutional investors have mandate constraints that
prevent immediate full repositioning. The market slowly
"digests" the earnings information over weeks.

**Predictability horizon:** 20–60 days post-announcement.
After 60 days: most of the drift has been captured.

**Practical use:** After earnings, enter in direction
of surprise, hold 20–40 days, exit before next earnings.
This is a distinct strategy from both momentum and stat arb.

**Research:**
Ball, R. & Brown, P. (1968). "An Empirical Evaluation of
Accounting Income Numbers." Journal of Accounting Research.
Bernard, V. & Thomas, J. (1989). "Post-Earnings-Announcement
Drift." Journal of Accounting and Economics 13, 305–340.

---

### 2.5 Cross-Sectional Returns — WEAKLY PREDICTABLE

**What it is:** Predicting whether stock A will go up OR
down tomorrow, without knowing where the whole market goes.

**Best documented predictor:** Implied volatility minus
realized volatility (IV - RV). When IV > RV (market
pricing in more fear than history justifies), stocks
with high IV-RV spread tend to underperform.
The predictive R² is 2–4% — statistically significant
but economically modest. [web:143]

**Practical use:** Alone, too weak to build a strategy.
Used as a signal overlay to existing strategies (reduce
position size when IV-RV is extreme).

---

### 2.6 Raw Price Direction (Tomorrow is Up or Down) — ESSENTIALLY UNPREDICTABLE

**The evidence:**
Best-in-class ML models (Transformer, GNN-LSTM hybrids)
achieve 52–58% directional accuracy on daily stock returns
out of sample. [web:79]

On S&P 500 index (most studied market):
ARIMA, GARCH, LSTM, Transformer all achieve near 50%
directional accuracy for next-day returns.

**Why unpredictable:**
The daily return of a well-followed stock reflects the
arrival of genuinely new, random information. By definition,
genuinely new information is not predictable from past data.
What IS predictable is the response to known recurring
events (earnings, Fed decisions) — but that is a different
signal (PEAD, macro timing) not raw direction.

**The subtle point:**
A model can have 52% accuracy and be highly profitable if
the 52% correct calls are systematically larger than the
48% wrong calls. This is what momentum achieves — not
better direction prediction, but better sizing of the
correct positions. Adaptation, not prediction.

---

## PART 3: CRYPTO — A DIFFERENT PREDICTABILITY STRUCTURE

### 3.1 Why Crypto Is More Predictable Than Equities

Three structural reasons:

**Reason 1 — Less institutional participation (historically):**
Until 2023, crypto markets were dominated by retail traders
with behavioral biases. A market with more naive participants
is more predictable. As institutions (BlackRock BTC ETF,
2024) enter, predictability decays — the same lifecycle
as equities.

**Reason 2 — 24/7 continuous trading:**
No overnight gaps. No earnings surprises. Price discovery
is continuous. This creates cleaner intraday patterns
because there is no information accumulation during
market closure.

**Reason 3 — Higher volatility = larger signal-to-noise:**
BTC σ_daily ≈ 3–5% vs SPY σ_daily ≈ 0.8%.
A momentum signal that produces 0.5% expected edge per
trade is meaningful relative to 5% noise in BTC.
Against 0.8% noise in SPY, the same signal may be invisible.

### 3.2 What Works in Crypto — Evidence

**Short-term mean reversion (hourly):**
Bitcoin exhibits mean reversion on hourly data.
Bollinger Band strategy outperformed buy-and-hold and
momentum on BTC using hourly bars, Sharpe 1.18 vs 0.74
for buy-and-hold. [web:141]

**Mechanism:** Retail overreaction to news causes sharp
spikes that revert within 2–8 hours. This is the crypto
equivalent of the intraday reversal in equity markets.

**Short-term momentum (daily):**
At daily resolution, BTC exhibits momentum.
Shorter lookback periods (10–20 days) work better than
longer periods (50 days). Local maxima in BTC tend to
be followed by continued upward moves (breakout behavior).
Local minima tend to bounce. [web:147]

**This is NOT contradictory:**
Hourly = mean reversion (overreaction corrects within hours)
Daily = momentum (trend persists over days to weeks)
Monthly = mean reversion again (long-run value reversion)

The regime changes with the timescale — same as equities.

### 3.3 Intraday Predictability in Crypto

Wen et al. (2022) studied crypto intraday patterns
from 2013–2020:

**Finding:** Intraday momentum (first hour of trading
predicts the rest of the day) exists and is profitable.
A timing strategy using intraday momentum outperformed
buy-and-hold by 6.8% annually on BTC.

**Mechanism:** "Late-informed investors" — retail traders
who see early price action and chase it, creating momentum
that lasts several hours before reversing.

**Intraday reversal** also exists: extreme moves within
a single hour tend to partially reverse in the next hour
(overreaction and overconfidence correction). [web:144]

**Practical implementation for crypto:**
- Intraday mean reversion: Bollinger Bands (20-period, 2σ)
  on 15-minute bars. Short when above band, long when below.
  Exit at midline. Suitable for BTC, ETH, SOL.
- Daily momentum: 10-day lookback. Long when 10d return > 0,
  short when negative. Volatility-scale position by ATR(14).

### 3.4 Crypto-Specific Risks Not in Equities

**1. Exchange liquidity risk:** A strategy profitable on
paper can fail if the order book is thin. Slippage of
0.1% on BTC = wipes the edge of many mean reversion trades.
Always account for bid-ask spread in backtests.

**2. Regulatory shock:** A single government ban (China 2021)
caused -50% in days. No technical signal predicts this.
Size positions to survive a -60% overnight move.

**3. Correlation instability:** BTC-altcoin correlations
are near 1.0 during crashes and 0.4–0.7 in calm periods.
A stat arb pair (BTC-ETH) that looks cointegrated in calm
periods may exhibit simultaneous freefall in a crash.
Test cointegration only on calm-period data and flag pairs
for suspension when BTC σ > 5% daily.

**4. Weekend effects:** BTC trades 24/7 but institutional
participants exit on weekends. Friday close to Monday open
has distinct microstructure — spreads wider, momentum
signals less reliable. Consider reducing position sizes
held over weekends.

---

## PART 4: COMPARISON TABLE — ALL STRATEGIES BY PREDICTABILITY

| What Is Predicted | Horizon | Approx Sharpe | Works in Crypto | Method |
|---|---|---|---|---|
| Volatility level | 1–5 days | 0.8–1.5 | Yes (stronger) | GARCH, realized vol |
| Spread mean reversion | 5–40 days | 0.5–1.2 | Yes (BTC/ETH pairs) | OU, cointegration |
| Momentum continuation | 20–120 days | 0.6–1.3 | Yes (10–30d best) | TSMOM, UMD |
| Earnings drift (PEAD) | 20–60 days | 0.4–0.8 | No (no earnings) | SUE signal |
| Intraday reversion | 15min–4hr | 0.8–1.4 | Yes (stronger) | Bollinger, OFI |
| Raw price direction | 1 day | ~0.0 | ~0.0 | Nothing works reliably |
| Sector rotation | 1–3 months | 0.4–0.7 | Partial (BTC leads alts) | GNN, HMM regime |

**When Each Strategy Wins:**
- Stat arb / OU: Low volatility, stable sector, ADX < 20
- Momentum: Trending market (ADX > 25), VIX < 30
- PEAD: Post-earnings, any market condition
- Intraday mean reversion: High volume sessions, liquid assets
- Volatility trading: Always (no directional bet needed)

**When Each Fails:**
- Stat arb: Trend regimes, regulatory breaks, M&A
- Momentum: Sharp reversals (2009, March 2020), VIX > 40
- PEAD: When earnings information was already leaked
- Intraday: Thin liquidity, major news day
- GARCH: Structural breaks (model mis-specified post-shock)
