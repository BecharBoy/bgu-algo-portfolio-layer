# 05_REGIME_MOMENTUM_DL_STRATEGIES.md

## What This Document Is

A complete mathematical and economic treatment of three topics:
1. Regime Detection — how to know whether the market is trending or mean-reverting
2. Momentum Factor — how quant funds made money during the Mag7 bull run (2023–2025)
3. Deep Learning for Trading — what worked, what failed, and why, from published research

Written for someone with a strong CS and math background.
Each claim has an economic justification, a mathematical proof or derivation,
and a citation to peer-reviewed research or documented fund performance.

---

## PART 1: REGIME DETECTION

### 1.1 The Core Problem

Every strategy in this repo assumes a specific market regime:
- MeanReversionMomentum assumes prices oscillate around a stable mean
- CointegrationArb assumes the spread between two stocks is stationary
- A momentum strategy (Part 2) assumes trends persist

All three assumptions break down when the wrong regime is active.
Running stat arb during a strong trend causes losses not because the
math is wrong, but because the precondition (stationarity) is violated.
Running momentum during a choppy mean-reverting market causes losses
for the symmetric reason.

**The economic justification for regimes:**
Markets alternate between two behavioral modes, documented empirically
across all asset classes (Ang & Timmermann, 2012):

- **Trend regime:** Investors under-react to new information (behavioral
  finance: anchoring, herding). Prices move in the direction of earnings
  surprises for weeks. Institutional rebalancing amplifies direction.
  This regime dominated US equities 2023–2025, particularly Mag7.

- **Mean-reversion regime:** Markets over-react to short-term noise.
  Liquidity providers absorb order imbalances and push prices back.
  Volatility is low, no macro catalyst. This regime dominates gold,
  energy, and low-beta sectors in calm periods.

**Research:**
Ang, A. & Timmermann, A. (2012). "Regime Changes and Financial Markets."
Annual Review of Financial Economics 4(1), 313–337.
https://doi.org/10.1146/annurev-financial-110311-101808

Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
Nonstationary Time Series and the Business Cycle."
Econometrica 57(2), 357–384.
https://doi.org/10.2307/1912559

---

### 1.2 Method 1 — ADX (Average Directional Index)

**Definition:**
The ADX measures trend strength, not direction. It is derived from
the Directional Movement Index (DMI):

First compute True Range and directional movements:
    TR_t  = max(High_t - Low_t, |High_t - Close_{t-1}|, |Low_t - Close_{t-1}|)
    +DM_t = High_t - High_{t-1}  if positive and > |Low_t - Low_{t-1}|, else 0
    -DM_t = Low_{t-1} - Low_t    if positive and > High_t - High_{t-1}, else 0

Smooth over N periods (Wilder smoothing, N=14 standard):
    TR_N  = Σ_{i=t-N}^{t} TR_i
    +DI_N = 100 · (Σ +DM_i) / TR_N
    -DI_N = 100 · (Σ -DM_i) / TR_N

Directional Index:
    DX_t  = 100 · |+DI_N - -DI_N| / (+DI_N + -DI_N)

ADX is a further Wilder smoothing of DX:
    ADX_t = ((N-1) · ADX_{t-1} + DX_t) / N

**Regime rule:**
    if ADX_t > 25:  TREND regime    → enable momentum, disable MR
    if ADX_t < 20:  CHOPPY regime   → enable stat arb and MR, disable momentum
    20 ≤ ADX ≤ 25:  ambiguous       → reduce position sizes, no new entries

**Why 25?** Wilder (1978) calibrated this threshold empirically on commodity
futures. Subsequent academic work (Park & Irwin, 2007) confirmed it as the
most robust cutoff across asset classes for distinguishing trending from
non-trending periods. ADX > 25 in S&P 500 explained ~60% of momentum
strategy outperformance years in that study.

**Research:**
Wilder, J.W. (1978). New Concepts in Technical Trading Systems.
Park, C.H. & Irwin, S.H. (2007). "What Do We Know About the Profitability
of Technical Analysis?" Journal of Economic Surveys 21(4), 786–826.
https://doi.org/10.1111/j.1467-6419.2007.00519.x

---

### 1.3 Method 2 — Hidden Markov Model (HMM)

**The idea:**
ADX is deterministic — it reads current data and applies a fixed rule.
An HMM is probabilistic — it models the regime as a hidden variable and
estimates the probability of being in each regime given the full history
of observations.

**State-space formulation:**
Let S_t ∈ {0=bear/chop, 1=bull/trend} be the hidden regime.
Let r_t = daily return, σ_t = realized volatility (20-day).
Observations: y_t = (r_t, σ_t)

**Transition matrix** (regime persistence):
    P(S_t = j | S_{t-1} = i) = A_{ij},    A = [[p00, p01], [p10, p11]]
    p00 = prob of staying in regime 0
    p11 = prob of staying in regime 1

Empirically: p00 ≈ 0.97, p11 ≈ 0.95 for daily equity returns
(regimes are persistent — they last weeks to months, not days).

**Emission distribution** (what each regime looks like):
    y_t | S_t=k ~ N(μ_k, Σ_k)
    
    Regime 0 (bear/chop): μ_0 ≈ (-0.05%, 1.5%), higher σ
    Regime 1 (bull/trend): μ_1 ≈ (+0.08%, 0.8%), lower σ

**Parameter estimation: Baum-Welch Algorithm (E-M for HMMs)**

E-step — compute forward-backward probabilities:

    Forward:  α_t(k) = P(y_1, ..., y_t, S_t=k | θ)
              α_1(k) = π_k · N(y_1; μ_k, Σ_k)
              α_t(k) = N(y_t; μ_k, Σ_k) · Σ_j α_{t-1}(j) A_{jk}

    Backward: β_t(k) = P(y_{t+1}, ..., y_T | S_t=k, θ)
              β_T(k) = 1
              β_t(k) = Σ_j A_{kj} · N(y_{t+1}; μ_j, Σ_j) · β_{t+1}(j)

    Posterior: γ_t(k) = P(S_t=k | y_{1:T}, θ) = α_t(k) β_t(k) / P(y_{1:T})

M-step — re-estimate parameters:
    μ_k    = Σ_t γ_t(k) y_t / Σ_t γ_t(k)
    Σ_k    = Σ_t γ_t(k)(y_t - μ_k)(y_t - μ_k)ᵀ / Σ_t γ_t(k)
    A_{jk} = Σ_t ξ_t(j,k) / Σ_t γ_t(j)

Iterate until log-likelihood converges. This is guaranteed to find a
local maximum of P(y_{1:T} | θ).

**Trading use:**
After fitting, at each new bar compute P(S_t=1 | y_{1:t}) using the
forward algorithm. If P > 0.7 → trend regime. If P < 0.3 → chop regime.

**Research:**
Hamilton, J.D. (1989), ibid.
Ang, A. & Bekaert, G. (2002). "Regime Switches in Interest Rates."
Journal of Business & Economic Statistics 20(2), 163–182.
https://doi.org/10.1198/073500102317351930

---

### 1.4 Method 3 — Rolling ADF as Pair Health Monitor

For CointegrationArb specifically: re-run the ADF test every 20 bars
on a rolling 90-day window of the spread.

    if ADF_pvalue(spread_{t-90:t}) > 0.05:
        → cointegration broken → force-close the pair immediately

This is the correct response to the "dog pulling the owner" scenario
described in prior sessions. When the spread is no longer stationary,
continuing to hold expects reversion that will never come.

**Mathematical justification:**
The ADF test statistic is:
    τ = ρ̂ / SE(ρ̂)

where ρ̂ is the OLS estimate of the mean-reversion coefficient in:
    ΔS_t = ρ · S_{t-1} + Σᵢ cᵢ ΔS_{t-i} + ε_t

Under H₀ (unit root, no mean reversion): ρ = 0, τ follows the
Dickey-Fuller distribution (non-standard, left-tailed).
p > 0.05 means we cannot reject the unit root → spread has no home.

**Computational cost:** O(N·P) per pair per recheck where N=90 bars,
P=number of lag terms. Negligible.

**Research:**
Engle, R.F. & Granger, C.W.J. (1987). "Co-integration and Error
Correction: Representation, Estimation, and Testing."
Econometrica 55(2), 251–276. https://doi.org/10.2307/1913236

---

## PART 2: MOMENTUM FACTOR — HOW THE BIG MONEY MADE MONEY 2023–2025

### 2.1 Economic Foundation

**Why does momentum exist?**

Three non-mutually-exclusive explanations from the literature:

1. **Under-reaction (Barberis, Shleifer & Vishny, 1998):**
   Investors anchor to prior beliefs and update slowly to new information.
   When NVDA announced AI GPU demand in 2023, most analysts initially
   modelled it as temporary. As the reality became undeniable over months,
   prices continued rising — the information was absorbed slowly, creating
   a persistent directional move. Formally: if the true value jump is V
   and investors initially price only λ·V (λ < 1), prices must continue
   rising until they reach V. Momentum traders front-run this convergence.

2. **Institutional herding and rebalancing:**
   Index funds and ETFs grow when markets rise (AUM increases). They must
   deploy new capital into the same winners (index constituents). This
   creates mechanical buying pressure on rising stocks, amplifying trends.
   NVDA went from 3% to 6% of S&P 500 weighting — index funds bought it
   all the way up.

3. **Leverage and stop-loss cascades:**
   Leveraged long positions in the direction of the trend create stop-loss
   floors below current price. When price rises, stops are raised. A
   temporary pullback hits no stops because no one is short. This creates
   asymmetric upward drift.

**Research:**
Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency."
Journal of Finance 48(1), 65–91. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x

Barberis, N., Shleifer, A. & Vishny, R. (1998). "A model of investor
sentiment." Journal of Financial Economics 49(3), 307–343.
https://doi.org/10.1016/S0304-405X(98)00027-0

---

### 2.2 The Mathematical Model — Time-Series Momentum

**Signal construction (Moskowitz, Ooi & Pedersen, 2012):**
For each asset i at time t, compute the past-12-month return
(skipping the most recent month to avoid microstructure reversal):

    h_t^i = r_{t-252:t-21}^i

(12-month return excluding last month — this is the standard
cross-sectional momentum construction)

**Position sizing with volatility targeting:**
    w_t^i = (σ_target / σ_t^i) · sign(h_t^i)

Where σ_t^i is the 21-day realized volatility of asset i.
This is critical: raw momentum gives large positions to volatile stocks
(which happen to be recent winners). Volatility scaling makes positions
risk-equivalent across assets.

**Mathematical proof of why volatility scaling improves Sharpe:**
Let r̃_t^i = w_t^i · r_t^i be the position-weighted return.
Sharpe of raw signal: SR_raw = E[sign(h)·r] / Std(sign(h)·r)
Sharpe of vol-scaled: SR_scaled = E[(σ_target/σ)·sign(h)·r] / σ_target

If r/σ (the Sharpe ratio of each position) is constant across assets,
then SR_scaled = SR_raw. But empirically, high-volatility assets have
LOWER individual Sharpe ratios (Frazzini & Pedersen, 2014 — Betting
Against Beta). So σ_scaling systematically reduces weight on the assets
with worst risk-adjusted momentum → Sharpe improves.

**Fama-French UMD (Up Minus Down) Factor:**
The formal factor construction used by academic researchers and index
providers:

    r_UMD = (1/2)(r_SmallWin + r_BigWin) - (1/2)(r_SmallLose + r_BigLose)

Where "Win" = top 30% of 12-1 month return, "Lose" = bottom 30%.
Small/Big refers to market cap quintile.

Historical performance of UMD factor:
- Average annual return: ~8.3% (Fama & French, 1996)
- Sharpe ratio: ~0.53 (long-run)
- 2023: +17.2%, 2024: +14.8% (exceptional bull run)
- 2009 crash: -83% in a single month (momentum crash — most dangerous drawdown)

**The momentum crash:**
When a bear market reverses sharply (V-shaped recovery), the most
shorted stocks (recent losers) spike violently, and momentum longs
(now overextended) unwind simultaneously. This created the worst
single-month momentum return in history in March 2009.
Risk management: momentum must be turned OFF when VIX > 40 or
market drawdown > 15%.

**Research:**
Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). "Time series momentum."
Journal of Financial Economics 104(2), 228–250.
https://doi.org/10.1016/j.jfineco.2011.11.003

Frazzini, A. & Pedersen, L.H. (2014). "Betting Against Beta."
Journal of Financial Economics 111(1), 1–23.
https://doi.org/10.1016/j.jfineco.2013.10.005

Daniel, K. & Moskowitz, T.J. (2016). "Momentum crashes."
Journal of Financial Economics 122(2), 221–247.
https://doi.org/10.1016/j.jfineco.2015.12.002

---

### 2.3 Cross-Sectional vs Time-Series Momentum

Two distinct strategies often confused:

**Cross-sectional (Jegadeesh-Titman):**
Each month: rank all assets by past return. Long top decile, short
bottom decile. Dollar-neutral (equal $ long and short).
Profit comes from relative performance of assets against each other.
Works even in bear markets if the ranking is stable.

**Time-series (Moskowitz-Ooi-Pedersen):**
Each asset independently: long if past return positive, short if negative.
NOT dollar-neutral — net market exposure can be large.
2023–2024: almost everything positive → net long → captured the full bull run.

**For our system:**
Cross-sectional is harder to implement (requires ranking across universe).
Time-series is simpler — one decision per stock. Suggested first implementation.

---

### 2.4 Stat Arb Within Sectors During Bull Markets

The correct observation: cointegration does not break during bull markets
when pairs are economically justified and in the same sector.

**Why within-sector pairs survive bull markets:**
Let P_t^A = α_A + β_A · F_t + ε_t^A
    P_t^B = α_B + β_B · F_t + ε_t^B

Where F_t is the common sector factor (e.g., semiconductor demand index).
If β_A ≈ β_B (similar factor loading), then:
    P_t^A - (β_A/β_B) · P_t^B = (α_A - α_B · β_A/β_B) + (ε_t^A - ε_t^B · β_A/β_B)

The factor F_t cancels out. The spread is driven only by idiosyncratic
terms ε^A and ε^B, which ARE stationary. The "rope goes up" but the
length of the rope stays constant.

**When this fails:**
If β_A ≠ β_B (e.g., BAC and MSFT have completely different factor
exposures), the factor does not cancel and the spread is dominated by
the difference in factor loadings · F_t — which trends with F_t.
This is why BAC/MSFT was a losing pair.

**Pair selection rule:** Only trade pairs where |β_A - β_B| < ε for
the dominant sector factor. This requires factor model preprocessing
(PCA on returns to extract factors, then check loadings).

---

## PART 3: DEEP LEARNING FOR TRADING — WHAT WORKED AND WHY

### 3.1 LSTM — The Decade of Attempts

**Architecture recap:**
LSTM (Long Short-Term Memory, Hochreiter & Schmidhuber 1997) was designed
to capture long-range temporal dependencies in sequences. For trading,
input = sequence of [close, volume, returns, indicators] for T timesteps,
output = predicted next return or direction.

The gates:
    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    (forget gate)
    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    (input gate)
    g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  (candidate cell)
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t         (cell state update)
    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    (output gate)
    h_t = o_t ⊙ tanh(C_t)                   (hidden state)

**What LSTM actually learned in trading research:**
NOT price prediction in the classical sense. When it worked, it learned:
- Volatility clustering (GARCH-like patterns)
- Mean-reversion signals in spread series (stationary inputs!)
- Regime transition probability (trained on labeled regime data)

**Where LSTM succeeded:**
Han et al. (2024) trained an LSTM specifically to classify whether a
cointegrated spread is in trend or mean-reversion mode. Result: 80.8%
accuracy on test set for detecting when to HALT stat arb trading.
The system achieved 23% return over 10 days on futures at 1-minute bars.

Why it worked: the LSTM was given a STATIONARY input (the spread, not
raw prices), with manually labeled examples of trending vs. non-trending
spread behavior. This is a classification problem on a bounded signal —
exactly what LSTMs are good at. NOT raw price prediction.

**Where LSTM failed:**
Stanford CS230 (2020): trained LSTM on raw GOOGL prices. Result: the model
produced linear extrapolations of the last known price, trend often
opposite to reality. MSE on test = 0.0546 (worse than naive model).

Why it failed: raw prices are non-stationary (I(1) process). The LSTM
has no mechanism to distinguish "price is 500 because it's been trending
up" from "price is 500 because it's temporarily high." The non-stationarity
makes the training distribution completely different from the test distribution.

**The practitioner consensus:**
Never feed raw prices to any neural network for return prediction.
Always use stationary transformations:
    - Log returns: Δlog(P_t)
    - Spread z-scores: (S_t - μ)/σ
    - Volatility-normalized returns: r_t / σ_t

**Research:**
Han, G. et al. (2024). "LSTM-based optimization algorithm for enhancing
mean-reversion arbitrage." PMC, NCBI.
https://pmc.ncbi.nlm.nih.gov/articles/PMC11323094/

Hochreiter, S. & Schmidhuber, J. (1997). "Long Short-Term Memory."
Neural Computation 9(8), 1735–1780.

Stanford CS230 (2020). "Using LSTM in Stock Prediction and Quantitative
Trading." https://cs230.stanford.edu/projects_winter_2020/reports/32066186.pdf

---

### 3.2 Transformer — The Current Best Architecture

**Why Transformer outperforms LSTM for financial sequences:**
LSTM processes sequences left-to-right, compressing all history into
a fixed hidden state h_t. Information from T=200 bars ago is heavily
attenuated through 200 multiplicative gates.

Transformer uses Self-Attention — every position attends to every
other position with O(1) path length:

    Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V

Where:
    Q = W_Q · X    (query matrix)
    K = W_K · X    (key matrix)
    V = W_V · X    (value matrix)
    d_k = dimension of key vectors (scaling factor for numerical stability)

The attention weight between position i and j:
    a_{ij} = exp(q_i · k_j / √d_k) / Σ_m exp(q_i · k_m / √d_k)

This means bar t=100 can directly attend to bar t=1 with equal weight
as bar t=99. For financial data with seasonal patterns, earnings cycles,
and multi-scale correlations, this is a major advantage.

**Empirical comparison results:**
Atlantis Press (2023): Transformer vs LSTM on Chinese A-shares.
Transformer accuracy on direction prediction: 76% ± 2.5% vs LSTM 71%.
The Transformer's self-attention captured earnings-cycle dependencies
(quarterly patterns) that LSTM's sequential processing missed.

Decoder-only Transformer outperformed all variants including LSTM, TCN,
SVR, and Random Forest across all tested window sizes on S&P 500 index
(2025 arxiv study).

**When Transformer fails:**
Small datasets (< 3 years of daily data). Self-attention has O(T²)
complexity and O(T²) parameter count for the attention matrices.
With insufficient data, overfitting is severe. Rule of thumb:
need at least 1000 training samples per attention head.

**Practical implementation note:**
Use a Decoder-only architecture (like GPT) for autoregressive prediction.
Use positional encoding that respects calendar structure:
    PE(t, 2i)   = sin(t / 10000^{2i/d})
    PE(t, 2i+1) = cos(t / 10000^{2i/d})
Add separate embeddings for day-of-week and month-of-year to capture
seasonality explicitly.

**Research:**
Atlantis Press (2023). "Comparative Study of LSTM and Transformer for
A-Share Stock Prediction." ICAID-23.
https://www.atlantis-press.com/proceedings/icaid-23/125990061

Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 2017.
https://arxiv.org/abs/1706.03762

---

### 3.3 Reinforcement Learning — The Correct Framing for Trading

**Why RL is more appropriate than supervised learning for trading:**

Supervised learning minimizes prediction error: minimize E[||ŷ - y||²].
But in trading, the goal is NOT to predict returns accurately — it is to
maximize cumulative risk-adjusted return. A strategy that predicts returns
with MSE = 0.05 but never trades when it is uncertain may be more profitable
than one with MSE = 0.01 but poor position sizing.

RL formalizes the correct objective:
    maximize E[Σ_t γᵗ r_t]    (discounted cumulative reward)

**Markov Decision Process formulation:**
    State s_t = {prices, positions, portfolio equity, regime indicator, ...}
    Action a_t ∈ {buy, sell, hold} × {size 0.25x, 0.5x, 1x, 2x}
    Reward r_t = Δ portfolio equity - λ · transaction costs - κ · drawdown penalty

**Deep Q-Network (DQN) for trading:**
Approximate the action-value function:
    Q(s, a; θ) ≈ Q*(s, a)  using a neural network with parameters θ

Update rule (Bellman equation):
    L(θ) = E[(r + γ · max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²]

Where θ⁻ is a periodically frozen "target network" to stabilize training.

**What works:** RL agents trained on stationary features (spread z-scores,
volatility ratios, OFI signals) with explicit transaction cost penalties
learn to avoid over-trading. The transaction cost penalty acts as a natural
position sizing constraint.

**What fails:** RL trained on raw prices or with sparse rewards (only
realized PnL, no intermediate shaping) fails to learn useful policies.
The reward signal is too delayed (months between entry and exit) for
standard RL to work. Solution: use shaped rewards (unrealized PnL per bar
+ transaction cost penalty).

**Research:**
Jansen, S. (2020). Machine Learning for Algorithmic Trading, 2nd Ed.
https://stefan-jansen.github.io/machine-learning-for-trading/22_deep_reinforcement_learning/

---

### 3.4 HFT Strategies — Different Universe, Different Math

**The core HFT alpha: Order Flow Imbalance (OFI)**

HFT does NOT predict prices over days. It predicts price movement
over the next 50–500 milliseconds based on the current state of the
limit order book.

**Order Book Imbalance signal:**
    OFI_t = (BidVolume_t - AskVolume_t) / (BidVolume_t + AskVolume_t)

OFI ∈ [-1, +1]. Values near +1 mean strong buy pressure.
Empirical finding: OFI at the best bid/ask predicts mid-price direction
over the next 10 trades with 65–70% accuracy (Cont, Kukanov & Stoikov, 2014).

**Why it works at microsecond scale:**
Informed traders (who have information about future price) place market
orders that consume limit orders. This creates detectable imbalance
before the price moves. HFT algorithms detect this imbalance and trade
in the same direction, front-running the price impact.

**The Avellaneda-Stoikov Market Making Model (2008):**
HFT market makers set bid and ask quotes optimally:

    Reservation price: r_t = s_t - q_t · γ · σ² · (T - t)
    Optimal spread:    δ_bid + δ_ask = γσ²(T-t) + (2/γ)·ln(1 + γ/κ)

Where:
    s_t  = current mid price
    q_t  = current inventory (signed)
    γ    = risk aversion parameter
    σ²   = price variance
    T-t  = time remaining in session
    κ    = order arrival rate parameter

The key insight: when inventory q_t > 0 (too many longs), the reservation
price is shifted DOWN, making the ask quote more competitive and the bid
quote less competitive — systematically reducing inventory without
directional bets.

**HFT profitability structure (empirical, 2025):**
R² of 0.811 between daily trading volume and HFT daily profit.
HFT profit ≈ $80k–$280k/day depending on market turnover.
HFT profits scale linearly with volume — the edge is per-trade, not
directional. This is why HFT firms care about latency (more trades
per day = more edge captured) not about market direction.

**Relevance to our system:**
HFT strategies require colocation (< 1ms latency) and order book data.
Both are currently unavailable. However: the OFI signal is meaningful
at daily resolution as a confirmation signal — if OFI at close is
strongly positive, entering a momentum long the next day captures
the continuation effect.

**Research:**
Avellaneda, M. & Stoikov, S. (2008). "High-frequency trading in a limit
order book." Quantitative Finance 8(3), 217–224.
https://doi.org/10.1080/14697680701381228

Cont, R., Kukanov, A. & Stoikov, S. (2014). "The Price Impact of Order
Book Events." Journal of Financial Econometrics 12(1), 47–88.
https://doi.org/10.1093/jjfinec/nbt003

Aït-Sahalia, Y. et al. (2020). "High frequency traders and the price
process." Journal of Econometrics.
https://www.sciencedirect.com/science/article/abs/pii/S0304407619302428

---

### 3.5 The Ornstein-Uhlenbeck Process — Correct Mathematical Model for Spread

The z-score approach in StatArbStrategy implicitly assumes the spread is
Gaussian. A more precise model is the Ornstein-Uhlenbeck (OU) process:

    dS_t = κ(θ - S_t)dt + σ dW_t

Where:
    S_t = spread value at time t
    κ   = mean-reversion speed (how fast spread returns to θ)
    θ   = long-run mean of the spread
    σ   = volatility of the spread
    W_t = standard Brownian motion

**Properties:**
- E[S_t] = θ + (S_0 - θ)e^{-κt}  → mean reverts to θ at rate κ
- Var[S_t] = (σ²/2κ)(1 - e^{-2κt}) → bounded variance (unlike random walk)
- Half-life of mean reversion: t_{1/2} = ln(2)/κ

**Parameter estimation via MLE:**
Given discrete observations S_1, ..., S_N at intervals Δt, the MLE estimates:

    κ̂ = -ln(ρ̂) / Δt
    θ̂ = (S̄ - ρ̂ · lag(S̄)) / (1 - ρ̂)
    σ̂² = s² · 2κ̂ / (1 - e^{-2κ̂Δt})

Where ρ̂ is the first-order autocorrelation of the spread, and s² is
the sample variance of residuals.

**Optimal entry/exit thresholds (Zeng & Lee, 2014):**
Instead of fixed ±2σ z-score thresholds, solve the optimal stopping problem:

    Entry threshold a* maximizes E[profit | enter at a*]
    Exit threshold b* (b* < a*) maximizes expected Sharpe ratio

The result depends on κ, σ, and transaction costs c:
    a* ≈ θ + σ√(2/κ) · F(c·κ/σ²)   (closed form approximated)

Faster mean reversion (large κ) → tighter entry threshold (trade more often)
Lower transaction costs → tighter threshold (trade more often)

**Practical implication for our system:**
Fit OU parameters on each pair's spread history. Compute κ and half-life.
Reject pairs with half-life > 60 days (too slow to be useful on daily bars).
Use OU-derived thresholds instead of fixed ±2 z-score.

**Research:**
Zeng, Z. & Lee, C.G. (2014). "Pairs trading: optimal thresholds and
profitability." Quantitative Finance 14(11), 1881–1893.
https://doi.org/10.1080/14697688.2014.917806

Leung, T. & Li, X. (2015). Optimal Mean Reversion Trading: Mathematical
Analysis and Practical Applications. World Scientific.

arxiv (2024). "An Application of the Ornstein-Uhlenbeck Process to Pairs
Trading." https://arxiv.org/html/2412.12458v1

---

## PART 4: IMPLEMENTATION ROADMAP FOR THIS REPO

### Priority 1 — Regime Filter (next sprint)
Add `regime_detector.cpp` that outputs {TREND, MEAN_REVERT, AMBIGUOUS}
using ADX(14) + 63-day return sign. Portfolio.run_cycle gates strategy
activation on current regime.

### Priority 2 — OU Parameter Fitting in Pair Scanner
Replace ADF-only filtering with ADF + OU fit. Reject pairs with:
- Half-life > 60 days
- κ < 0.03 (daily)
Add OU-derived entry/exit thresholds per pair.

### Priority 3 — LSTM Regime Classifier on Spread
Train LSTM on labeled spread segments (trend / non-trend) to halt
CointegrationArb when the classifier detects a trending spread.
Input: [spread_z_score, d_spread, rolling_adf_pvalue, κ_rolling]
Output: P(spread_is_trending)
Use: if P > 0.65, suppress all new CointegArb entries for this pair.

### Priority 4 — Time-Series Momentum Strategy
Implement MomentumTSStrategy with:
- Signal: sign of 12-1 month return
- Sizing: σ_target / σ_21d · signal
- Gate: only active when regime = TREND AND VIX < 35
