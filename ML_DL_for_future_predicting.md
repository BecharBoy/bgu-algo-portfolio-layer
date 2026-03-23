# 05_ML_DL_SIGNAL_MODELS.md

## What This Document Is

A full mathematical treatment of every ML/DL model relevant
to signal generation in my_traders. Each model is explained
from zero, proven mathematically, and evaluated for trading
applicability. Written like a submission to a CS/statistics
research journal — but readable by anyone.

---

## 1. Linear Regression as a Forecasting Baseline

### 1.1 What It Is

Before any neural network, the right question is: does
a simple linear model have predictive power? If it does,
the signal exists. If it doesn't, a deeper model won't save it.

### 1.2 The Model

Given feature vector xₜ ∈ ℝᵈ at time t, predict next-day
return rₜ₊₁:

    r̂ₜ₊₁ = wᵀ xₜ + b

Where w ∈ ℝᵈ is the weight vector and b is the bias.

Features xₜ might include:
- RSI_t, MACD_t, BB_position_t (how far price is from bands)
- Sentiment_t (Reddit/news score)
- Spread_t (current spread z-score for stat arb pairs)
- VIX_t (market volatility regime)

### 1.3 Training via OLS

Minimize MSE over training samples:

    min_w Σₜ (rₜ₊₁ - wᵀxₜ - b)²

Closed-form solution:
    w* = (XᵀX)⁻¹ Xᵀ r

Where X is the feature matrix (T × d) and r is the
return vector. This is the same OLS derivation as in MD #1
but applied to prediction rather than hedge ratio estimation.

### 1.4 Why Start Here

- Interpretable: each feature has a coefficient you can
  examine
- Fast: closed-form, no gradient descent needed
- Baseline: if your LSTM achieves the same accuracy as OLS,
  the LSTM is not adding value — it's just more complex noise

---

## 2. LSTM (Long Short-Term Memory)

### 2.1 What It Is

An LSTM is a recurrent neural network designed to model
sequences. Unlike simple RNNs, LSTMs explicitly track what
to remember, what to forget, and what to output — making
them capable of learning long-range dependencies in time series.

### 2.2 The LSTM Cell — Full Derivation

At each time step t, an LSTM cell takes input xₜ and
previous hidden/cell states (hₜ₋₁, cₜ₋₁) and computes:

**Forget gate:** Decides what to erase from cell state
    fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)

**Input gate:** Decides what new information to store
    iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)

**Candidate cell state:** Proposed new values
    c̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)

**Cell state update:**
    cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ

**Output gate:**
    oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)

**Hidden state:**
    hₜ = oₜ ⊙ tanh(cₜ)

Where σ is sigmoid, ⊙ is elementwise multiplication,
and W matrices + b vectors are learned via backpropagation
through time (BPTT).

### 2.3 Why Gates Solve the Vanishing Gradient Problem

Standard RNNs suffer from vanishing gradients: gradients
shrink exponentially as they propagate back through time,
making it impossible to learn dependencies beyond ~10
time steps.

The LSTM cell state cₜ flows through time with only
elementwise multiplication by fₜ (the forget gate).
If fₜ ≈ 1, the gradient through cₜ is:

    ∂cₜ/∂cₜ₋₁ = fₜ ≈ 1

This is the "constant error carousel" — gradients do not
decay when the forget gate is open. The network can
learn dependencies across 100+ time steps.

### 2.4 Trading Application

**Input sequence:** Last 60 days of features per ticker
    X = [x₁, x₂, ..., x₆₀] where xₜ ∈ ℝᵈ

**Output:** Predicted return direction (binary: +1 or -1)
or magnitude (regression output)

**Architecture:**
```
Input (60 × d)
    ↓
LSTM layer 1 (128 units)
    ↓
Dropout (0.2)
    ↓
LSTM layer 2 (64 units)
    ↓
Dense (1) + sigmoid (classification) or linear (regression)
    ↓
Signal: if output > 0.55 → BUY, if < 0.45 → SELL
```

### 2.5 Why It Can Fail for Financial Data

1. **Non-stationarity:** Financial return distributions
   shift over time. A model trained on 2020-2022 will
   fail on 2024-2026 because the market regime changed.
   Fix: retrain monthly with a rolling window.

2. **Low signal-to-noise ratio:** Daily stock returns have
   a Sharpe ratio near zero at the individual stock level.
   The model is trying to learn signal from near-pure noise.
   Fix: use longer prediction horizons (5-day returns)
   or cross-sectional ranking (rank stocks by predicted
   return, trade the top/bottom quintile).

3. **Overfitting:** LSTM has thousands of parameters.
   With 2 years of daily data per stock (500 samples),
   you have far fewer samples than parameters.
   Fix: aggressive regularization (L2, Dropout 0.3-0.5),
   cross-validation, out-of-sample testing.

**Research:**
- Fischer & Krauss (2018), "Deep learning with long
  short-term memory networks for financial market
  predictions", EJOR 270(2).
  https://doi.org/10.1016/j.ejor.2017.11.054
- Siami-Namini et al. (2019), "A Comparison of ARIMA and
  LSTM in Forecasting Time Series", IEEE ICMLA.

---

## 3. Temporal Fusion Transformer (TFT)

### 3.1 What It Is

TFT is Google's purpose-built architecture for multi-horizon
time series forecasting with mixed input types (static
features, time-varying known inputs, time-varying unknown
inputs). Outperforms LSTM on most financial forecasting
benchmarks in published literature.

### 3.2 Key Components

**Variable Selection Networks:** Learned soft selection
of which features matter at each time step. Instead of
using all features equally, the model learns:
"RSI matters more than ATR for this stock at this time."

Implemented as:

    ξₜ = Σⱼ aₜⱼ · hₜⱼ

Where hₜⱼ is the processed feature j and aₜⱼ is its
attention weight (from a softmax over learned importances).

**Gated Residual Networks (GRN):**
    GRN(a, c) = LayerNorm(a + GLU(η₁))
    η₁ = W₂ · ELU(W₁·a + W₃·c + b₁) + b₂
    GLU(x) = x[:d] ⊙ σ(x[d:])

This allows the network to skip irrelevant computations
with near-zero gates.

**Multi-Head Attention:**
    Attention(Q, K, V) = softmax(QKᵀ/√dₖ) V

Applied across time steps — the model learns which past
time steps are most relevant for the current prediction.

### 3.3 Why TFT > LSTM for Your Use Case

| Property | LSTM | TFT |
|---|---|---|
| Handles static features (sector, market cap) | ❌ | ✅ |
| Multi-step forecast (5-day return) | Poor | Strong |
| Feature importance output | ❌ | ✅ (interpretable) |
| Training stability | Moderate | High |
| Published finance benchmarks | Moderate | Strong |

The interpretability is especially valuable: TFT tells you
which features drove each prediction. When your model
generates a BUY signal, you can see it was 70% driven by
sentiment score and 30% by spread z-score.

**Research:**
- Lim, Arık, Loeff & Pfister (2021), "Temporal Fusion
  Transformers for Interpretable Multi-horizon Time Series
  Forecasting", International Journal of Forecasting.
  https://doi.org/10.1016/j.ijforecast.2021.03.012

---

## 4. BERT / FinBERT for Sentiment Scoring

### 4.1 What It Is

BERT (Bidirectional Encoder Representations from
Transformers) is a language model pre-trained on massive
text corpora. FinBERT is fine-tuned specifically on
financial text (analyst reports, earnings calls, financial
news).

### 4.2 The Transformer Attention Mechanism

The core of BERT is multi-head self-attention over token
embeddings. Given input tokens t₁, ..., tₙ:

    Q = X·Wq,  K = X·Wk,  V = X·Wv

    Attention(Q, K, V) = softmax(QKᵀ/√dₖ) · V

Each token attends to all other tokens. "Iran war oil"
— the word "oil" attends strongly to "Iran" and "war"
because they co-occur in financial contexts.

The output is a contextual embedding for each token.
The [CLS] token's embedding is used as the sentence
representation for classification.

### 4.3 Fine-Tuning for Sentiment

Add a classification head on top of the [CLS] embedding:

    logits = W · h_CLS + b
    P(positive) = softmax(logits)[1]

Fine-tune on labeled financial text:
- PhraseBank dataset (2800 analyst sentences, labeled)
- Financial Twitter dataset (Malo et al., 2014)

Output: P(positive) ∈ [0,1] per text piece.
Sentiment score: sₜ = P(positive) - P(negative) ∈ [-1, 1]

### 4.4 Practical Implementation

```python
from transformers import pipeline

sentiment_pipeline = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    return_all_scores=True
)

def score_text(text: str) -> float:
    results = sentiment_pipeline(text[:512])
    scores  = {r["label"]: r["score"] for r in results}
    return scores.get("positive", 0) - scores.get("negative", 0)
```

**Cost:** FinBERT inference is ~50ms per text on CPU,
~5ms on GPU. For 1000 Reddit posts/day: 50 seconds on CPU
(run offline), 5 seconds on GPU.

**Research:**
- Araci (2019), "FinBERT: Financial Sentiment Analysis
  with Pre-trained Language Models", arXiv:1908.10063.
  https://arxiv.org/abs/1908.10063

---

## 5. Reinforcement Learning (RL) — The True Prediction Engine

### 5.1 What It Is

All models above are **supervised**: trained on (features,
label) pairs where the label is a historical return.
The problem: historical returns are not the same as
the optimal trading decision. A return of +1% might be
a good exit (if you entered at -3%) or a terrible entry
(if the stock goes to +15% tomorrow).

Reinforcement Learning frames trading correctly:
an **agent** takes **actions** (BUY, SELL, HOLD) in an
**environment** (the market), receives **rewards** (PnL),
and learns a **policy** (when to take which action) that
maximizes cumulative reward.

### 5.2 The MDP Formulation

A Markov Decision Process (MDP) for trading:

**State** sₜ: market features at time t
    sₜ = [RSI_t, MACD_t, spread_t, sentiment_t, position_t]

**Action** aₜ ∈ {BUY, SELL, HOLD}

**Reward** rₜ:
    rₜ = PnL_t - λ·TC_t

Where TC_t is transaction cost and λ penalizes overtrading.

**Policy** π(aₜ|sₜ): probability of each action given state

**Value function:** Expected cumulative discounted reward:
    Vᵝ(s) = E[Σₜ γᵗ rₜ | s₀ = s, π = β]

Where γ ∈ (0,1) is the discount factor.

### 5.3 Q-Learning / DQN

Q-function: expected return from taking action a in state s:

    Q(s, a) = E[Σₜ γᵗ rₜ | s₀=s, a₀=a, π]

Bellman equation:
    Q(s, a) = r + γ · max_{a'} Q(s', a')

DQN (Deep Q-Network) approximates Q(s,a) with a neural
network. Training update:

    L(θ) = E[(r + γ·max_{a'} Q(s', a'; θ⁻) - Q(s, a; θ))²]

Where θ⁻ are "target network" weights updated periodically
for stability.

### 5.4 Why RL Is Hard for Finance

1. **Non-stationarity:** The environment (market) changes.
   A policy learned on 2020 data is suboptimal in 2026.

2. **Sparse rewards:** A trade might take 5 days to close.
   You don't know if the action was correct until the
   position closes. Long reward delays make learning slow.

3. **Overfitting to historical environment:** The agent
   learns to exploit specific historical patterns that
   may not repeat.

4. **Transaction cost sensitivity:** A small error in
   TC modelling makes the learned policy unprofitable
   in production even if it looks good in simulation.

**Best current approach for your system:** PPO (Proximal
Policy Optimization) with a shared LSTM backbone for
state encoding. This is state-of-the-art for trading RL.

**Research:**
- Mnih et al. (2015), "Human-level control through deep
  reinforcement learning", Nature 518.
  https://doi.org/10.1038/nature14236
- Jiang, Xu & Liang (2017), "A Deep Reinforcement Learning
  Framework for the Financial Portfolio Management Problem",
  arXiv:1706.10059.
- Schulman et al. (2017), "Proximal Policy Optimization
  Algorithms", arXiv:1707.06347.

---

## 6. Recommended Model Roadmap for my_traders

```
Phase 1 (now):    Rule-based signals (MeanReversion, StatArb)
                  → backtested → Sharpe measured

Phase 2 (3 months): Linear regression on features from Phase 1
                    → does adding ML improve Sharpe?

Phase 3 (6 months): FinBERT scoring of Reddit + news
                    → do sentiment features add predictive power?
                    → Granger test before building full pipeline

Phase 4 (12 months): TFT on full feature set
                     → multi-stock, multi-horizon forecast
                     → cross-sectional ranking signal

Phase 5 (MSc level): RL agent integrating all signals
                     → the agent decides position sizing
                     + entry/exit simultaneously
```

Do not skip phases. Each phase validates that the signal
exists before building more complexity on top of it.
