# 07_PAIR_HEALTH_CHECK_AND_OU_MODEL.md

## What This Document Is

Mathematical treatment of two related topics:
1. The Ornstein-Uhlenbeck process as the correct model for spread dynamics
2. Pair health monitoring — when to keep holding vs. force-close

---

## PART 1: THE ORNSTEIN-UHLENBECK PROCESS

### 1.1 Grade 4 Explanation

Imagine a dog on a rubber leash connected to a fixed pole.
The dog can wander, but the further it goes, the harder
the rubber pulls it back. If the dog goes very far right,
the pull is very strong. If it's near the pole, the pull
is weak. Eventually the dog always comes back near the pole.

The spread between two cointegrated stocks is exactly this.
The "pole" is the long-run mean of the spread.
The "rubber" is the cointegration relationship.

The question is: how strong is the rubber? How fast does
the dog come back? These are the OU parameters.

### 1.2 The Continuous-Time Model

The OU process is defined by the SDE:

    dS_t = κ(θ - S_t)dt + σ dW_t

Where:
- S_t = spread value at time t
- κ > 0 = mean-reversion speed (strength of the rubber)
- θ = long-run mean (position of the pole)
- σ > 0 = volatility (how much random noise the dog adds)
- W_t = standard Brownian motion

### 1.3 Solving the SDE — Full Derivation

This is a linear SDE. Solve using integrating factor e^{κt}.

Rewrite:
    dS_t + κ S_t dt = κθ dt + σ dW_t

Multiply both sides by e^{κt} (integrating factor):
    d(e^{κt} S_t) = κθ e^{κt} dt + σ e^{κt} dW_t

Integrate from 0 to t:
    e^{κt} S_t - S_0 = κθ ∫_0^t e^{κs} ds + σ ∫_0^t e^{κs} dW_s

    e^{κt} S_t = S_0 + θ(e^{κt} - 1) + σ ∫_0^t e^{κs} dW_s

Divide by e^{κt}:

    S_t = S_0 e^{-κt} + θ(1 - e^{-κt}) + σ ∫_0^t e^{-κ(t-s)} dW_s

**Properties derived from the solution:**

Expected value:
    E[S_t | S_0] = S_0 e^{-κt} + θ(1 - e^{-κt})
    → As t → ∞: E[S_t] → θ    (mean reversion proven)

Variance:
    Var[S_t | S_0] = (σ²/2κ)(1 - e^{-2κt})
    → As t → ∞: Var[S_t] → σ²/2κ    (bounded! not random walk)

This bounded variance is the mathematical proof that the
spread is stationary. Compare to random walk where Var grows
linearly in t — unbounded.

**Half-life of mean reversion:**
Time for expected deviation to decay by half:
    E[S_t - θ] = (S_0 - θ)e^{-κt}
    Half when e^{-κt} = 0.5
    t_{1/2} = ln(2) / κ ≈ 0.693 / κ

This is the most important single number for a pair:
it tells you how long to expect to hold a position.
κ = 0.03 daily → t_{1/2} = 23 days (fast, good)
κ = 0.005 daily → t_{1/2} = 139 days (too slow, avoid)

### 1.4 Discrete-Time Approximation (What We Actually Compute)

Daily bars means discrete time. The OU SDE discretizes to:

    S_t = S_{t-1} · e^{-κΔt} + θ(1 - e^{-κΔt}) + η_t

where η_t ~ N(0, σ²(1-e^{-2κΔt})/(2κ)) and Δt = 1/252 (annual).

This is exactly an AR(1) process:
    S_t = a + b · S_{t-1} + η_t

where:
    b = e^{-κΔt}           (AR coefficient)
    a = θ(1 - b)           (intercept)

Therefore: OLS regression of S_t on S_{t-1} directly
gives us the OU parameters.

### 1.5 MLE Parameter Estimation

Given spread observations S_1, ..., S_N:

**Step 1 — Regress S_t on S_{t-1}:**
    b̂ = Σ(S_t - S̄)(S_{t-1} - S̄_{lag}) / Σ(S_{t-1} - S̄_{lag})²
    â = S̄ - b̂ · S̄_{lag}

**Step 2 — Recover OU parameters:**
    κ̂ = -ln(b̂) / Δt
    θ̂ = â / (1 - b̂)

**Step 3 — Estimate σ:**
    residuals: η_t = S_t - â - b̂ · S_{t-1}
    s² = Var(η_t) = (1/N) Σ η_t²
    σ̂² = s² · 2κ̂ / (1 - e^{-2κ̂Δt})

**Pair rejection criteria based on OU:**
1. Reject if t_{1/2} = ln(2)/κ̂ > 60 days
   (spread too slow — capital locked too long)
2. Reject if σ̂²/(2κ̂) < min_spread_variance
   (spread too tight — no profitable amplitude)
3. Keep existing ADF p < 0.05 as pre-filter

### 1.6 Optimal Entry and Exit Thresholds (Zeng & Lee, 2014)

Current implementation uses fixed ±2σ z-score.
The OU model gives optimal thresholds from first principles.

**Problem formulation:**
Enter when S_t > a (for a short position).
Exit when S_t < b (b < a).
Transaction cost per trade: c ($ per dollar of position).

Maximize the Sharpe ratio of the strategy over infinite horizon.

**Result (closed-form approximation):**
The optimal entry threshold a* satisfies:

    a* - θ = σ/√(2κ) · f(c·κ/σ²)

where f is a function of the cost-to-variance ratio that
decreases as κ increases (faster mean reversion → tighter
entry threshold → trade more often).

**Intuition:**
- High κ (fast reversion): enter closer to mean (smaller a*-θ)
  because the spread comes back quickly even from small deviations
- Low κ (slow reversion): need larger deviation before entering
  because you'll hold longer and need bigger profit to cover costs

**Practical implementation:**
For each pair, compute a* = θ + λ · σ/√(2κ) with λ ≈ 1.5
instead of fixed λ=2. This gives per-pair calibrated thresholds.

**Research:**
Zeng, Z. & Lee, C.G. (2014). "Pairs trading: optimal thresholds
and profitability." Quantitative Finance 14(11), 1881–1893.
https://doi.org/10.1080/14697688.2014.917806

---

## PART 2: PAIR HEALTH MONITORING

### 2.1 The Problem

A pair entered at z = +2.3 (spread is high, expecting reversion).
30 days pass. Spread is at z = +3.1. Do we:
a) Hold — spread must come back eventually
b) Close — something has changed

The answer cannot be "always hold" because sometimes the
cointegration relationship permanently breaks. The answer
cannot be "always close after X days" because sometimes
legitimate slow mean reversion takes 60–90 days.

The correct answer: **close if and only if there is evidence
the statistical relationship has broken.**

### 2.2 Method 1 — Rolling ADF Recheck

Every 20 bars, re-run ADF on the spread using the last 90 bars:

    test_stat, p_value = ADF(spread_{t-90:t}, maxlags=5)

    if p_value > 0.10:
        → cointegration broken → force close immediately

**Mathematical justification:**
ADF tests H₀: ρ = 0 in ΔS_t = ρ·S_{t-1} + ΣᵢcᵢΔS_{t-i} + ε_t
p > 0.10 means we cannot reject a unit root → no mean reversion.

The 90-bar window is a tradeoff:
- Too short (30 bars): too many false positives (noisy test)
- Too long (252 bars): too slow to detect genuine breaks

**Research:**
Engle, R.F. & Granger, C.W.J. (1987). ibid.

### 2.3 Method 2 — Spread Exceedance Rule

Based on OU theory: if spread follows OU, the probability
of exceeding 3σ is:

    P(|S_t - θ| > 3σ_∞) = P(|Z| > 3√2) ≈ 0.003

(where σ_∞ = σ/√(2κ) is the stationary standard deviation)

Probability of exceeding 4σ_∞:
    P(|S_t - θ| > 4σ_∞) = P(|Z| > 4√2) < 0.0001

Therefore: if spread exceeds 4σ_∞ of the OU stationary
distribution, this is a near-impossible event under
the current OU model → model has broken → force close.

    if S_t - θ̂ > 4 · σ̂/√(2κ̂): force close, pair suspended

### 2.4 Method 3 — OU Parameter Drift Detection

Track the estimated κ̂ on a rolling window. If κ̂ drops below
a threshold — mean reversion is slowing down:

    if κ̂_rolling < 0.01 daily (half-life > 70 days):
        → flag pair for review, stop new entries
        → allow existing position to run with tighter stop

### 2.5 The Dog-on-Leash Scenario Revisited

In the example: Google fell 20%, recovered to -5%.
Small company fell 20%, took 1 month to recover.

During that month, the spread was large. Should we close?

Rolling ADF check: if the economic relationship is intact
(the small company is still Google's vendor), the ADF on
the 90-day spread window will still reject the unit root.
κ̂ may have temporarily dropped but remains positive.
The spread has not exceeded 4σ_∞.

→ Correct decision: hold.

The rolling ADF provides the statistical confidence to
stay in the trade. Without it, emotion (or a fixed time
stop) would force an exit at maximum loss.

---

## PART 3: IMPLEMENTATION IN C++

```cpp
struct OUParams {
    double kappa;      // mean-reversion speed (daily)
    double theta;      // long-run mean
    double sigma;      // volatility
    double half_life;  // ln(2) / kappa in days
    double stat_std;   // sigma / sqrt(2*kappa)
};

OUParams fit_ou(const std::vector<double>& spread) {
    int N = spread.size();
    // OLS of S_t on S_{t-1}
    double mean_lag = 0, mean_cur = 0;
    for (int i = 1; i < N; i++) {
        mean_lag += spread[i-1];
        mean_cur += spread[i];
    }
    mean_lag /= (N-1); mean_cur /= (N-1);

    double cov = 0, var = 0;
    for (int i = 1; i < N; i++) {
        cov += (spread[i-1] - mean_lag) * (spread[i] - mean_cur);
        var += (spread[i-1] - mean_lag) * (spread[i-1] - mean_lag);
    }
    double b = cov / var;
    double a = mean_cur - b * mean_lag;

    OUParams p;
    p.kappa     = -std::log(b);          // per bar (daily)
    p.theta     = a / (1.0 - b);
    double s2   = 0;
    for (int i = 1; i < N; i++) {
        double resid = spread[i] - a - b * spread[i-1];
        s2 += resid * resid;
    }
    s2 /= (N-1);
    p.sigma     = std::sqrt(s2 * 2*p.kappa / (1 - std::exp(-2*p.kappa)));
    p.half_life = std::log(2.0) / p.kappa;
    p.stat_std  = p.sigma / std::sqrt(2.0 * p.kappa);
    return p;
}

bool is_pair_healthy(const std::vector<double>& spread_window,
                     const OUParams& ou, double current_spread) {
    // Condition 1: Rolling ADF (simplified — use full ADF library)
    double adf_pvalue = run_adf_test(spread_window);
    if (adf_pvalue > 0.10) return false;

    // Condition 2: Spread exceedance
    double deviation = std::abs(current_spread - ou.theta);
    if (deviation > 4.0 * ou.stat_std) return false;

    // Condition 3: Half-life still acceptable
    if (ou.half_life > 60.0) return false;

    return true;
}
