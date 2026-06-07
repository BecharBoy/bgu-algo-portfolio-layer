# Arbitraging the Information Diffusion Lag
### A Prediction-Market-Anchored Framework for Cross-Sectional Equity Mispricings

> **Status: Backtesting Phase** — Core pipeline architecture is implemented. Active backtesting of ML engine and momentum strategy is underway.

---

## Overview

This system exploits the structural lag between **prediction market probability shifts** and the **cross-sectional repricing of individual equities** around macro and geopolitical events.

When a macro shock occurs (e.g., a Fed rate cut), HFT algorithms and ETF arbitrageurs instantly reprice broad indices — forcing all S&P 500 constituents to move in lockstep regardless of their individual fundamentals. This non-fundamental comovement creates a temporary mispricing window that corrects slowly as firm-specific capital flows in. We use [Polymarket](https://polymarket.com) as a real-time oracle to detect when this gap is opening — **before** the cross-sectional correction begins.

Full strategy is described in [`algotrading_strategy.pdf`](./algotrading_strategy.pdf).

---

## Pipeline Architecture

```
+--------------------------------------------------------+
|             Stage I: Polymarket Signal Intake          |
|      Per-minute ingestion -> tag ontology filter ->    |
|              duration gate [D_lo, D_hi]                |
+------------------------------+-------------------------+
                               | theta crossed?
                               v
+--------------------------------------------------------+
|             Stage II: LLM Semantic Router              |
|   f_LLM(event) -> {Ticker1, Ticker2, ..., TickerN}    |
|       RL feedback loop improves routing over time      |
+----------+----------------------------+----------------+
           | Sufficient historical data?| Novel event?
           v                            v
+---------------------+      +----------------------------+
|  Stage III-A        |      |  Stage III-B               |
|  ML Execution       |      |  Sentiment-Gated           |
|  Engine             |      |  Momentum Strategy         |
|  - SBC: direction   |      |  - Sentiment gate (news)   |
|  - Ridge: magnitude |      |  - ROC momentum            |
|  - Arbitrage gap Gi |      |  - Trailing ATR stop       |
+----------+----------+      +--------------+-------------+
           +------------------+--------------+
                              v
+--------------------------------------------------------+
|             Stage IV: Portfolio Risk Layer             |
|   Approve / Reduce / Reject / Paper-Trade / Review    |
+--------------------------+-----------------------------+
                           v
                   Execution via IB API
```

---

## Repository Structure

```
my_traders/
├── main.py                    # Entry point / orchestrator
├── main_backtesting/          # Backtesting framework (active)
├── strategies/                # Strategy implementations (III-A & III-B)
├── LLM/                       # LLM semantic router + RL feedback loop
├── Clustering_snp/            # S&P 500 sector/asset clustering
├── database/                  # PostgreSQL schema + ORM layer
├── Connection/                # IB API + Polymarket feed connectors
├── docs/                      # Architecture diagrams and notes
├── testing/                   # Unit + integration tests
├── theory_testing/            # Feasibility studies and POC scripts
├── second_phase_testing/      # Phase II validation scripts
├── general_testing/           # Misc experiments
└── algotrading_strategy.pdf   # Full strategy paper
```

---

## Key Components

### Stage I — Signal Intake
Polls the Polymarket API per minute. Filters the raw event universe by a curated tag ontology (`macro-indicators`, `equities`, `geopolitics`) and enforces a bounded duration window `D_lo <= D_e <= D_hi` to eliminate noise from ultra-short or long-horizon events.

### Stage II — LLM Semantic Router
A locally-hosted LLM maps triggered events to economically exposed tickers. Zero conversational text — pure clustering function `f_LLM(M) -> {T1, ..., Tn}`. A reinforcement learning loop rewards the router when mapped assets subsequently exhibit event-driven realized volatility.

### Stage III-A — ML Engine (Known Events)
- **Model I (SBC):** Predicts directional bias `Y_hat in {-1, +1}`
- **Model II (Ridge):** Forecasts drift magnitude `y_mag`
- **Entry logic:** Computes arbitrage gap `Gi = y_mag - |r_current|`. Trade forwarded only if `Gi > 0`

All models use **Asset-Event Specific Weights** (`W_{i,c}`) to prevent cross-contamination between event categories.

### Stage III-B — Sentiment-Gated Momentum (Novel Events)
- Sentiment gate requires corroborating signal from trusted financial news sources
- Entry on positive ROC momentum `Mt = (Pt - Pt-n) / Pt-n`
- Exit on trailing ATR stop `St = Hn - k * ATRn` or momentum reversal `Mt <= 0`

### Stage IV — Portfolio Risk Layer
Final checkpoint before execution. Integrates live portfolio state (PostgreSQL), Polymarket feeds, and IB market data. Enforces exposure caps at asset, sector, event-type, and strategy-channel levels. Includes a kill switch on daily loss / drawdown / consecutive loss thresholds.

---

## Current Status

| Component | Status |
|-----------|--------|
| Polymarket signal intake | Implemented |
| Tag ontology + duration filter | Implemented |
| LLM semantic router | Implemented |
| RL feedback loop | In progress |
| ML Engine (SBC + Ridge) | Backtesting |
| Sentiment-gated momentum | Backtesting |
| Portfolio risk layer | Implemented |
| IB API execution | Implemented |
| Full backtest evaluation | Active |

---

## Hyperparameters

All parameters are optimized via walk-forward cross-validation to prevent look-ahead bias.

| Parameter | Description |
|-----------|-------------|
| `S` | Target tag ontology — filters events by category |
| `D_lo`, `D_hi` | Event duration bounds |
| `T_s` | Observation window for asset price trajectory |
| `theta` | Oracle trigger threshold (Polymarket probability) |
| `sigma_min` | LLM RL reward threshold (minimum realized volatility) |
| `n` | Momentum lookback window (ROC + ATR) |
| `k` | ATR multiplier for trailing stop |

---

## Performance Metrics (Target)

- **Total Return** — net of transaction costs and rejected trades
- **Sharpe Ratio** — vs. buy-and-hold, momentum baseline, and no-signal ML baseline
- **Maximum Drawdown** — worst peak-to-trough loss
- **Hit Rate** — share of trades with correct direction and positive P&L
- **Classification Precision/Recall** — direction accuracy from Model I
- **Regression MAE/RMSE** — magnitude accuracy from Model II

---

## Theoretical Basis

- Wolfers & Zitzewitz (2004) — prediction markets as efficient information aggregators
- Da & Shive (2018) — ETF arbitrage induces non-fundamental comovement
- Boguth et al. (2023) — noisy FOMC returns and post-announcement reversals
- Ai, Han, Pan & Xu (2021) — heterogeneous monetary policy announcement premiums
- Duffie (2010) — asset price dynamics with slow-moving capital
- Diercks, Katz & Wright (2026) — Kalshi-implied densities outperform institutional forecasts

---

## Tech Stack

- **Python** — core pipeline and ML models
- **PostgreSQL** — position, trade history, and portfolio state
- **Interactive Brokers (IBKR TWS/Gateway)** — live execution
- **Polymarket API** — prediction market signal source
- **Local LLM** — semantic routing

---

*Author: Liran Attar — MSc CS, Ben-Gurion University*
