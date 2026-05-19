# my_traders

A production-grade macro event trading system that monitors Polymarket prediction markets, detects tradable macroeconomic signals, and executes cross-sectional equity positions through Interactive Brokers.

---

## What This System Does

Prediction markets reprice macro expectations continuously. Equity indices absorb the first shockwave nearly instantly via ETF arbitrage. Individual stocks, however, take minutes to days to fully reprice according to their fundamental macro sensitivities. This system exploits that lag.

**The edge:**
1. Polymarket signals a macro surprise (CPI, FOMC, NFP) before the official release.
2. ETF arb collapses the index immediately — we do not compete with that.
3. Within the cross-section, high-sensitivity stocks diverge from low-sensitivity ones over the following hours.
4. We enter after the initial noise, hold through the repricing, and exit when divergence peaks.

**Feasibility is proven.** Historical backtesting across CPI and FOMC events (2024–2025) confirms that the information diffusion lag is real and exploitable. See `theory_testing/` for results and `docs/` for the research paper.

---

## System Architecture

The system is divided into two concerns: **research** (offline) and **runtime** (live).

```
my_traders/
├── Connection/            # IB Gateway integration (IB.py, Wrapper.py, API.py)
├── database/              # PostgreSQL schema, models, connector (DB.py, schemas.py)
├── strategies/            # Strategy logic and orchestration (Strategy.py)
├── Clustering_snp/        # S&P 500 event-response clustering (offline research)
├── theory_testing/        # Feasibility backtesting scripts and outputs
├── general_testing/       # Integration and smoke tests
├── testing/               # Unit tests
├── docs/                  # Research paper (LaTeX / PDF), figures
├── main.py                # Runtime entrypoint
├── jobs.py                # Scheduled jobs (polling, signal detection)
├── config.py              # Environment and runtime config
├── requirements.txt       # Python dependencies
└── README.md
```

### Runtime Flow

```
Polymarket REST API (polling)
        │
        ▼
Signal Detector — ΔP > γ and volume-confirmed?
        │
        ├── No  → sleep, next poll
        │
        └── Yes → Event classified (CPI / FOMC / NFP / ...)
                        │
                        ▼
              Basket Loader — load pre-clustered stock universe
              (HIGH sensitivity shorts + LOW sensitivity longs)
                        │
                        ▼
              Position Sizer — risk-based sizing, optional market-neutral weights
                        │
                        ▼
              IB Gateway — order submission, fill tracking, position management
                        │
                        ▼
              State Store (PostgreSQL) — log signal, trades, PnL, runtime state
```

---

## Key Components

### `Connection/`
IB Gateway integration. Handles connection lifecycle, order routing, fill callbacks, and portfolio state. Built on the official IBKR Python API.

### `database/`
PostgreSQL-backed persistence layer. Stores event logs, Polymarket probability snapshots, trade history, beta/cluster artifacts from research, and runtime state.

### `strategies/`
Strategy orchestration. Currently contains the macro-event cross-sectional strategy. Entry/exit logic, signal thresholding, and basket construction live here.

### `Clustering_snp/`
Offline research module. Clusters the S&P 500 universe by realized response to each macro event type (CPI hawkish, CPI dovish, FOMC hold, etc.). Outputs labeled baskets consumed by the runtime.

### `theory_testing/`
Feasibility backtesting. Event-study scripts that validate whether the information diffusion lag is detectable and tradable. Results are the empirical foundation for the strategy. **Do not move production code here.**

---

## Backtesting Status

Hypothesis backtesting is **complete**. The arbitrage gap is confirmed across:

| Event | Direction | Ticker | Strategy Return (H=6) | SPY Return (H=6) |
|---|---|---|---|---|
| CPI Hot April 2024 | Hawkish SHORT | RUN | +8.87% | +0.01% |
| CPI Hot April 2024 | Hawkish SHORT | UPST | +2.98% | -0.09% |
| FOMC Jan 2025 | Hawkish SHORT | RGTI | +14.21% | -0.23% |
| FOMC Jan 2025 | Hawkish SHORT | RGTI | +21.77% (H=75) | -2.17% |

**Algorithm backtesting is in progress.** The next phase adds entry/exit rules, position sizing, and transaction costs to produce a proper equity curve.

---

## Stock Selection Strategy (Next Phase)

Manually curated baskets work for initial testing but do not scale. The roadmap:

1. **Cluster the S&P 500** by event-response signature (already started in `Clustering_snp/`).
2. **Label clusters** per event type: CPI-hawkish-sensitive, FOMC-rate-sensitive, NFP-cyclical, etc.
3. **Auto-select** short and long baskets at runtime from cluster labels + live beta estimates.
4. This replaces hardcoded tickers with a data-driven, generalizable approach.

---

## Entry / Exit Design (Open Question)

The backtesting phase is helping answer this. Current candidates:

- **Entry:** T₀ + delay (avoid initial ETF noise), triggered after signal confirmation.
- **Exit options:** fixed holding window (2–24h), PnL target, or detected divergence peak.
- **Market neutrality:** optional. Useful for risk control but may reduce alpha capture. Decision pending empirical comparison.

---

## Infrastructure

The system runs 24/7 on a dedicated server. No Docker dependency for runtime — direct process management. Polymarket polling runs on a scheduler. IB Gateway maintains a persistent TCP connection.

**Requirements:** Python 3.11+, PostgreSQL, IBKR TWS/Gateway (paper or live).

```bash
pip install -r requirements.txt
cp .env.example .env          # fill in your credentials
python main.py
```

---

## What Is NOT in This Repository

- Raw CSV backtesting outputs (generated artifacts, not source)
- `.idea/` or other IDE metadata (gitignored)
- `.env` secrets (use `.env.example` as template)
- Exploratory one-off scripts — these live in `theory_testing/` during research, then get cleaned up

---

## Research Paper

The theoretical motivation and empirical backtesting methodology are documented in `docs/`. The paper covers:

- Information Diffusion Lag Hypothesis
- Mathematical framework (macro-factor pricing, beta estimation)
- Hypothesis backtesting results (feasibility confirmed)
- Algorithm backtesting methodology (in progress)
- Future: clustering pipeline and full strategy evaluation

---

## Roadmap

### Phase 1 — Cleanup ✅
- Separated research code from runtime
- Removed throwaway scripts from root
- Standardized module naming

### Phase 2 — Hypothesis Backtesting ✅
- Event studies across CPI and FOMC events
- Confirmed information diffusion lag exists
- Identified best candidate tickers per event type

### Phase 3 — Algorithm Backtesting (In Progress)
- Add entry/exit rules and holding window logic
- Include transaction costs and borrow fees (short leg)
- Produce equity curve and drawdown analysis
- Compare market-neutral vs directional variants

### Phase 4 — Clustering Pipeline
- Cluster S&P 500 by macro-event response
- Export labeled baskets for runtime consumption
- Validate cluster stability across event dates

### Phase 5 — Live Runtime
- Stabilize IB Gateway execution path
- Add runtime state, replay safety, health checks
- Paper trading validation before live capital
