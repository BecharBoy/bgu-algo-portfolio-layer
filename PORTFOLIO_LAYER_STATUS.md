# Portfolio Layer — Current Status

**Source of truth for "what is true right now."** This file is grounded in the actual code on
`master` (through commit `6911641`) and the target spec `PORTFOLIO_LAYER_PLAN.md`. It is intentionally
honest about the gap between *implemented + unit-tested* and *verified end-to-end on real data*.

> **One-line status:** The Milestone 1 / Option 2 Core logic is implemented and passing offline unit
> tests (18 passed, 1 skipped), and the blocking correctness bugs found in review have been fixed.
> **It is not yet verified against the real DB/pipeline.** Do not treat the layer as "done" until the
> legacy and passthrough checks in Part 2 pass.

### Terminology (matches the approved plan)
- **Milestone 1 / Option 2 Core** — the first shippable core: **risk-based downside-to-stop sizing as
  the default sizer**, two-pass replay, one shared cash / MTM-equity base, structured decisions, the
  core caps (position / event / sector / gross / cash) + **portfolio heat**, and long/short cash-margin
  accounting.
- **Milestone 2 / Core hardening** — theme / country / liquidity controls, drawdown-scaling +
  kill-switch validation, Polymarket volume-gate enforcement, paper-trade routing, and other non-core
  hardening.
- **Future / separate workstreams** — PortfolioMonitor, joint construction, factor/beta/correlation,
  confidence-weighted sizing, learning loop, execution adapter, DB persistence.

> **Note on terminology:** risk-based sizing is part of **Milestone 1 / Option 2 Core**, not a later
> phase. Earlier drafts framed risk-based sizing as "Option B / Phase 2"; the approved plan makes
> Option 2 Core the **first** shippable deliverable, with fixed-fraction sizing only an internal
> bring-up scaffold.

---

## 1. Current behavior — what the Portfolio Layer does today

### How it connects to the existing backtester
The layer is **additive** and gated by `BacktestConfig.portfolio_enabled` (default `False`).

- **`portfolio_enabled=False` (default / legacy):** behavior is unchanged — each signal opens a fixed
  `trade_notional` (≈ $1,000) position and is saved immediately. No portfolio code runs, no portfolio
  reports are written. (`main_backtesting/stages/simulation.py`, legacy branches at `:474` and `:654`.)
- **`portfolio_enabled=True` (portfolio mode):** the simulation stage runs a **two-pass** flow instead
  of booking trades inline. Wiring lives in `main_backtesting/stages/simulation.py` and
  `main_backtesting/stages/simulation_portfolio.py`.

The connection point is exactly where the old code booked a trade: the `Trade` returned by the
strategy is treated as a **candidate**, and booking is deferred to Pass 2.

### What Pass 1 does (capital-free, existing logic)
Per work item, the upstream gates and `simulate_one_trade()` run unchanged to compute each trade's
**path**: `entry_at`, `exit_at`, `entry_price`, `exit_price`, `exit_reason`, and the ATR stop geometry.
In portfolio mode the resulting trade is **not saved**; instead a `TradeCandidate` is assembled
(`assemble_candidate`, `simulation_portfolio.py:49`) and persisted as JSON into the simulation stage's
`stage_work.result.pass1_candidates` (`simulation.py:450-472`, `:631-653`). Momentum **shadow** grid
variants stay capital-free diagnostics — only the selected variant produces a candidate.

### What Pass 2 does (portfolio replay)
After all work items finish, every `pass1_candidate` is reloaded
(`load_pass1_candidates_from_work`), bars are reloaded by key, and the candidates are replayed on **one
global timeline** (`run_portfolio_pass2`, `simulation_portfolio.py:135`; `replay_portfolio`,
`portfolio/replay.py:52`):

- Events are sorted **by timestamp first, then exits-before-entries within the same timestamp**, then a
  deterministic tie-break (`entry_at`, `market_id`, `symbol`) — `portfolio/replay.py:10-49`.
- At each exit event the position is closed; at each entry event `Portfolio.evaluate()` decides and, if
  approved, books a sized `Trade`.
- **Invariant:** the portfolio only scales **quantity**. Entry/exit timestamps, prices, and exit
  reasons come from Pass 1 and are never changed.

Booked trades are saved to `historical_trades`, and portfolio reports are written. **Before re-saving,
Pass 2 deletes _all_ `machine_learning` and `polymarket_momentum` trade rows for the run** — not only
rows previously booked by the portfolio layer — so replay is idempotent
(`simulation_portfolio.py:146-153`: `DELETE ... WHERE run_id = $1 AND portfolio IN ('machine_learning',
'polymarket_momentum')`). In portfolio mode Pass 2 re-creates every booked row, so this is intended;
but because the delete is broad, **its scope must be verified in the DB/pipeline smoke test (Part 2.3)**
to confirm it does not remove rows that should survive (e.g. legacy/other-portfolio rows for the same
run).

### What decisions it makes
For every entry candidate, `Portfolio.evaluate()` (`portfolio/portfolio.py:55`) returns a structured
`PortfolioDecision` with status:

- `approved` — booked at full risk-target size;
- `approved_capped` — booked, but a cap reduced the size (binding constraint recorded);
- `rejected` — a hard gate failed or the size fell below the minimum;
- `kill_switch_blocked` — produced if the kill-switch state is active.

`paper_trade`, `reduce_only`, `human_review` exist in the enum but are not produced by the core
(reserved). The evaluation order matches spec §9.3: validity → missing-data normalization → kill-switch
→ short handling → duplicate/count gates → volume gate (gated, see below) → risk sizing → caps →
min-notional/cash → final decision.

### How risk-based sizing works
Default sizer is **risk-based downside-to-stop** (`sizing_mode="risk_based"`, `portfolio/config.py:22`;
math in `portfolio/sizing.py`):

```
risk_dollars  = effective_risk_pct × equity        # effective = base × drawdown derisk factor
qty_target    = risk_dollars / stop_distance        # stop_distance = |entry − initial_stop|
notional_risk = qty_target × entry_price
```

Each trade therefore risks ~`risk_per_trade_pct` of live equity to its stop. `fixed_fraction` exists
**only** as a bring-up scaffold and is not the default. The layer only ever sizes **down** from the
risk target, never up.

### What caps / checks are currently implemented
Caps are computed as "remaining headroom" ceilings and the final size is the **minimum** of the risk
target and all active caps (`portfolio/constraints.py:96`, applied in `portfolio/portfolio.py:155`):

- **Hard gates (reject):** invalid entry price / stop, invalid candidate, kill-switch active, short in
  long-only mode, duplicate `(event_id, symbol)`, `max_open_positions`, below-min notional, insufficient
  cash.
- **Size caps (downsize):** position (`max_position_notional_pct`), event (`max_event_exposure_pct`),
  sector (`max_sector_exposure_pct`), gross (`max_gross_exposure_pct`), net (`max_net_exposure_pct`),
  heat (`max_portfolio_heat_pct`), liquidity/ADV (`max_adv_participation_pct`), theme, country.
- Tie-break for the binding cap is the fixed `CAP_ORDER` (`portfolio/models.py:23`).

> Note: theme, country, liquidity/ADV, and net caps are **coded and active when their data is present**,
> but per the plan they are Milestone 2 controls. In Milestone 1 they are typically inert (tags/ADV often
> absent → cap skipped). Treat them as not-yet-validated policy.

### How cash, margin, MTM equity, heat, and exposure are managed
State lives in `PortfolioState` (`portfolio/models.py:106`); bookkeeping in `portfolio/portfolio.py`
and `portfolio/mtm.py`:

- **Cash:** on open, `cash -= notional + commission`; on close, `cash += notional + gross_pnl −
  exit_commission` (`portfolio.py:244-284`). Cash-conservation holds for longs and shorts.
- **Margin (short):** 100% cash margin (`short_margin_pct=1.00`) — a short reserves its notional as cash;
  no borrow fees (MVP).
- **MTM equity (point-in-time):** `equity = cash + Σ mark(open positions)` using the **last bar close at
  or before** the timestamp per position resolution (`portfolio/mtm.py:31-50`). Sizing uses this live
  equity. Shorts are marked as `qty × (2·entry − mark)`.
- **Portfolio heat:** `Σ qty × stop_distance` (initial stops), updated on open/close
  (`portfolio.py:345-363`); capped at `max_portfolio_heat_pct`. This is the real aggregate-risk governor.
- **Exposure buckets:** gross/net and per-event/sector/theme/country gross notional, updated with a
  `+1/−1` multiplier on open/close so they release correctly on exit.
- **Invariants:** NaN accounting always aborts the run; negative-cash-while-holding aborts in normal
  mode (relaxed only in the caps-disabled passthrough profile, where unlimited capital is intentional)
  — `portfolio/models.py:139`.

### What reports/files are currently produced
Written to `run_dir/reports/` after Pass 2 (`portfolio/reporting.py:126`):

- `portfolio_decisions.csv` — one row per candidate (status, reason, binding constraint, quantity,
  notional, risk_dollars, effective_risk_pct, equity/cash/heat before).
- `portfolio_equity_curve.csv` — timestamp, cash, equity, drawdown.
- `portfolio_exposure.csv` — gross/net exposure, heat, open position count.
- `portfolio_summary.json` — start/final equity, final cash, max drawdown, decision counts, booked count.
- `trade_alpha.csv` — per-trade attribution **scaffold** (no benchmark/alpha math yet).
- `baseline_vs_portfolio.csv` — **stub** (single `booked_trade_count` row; legacy column blank).
- `PORTFOLIO_INTEGRATION_NOTES.md` — assumptions summary.

### What is implemented but gated, off by default, or not yet validated
- **Polymarket volume-gate enforcement:** behind `enforce_polymarket_volume_gate` (default `False`,
  `config.py:52`). Quality is still computed and logged; it does **not** reject trades by default.
- **`fixed_fraction` sizer:** scaffold only; never the default.
- **`paper_trade` routing:** code paths exist but `paper_trade_enabled=False`.
- **Drawdown-derisk schedule + kill-switch:** **active under the default `PortfolioConfig`** — not off.
  The default schedule `{0.10: 0.5, 0.20: 0.0}` halves per-trade risk at ≥10% drawdown and stops new
  risk at ≥20%, and the kill-switch trips at `kill_switch_drawdown_pct = 0.20`
  (`config.py:39-42`, `sizing.py:12`, `mtm.py:53-68`). They are **coded and on by default but not yet
  validated end-to-end**, so they are classified under **Milestone 2 / hardening** for validation. (Only
  the passthrough profile disables them.)
- **`gate_applied` diagnostic:** set to `True` only in portfolio mode; legacy mode keeps `False`
  (pre-portfolio parity) — `simulation.py:341-349`.

---

## 2. Required verification before continuing

None of the items below can be run offline; they need the DB and a populated pipeline. **Until they
pass, the layer is "implemented + unit-tested," not "verified."**

### 2.1 Legacy mode preserved — Test 38
- **Purpose:** prove the Pass-1/Pass-2 refactor did not disturb the existing fixed-$1,000 path.
- **How:** run a known historical backtest with `portfolio_enabled=False` and compare to a
  pre-portfolio baseline.
- **Pass:** opened trades, their entry/exit timestamps and prices, and the legacy reports are identical
  to the baseline; `gate_applied` still reads `False` in legacy decision logs.
- **Fail:** any difference in the opened-trade set or timing — indicates the refactor changed legacy
  behavior; must be fixed before anything else.

### 2.2 Passthrough regression — Test 39
- **Purpose:** prove that with all caps off and unlimited capital, the portfolio path opens exactly the
  same trades as the current system (only sizing differs). This is the Milestone 1 gate.
- **How:** run `portfolio_enabled=True` with `PortfolioConfig().with_passthrough_profile()`.
- **Pass:** the set of opened trades and their `entry_at`/`exit_at` timestamps match the current system
  exactly; only `quantity` differs.
- **Fail:** any trade missing, added, or retimed — the refactor altered signal/entry/exit behavior.
- **Note:** an offline *proxy* of this exists (synthetic candidates) and passes; the real test on
  pipeline-produced candidates is still required.

### 2.3 Real portfolio-mode smoke run
- **Purpose:** confirm the full path runs and produces sane portfolio output on real data.
- **How:** full 7-stage run with default `PortfolioConfig()`.
- **Pass:** Pass 2 completes; `portfolio_decisions.csv` shows a sensible mix of approved/capped/rejected;
  equity curve and exposure look coherent; sized quantities differ from $1,000; `final_equity` ≈ start ±
  strategy PnL. **Also confirm the Pass-2 delete scope** (`simulation_portfolio.py:146-153`) only removes
  the run's `machine_learning` / `polymarket_momentum` rows it then re-creates, and does not wipe rows
  that should survive.
- **Fail:** crash, empty/implausible reports, quantities that don't reflect sizing, or the delete
  removing trade rows that are not re-booked by Pass 2.

### 2.4 DB/pipeline round-trip (Pass-1 serialization + Pass-2 replay)
- **Purpose:** confirm the integration seams work on real rows — candidate JSON written to and reloaded
  from `stage_work`, and the Pass-2 replay reconstructs state deterministically from empty.
- **Pass:** candidates reload without loss; the same run produces identical decisions/equity on re-replay.
- **Fail:** serialization loss, missing fields, or non-deterministic Pass-2 output.

### 2.5 Real metadata coverage — `asset_metadata`, sector, ADV, volume
- **Purpose:** the caps depend on real data presence; confirm coverage on the live symbol universe.
- **Pass:** sector resolves (or falls back to `UNKNOWN` with a warning); ADV computes where bars exist;
  volume quality present where `condition_id` exists; missing data degrades the affected cap (per §10.1),
  never mis-sizes.
- **Fail:** crashes on missing metadata, or caps silently sizing on bad/absent data.

### 2.6 Resume path
- **Purpose:** confirm an interrupted simulation reloads `pass1_candidates` and Pass 2 still replays from
  empty deterministically.
- **Pass:** resumed run yields the same Pass-2 result as an uninterrupted run.
- **Fail:** partial state leaks or differing output.

### Offline status (already done)
- `testing/test_portfolio_layer.py`: **18 passed, 1 skipped.** The skip is the legacy `gate_applied`
  parity test, which imports the simulation stage and needs `asyncpg` (absent in a pure offline
  checkout). It should run green in the Docker/DB environment — confirm there.

---

## 3. Remaining build work (only after verification passes)

**Milestone 1 completeness (reporting/comparison):**
- **Real before/after legacy harness** — run the same candidate set through both paths and emit
  comparative metrics into `baseline_vs_portfolio.csv` (currently a one-row stub). Spec §18 deliverable.
- **Charts/PNGs** — `portfolio_equity_curve.png`, `portfolio_drawdown.png`, `baseline_vs_portfolio.png`;
  standalone `portfolio_drawdown.csv`.

**Attribution (partly M1 schema, substance M2):**
- Real alpha vs benchmark: `alpha = direction_sign × (asset_return − benchmark_return)`, SPY and
  sector-ETF comparison, and the `attribution_by_{event,archetype,branch,symbol}.csv` files. Today only
  the per-trade `trade_alpha.csv` scaffold exists.

**Milestone 2 items (coded but treat as not-enabled/not-validated):**
- Theme / country caps, liquidity/ADV cap as enforced policy.
- Drawdown-aware scaling + kill-switch validation.
- Polymarket volume-gate enforcement (flip `enforce_polymarket_volume_gate` on, with tests).
- `paper_trade` routing policy.

**Out of scope for now (future workstreams, do not start):**
- PortfolioMonitor / dynamic de-risk of open positions (breaks the two-pass invariant).
- Joint portfolio construction; factor/beta/correlation models.
- Confidence-weighted sizing (gated on a learning loop proving confidence→alpha).
- Learning loop; execution adapter; PostgreSQL persistence of portfolio state; slippage/borrow fees.

**Known limitation to revisit before relevant:** the open-positions map is keyed by
`(event_id, symbol)`; it cannot hold two same-key positions. Harmless while the duplicate guard is on
(default); must be addressed before any mode allows duplicate `(event, symbol)` positions.

---

## 4. Docs state

- **`PORTFOLIO_LAYER_PLAN.md`** — the **spec / target architecture** (Option 2 Core, build order, §23
  acceptance tests). Accurate as *intent*; describes what the layer should be, not its current
  verification state. Lives in the planning workspace, not necessarily in this repo tree.
- **`portfolio.md`** — the partner's technical handoff. Useful for module-by-module detail, but
  **partially outdated**: it predates the review fixes and still presents the pre-fix state (it does not
  reflect the replay-ordering, close-ownership, reporting-import, or passthrough-profile fixes, nor the
  `enforce_polymarket_volume_gate` flag). Read it for structure, not for current correctness/status.
- **`PORTFOLIO_LAYER_STATUS.md`** (this file) — the **current source of truth for status**: what the
  code does today, what is gated, what still needs DB verification, and what remains to build. When the
  spec (`PLAN`) and the handoff (`portfolio.md`) disagree with this file, **this file reflects the code.**

**Recommendation:** keep all three. Use `PLAN` for target intent, `portfolio.md` for module detail, and
this file for live status. Update this file as the Part 2 verifications pass.

---

*Last updated against `master` commit `6911641`. Update when DB verification (Part 2) progresses or
scope changes.*
