# Trading Bot Skeleton TODO

This file combines:
- Previous strategy/signal feedback.
- Full-project gap audit (Python + C++ + integration layer).

## Phase 0: Runnable Base (do first)

- [ ] Add an app entrypoint (for example `main.py`) that wires: config -> DB -> data feed -> strategies -> portfolio -> IB execution.
- [ ] Add dependency management (`requirements.txt` or `pyproject.toml`) including `pandas`, `ib_insync`, `yfinance`, `talib` (or alternative), DB driver, and C++ build deps.
- [ ] Add environment/config loader for secrets and runtime settings (`IB host/port/client_id`, DB URL, Gemini key, trade mode, universe list).
- [ ] Implement all current `pass` methods in:
  - [ ] `agents.py`
  - [ ] `API.py`
  - [ ] `Data_Feed.py`
  - [ ] `DB.py`
  - [ ] `IB.py`
  - [ ] `Portfolio.py`
  - [ ] `Wrapper.py`
  - [ ] `strategies/cointegration/StatArbStrategy.py`
- [ ] Define one shared signal schema (dataclass/Pydantic/dict contract) used by all strategies and portfolio methods.
- [ ] Define one shared order schema (side, qty, type, tif, strategy, metadata).
- [ ] Implement `Portfolio.run_cycle()` end-to-end:
  - [ ] load market data
  - [ ] load current positions
  - [ ] gather strategy signals
  - [ ] apply risk gate
  - [ ] allocate capital/size
  - [ ] place orders
  - [ ] persist signals and executions

## Strategy and Signal Issues (existing findings, expanded)

- [ ] `MeanReversionMomentum` currently acts mostly like MACD cross logic; complete actual mean-reversion rules.
- [ ] `MeanReversionMomentum._get_sell_signal()` is empty; implement exits.
- [ ] Indicators are calculated (RSI/ATR/Bollinger) but mostly unused in decision logic.
- [ ] Add explicit anti-lookahead convention (signal on close, execute next session/bar).
- [ ] Signal payload is too thin; include at least timestamp, strategy, symbol(s), side, rationale/features, and unique id.
- [ ] Keep signal generation separate from position sizing/execution details.
- [ ] For stat-arb, emit pair-leg signals (long/short pair) instead of single-symbol BUY/SELL only.
- [ ] Add de-duplication/conflict handling when multiple strategies signal the same symbol.

## Python Code-Level Gaps / Bugs

- [ ] `IB.py` uses `IB()` without importing it (likely `from ib_insync import IB`).
- [ ] `Strategy.py` type hint uses `pd.dataFrame` instead of `pd.DataFrame`.
- [ ] `Portfolio.py` has a typo in `gather_signals` list flattening variable name (`siganl`); not fatal but fix.
- [ ] `Data_Feed.fetch_historical_data()` return type likely mismatches multi-ticker usage (consider `Dict[str, pd.DataFrame]`).
- [ ] Standardize imports and naming conventions (mixed casing: `Data_Feed`, `IB_Connect`, `agents`, etc.).
- [ ] Add structured logging instead of silent failures/placeholders.
- [ ] Add clear error handling/retries around external IO (Yahoo, IB, DB).

## C++ Cointegration Engine Gaps / Bugs

- [ ] Add missing includes and full implementation in `strategies/cointegration/binding.cpp`.
- [ ] Bind `CointegratedPair` and `run_cpp_scan` in `PYBIND11_MODULE`.
- [ ] `MarketData.cpp`: fix invalid signature `void::MarketData::set_price(...)`.
- [ ] `PairScanner.cpp`: review thread partitioning logic (possible off-by-one and row skip/duplication).
- [ ] `PairScanner.cpp`: join using actual thread count (`threads.size()`), not raw `num_threads`.
- [ ] `PairScanner.cpp`: clear `top_pairs` before each scan to avoid stale accumulation across runs.
- [ ] `MathStats.cpp`: guard against divide-by-zero in correlation/OLS paths.
- [ ] Add input validation and bounds checks in `MarketData` accessors.
- [ ] Add build tooling for extension module (CMake + pybind11 or setuptools build config).
- [ ] Add Python-side import/loading path for compiled `cointegration_engine`.

## Data + DB Skeleton Missing Pieces

- [ ] Define DB schema for:
  - [ ] OHLCV bars
  - [ ] signals
  - [ ] orders
  - [ ] fills/executions
  - [ ] daily account snapshots
- [ ] Implement initial historical load job (2y bulk universe).
- [ ] Implement daily incremental updater job (append latest bar).
- [ ] Add idempotent upsert behavior for data ingestion.
- [ ] Add a universe table/config and one place to manage tradable symbols.
- [ ] Add query helpers for strategy windows (for example last N bars per ticker).

## IB Execution Skeleton Missing Pieces

- [ ] Implement connect/disconnect lifecycle and reconnect behavior.
- [ ] Implement account summary and positions fetch mapping to internal schema.
- [ ] Implement market and bracket order placement with returned order ids.
- [ ] Implement order status polling/callback handling.
- [ ] Persist order and fill lifecycle events in DB.

## API + Agents Skeleton Missing Pieces

- [ ] Define API command surface (what commands are accepted and expected outputs).
- [ ] Implement API handlers for reading signals/trades/performance snapshots.
- [ ] Define agent responsibilities and contracts:
  - [ ] trade-insight agent inputs/outputs
  - [ ] accountant/risk agent inputs/outputs
- [ ] Add Gemini client wrapper and prompt templates (once key is added).
- [ ] Add guardrails so agents cannot trigger live actions unintentionally.

## Minimal Quality Baseline

- [ ] Add `README.md` with setup/run flow.
- [ ] Add simple tests for:
  - [ ] indicator/signal generation
  - [ ] portfolio sizing logic
  - [ ] DB read/write adapters
  - [ ] C++ scanner smoke test
- [ ] Add lint/format config and a single command to run checks.
- [ ] Add one dry-run mode (`no order placement`) for safe daily validation.

## Suggested Build Order (practical)

- [ ] 1) Shared schemas + config + dependency file.
- [ ] 2) Data_Feed + DB ingestion (bulk + daily).
- [ ] 3) Portfolio run cycle + mean reversion strategy complete.
- [ ] 4) IB paper execution integration.
- [ ] 5) Cointegration wrapper + C++ module integration.
- [ ] 6) API read layer + agents integration.
- [ ] 7) Tests and basic observability.
