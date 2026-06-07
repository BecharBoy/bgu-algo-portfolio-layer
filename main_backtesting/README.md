# Full Resumable Historical Backtest

The historical backtest covers January 1, 2022 through June 1, 2026 and preserves
all source data, reusable downloads, model results, runs, reports, and graphs.

The engine is staged and resumable. A real external-service or validation failure
records the exact failed work item and stops the stage. It does not silently skip,
retry, split batches, or fall back to individual requests.

Article research is temporarily excluded from the active pipeline. The fallback branch
uses the first observed Polymarket crossing above 55%, positive asset ROC momentum,
and completed-hour Polymarket volume confirmation when enough volume history exists.
A completed run verifies that every asset candidate reached simulation.
`entry_decisions.csv` explains every opened or blocked trade candidate.

The active stages are:

1. Apply tag and 5-to-60-day duration filters.
2. Batch-filter individual market questions with Ollama. Rejected markets are removed
   from the run and written to `deleted_non_relevant_markets.csv`; canonical source rows
   are preserved.
3. Download hourly Polymarket probability and completed-hour traded volume from each
   surviving market's start through the simulation boundary.
4. Keep only the first 55% crossing and batch-create asset worlds for crossed markets.
5. Download only candidate-specific yfinance windows plus the daily bars required by
   the four-feature ML models.
6. Train walk-forward asset-event models and simulate ML or Polymarket-momentum trades.

## Setup

```powershell
.\.venv\Scripts\pip.exe install -r main_backtesting\requirements.txt
```

PostgreSQL and Ollama use the existing project `.env`.

## Commands

Calibrate fixed Ollama batch sizes once:

```powershell
.\.venv\Scripts\python.exe -m main_backtesting.main calibrate-batches
```

List eligible January 2026 markets without calling Ollama or downloading data:

```powershell
.\.venv\Scripts\python.exe -m main_backtesting.main list-candidates --limit 20
```

Run small tests progressively. Each command creates a resumable, permanently
recorded run limited to one event. Use the printed run ID to advance the same run one
stage at a time:

```powershell
# Create a run and add one Ollama market-filter request
.\.venv\Scripts\python.exe -m main_backtesting.main smoke-test --event-id <event-id> --through event_filter

# Add probability history and pass detection
.\.venv\Scripts\python.exe -m main_backtesting.main resume --run-id <run-id> --through probabilities

# Finally run the remaining stages for the same event
.\.venv\Scripts\python.exe -m main_backtesting.main resume --run-id <run-id>
```

Run the full historical backtest:

```powershell
.\.venv\Scripts\python.exe -m main_backtesting.main run
```

Resume an interrupted run:

```powershell
.\.venv\Scripts\python.exe -m main_backtesting.main resume --run-id <run-id>
```

Regenerate reports or explicitly purge one run:

```powershell
.\.venv\Scripts\python.exe -m main_backtesting.main report --run-id <run-id>
.\.venv\Scripts\python.exe -m main_backtesting.main purge-run --run-id <run-id>
```

Run outputs are stored under `main_backtesting/output/runs/<run-id>/`.
When a stage fails, the CLI prints the saved run ID, failed stage, and work key
before re-raising the error.
