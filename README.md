# Financial Strategy

Point-in-time research workflows for ranking US common stocks. The original workflow ranks
expected absolute open-to-close movement. The opt-in `adaptive_range_v1` workflow forecasts
the regular-session relative range, `(high-low)/low`, using bounded adaptive research.
Both preserve frozen inputs, structured research, verification, and replayable audit output.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

## Adaptive daily-range experiment

The adaptive example runs in **scripted mode by default**. It exercises an orchestrator,
general research workers, local tools, card verification, allocation, and dollar accounting
without calling a model provider. Its scripted forecaster deliberately preserves the
historical-range baseline; it is an integration fixture, not a financial forecasting model.

```bash
uv run financial-strategy screen --config configs/adaptive_range_v1.yaml
uv run financial-strategy replay RUN_ID
uv run financial-strategy prepare-range-labels --config configs/adaptive_range_evaluation.yaml
uv run financial-strategy evaluate --config configs/adaptive_range_evaluation.yaml
```

Replace `RUN_ID` with the screening command's published run ID. On Windows, an already
synchronized environment can run `.venv\Scripts\financial-strategy.exe` in place of
`uv run financial-strategy`.

The screening output contains expected ranges for the supplied eligible stocks and a
configurable top-`k` watchlist. For example, `predicted_range: 0.03` means an expected 3%
regular-session range, not a 3% directional return or a 3% probability.

Default research limits are two concurrent workers, eight batches, and a $2 normalized
uncached model-spend cap per run. Live inference requires manually supplied, dated prices
for every exact model ID; the example's offline reference prices are not provider quotes.
The existing snapshot supplies high/low labels for nine of the ten configured October
sessions. October 14 remains screening-only until genuine target-day high/low are supplied.
Missing labels are reported rather than replaced with the legacy open/close labels.

See [the adaptive workflow guide](docs/adaptive-range-v1.md) for the allocation derivation,
configuration, budgets, masking limits, memory, replay, and comparison methodology.

For the five-session live-model pilot, use
`configs/adaptive_range_live_five_day_evaluation.yaml` with
`scripts/run_live_range_study.py`. The evaluation retains a top-10 watchlist and separately
scores its top-3 and top-5 prefixes against the full eligible universe. It writes per-day
costs, metrics, watchlists, and immutable model/tool audits. This historical pilot uses paid
inference and an aggregate cap; see the workflow guide before starting a new run.

## Real-data snapshot

The report builders produce local offline briefs, daily cost records, and top-3/5/10
metrics. Generated results, datasets, source extracts, caches, and credentials are
excluded from this code package. These historical studies are descriptive results,
not evidence of statistical superiority.

The 2026-10-01 through 2026-10-14 snapshot freezes real inputs for 20 stocks from Yahoo
Finance, SEC EDGAR, verified company news pages, BLS, and pre-release Dow Jones consensus
reports from CNBC. The model only receives evidence that passed the point-in-time gateway.

```bash
uv run python scripts/fetch_real_data.py
uv run financial-strategy screen --config configs/screening_2026-10-14.yaml
uv run financial-strategy evaluate --config configs/evaluation_2026-10-01_2026-10-14.yaml
```

The original real-data screening example uses live OpenRouter structured output.
Configure the shared model with `agents.model`
(the real-data example uses `openai/gpt-5.6-luna`) and put the credential in an untracked
`.env` file:

```dotenv
OPENROUTER_API_KEY=your-key
```

Each role inherits `agents.model` by default. Set `agents.roles.<role>.model` to give a
specific role a different model.

Live runs remain bounded by per-symbol and task-level model/tool call limits. The screening
configuration uses `top_k=10`; the ten-session evaluation is an integration example, not
evidence of strategy performance.

`data/2026-10-01_2026-10-14/provenance.json` records source parameters, retrieval times, known
availability assumptions, package versions, and hashes. Yahoo history is explicitly treated
as split-adjusted and is never adjusted a second time by the gateway. Yahoo is an unofficial
source and can revise historical data, while GDELT is used only for discovery.

## Commands

- `screen`: create a new immutable screening run.
- `replay`: verify and deterministically rebuild an existing successful run.
- `evaluate`: create new historical screening tasks, freeze predictions, then load labels.
- `prepare-range-labels`: derive adaptive range labels and report coverage from frozen high/low.

The project is research software and does not place trades. The original workflow's
heuristic score is not a probability. Adaptive numerical forecasts and allocation scenarios
are uncalibrated experimental estimates; the bundled fixture cannot establish superiority
over the historical baseline or simpler research policies.

## Reproduce the 100-stock architecture experiments

See [the architecture experiment guide](docs/architecture-experiments.md) for local
data preparation, deterministic checks, optional paid comparisons, and audit records.
The package includes source, tests, configurations, and reproducibility scripts.
Supply the data separately; keep `.env`, `data/`, `runs/`, and `cache/` local.
