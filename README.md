# Financial Strategy

Point-in-time research workflows for ranking US common stocks. Both the ranking workflow
and the opt-in `adaptive_range_v1` workflow forecast the regular-session range,
`(high - low) / low`. Ranking outputs an order; adaptive range also outputs the range
itself. They share frozen inputs, structured research, verification, and replayable audit
output. That range is the only rank label. Open-to-close magnitude remains a market
feature named `median_abs_oc_return_*`.

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

Both the fixed ranking workflow and `adaptive_range_v1` read this same snapshot. The
workflow differs only in the configuration: ranking configs omit `workflow`, while range
configs set `workflow: adaptive_range_v1`.

```bash
uv run python scripts/fetch_real_data.py
uv run financial-strategy screen --config configs/screening_2026-10-14.yaml
uv run financial-strategy screen --config configs/adaptive_range_v1.yaml
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

`tests/fixtures/integration/provenance.json` records source parameters, retrieval times, known
availability assumptions, package versions, and hashes. Yahoo history is explicitly treated
as split-adjusted and is never adjusted a second time by the gateway. Yahoo is an unofficial
source and can revise historical data, while GDELT is used only for discovery.

## Commands

- `screen`: create a new immutable screening run.
- `replay`: verify and deterministically rebuild an existing successful run.
- `evaluate`: create new historical screening tasks, freeze predictions, then load labels.
- `prepare-range-labels`: derive adaptive range labels and report coverage from frozen high/low.
- `evaluate-retrieval`: score company, event, and sufficiency judgments separately from rank.
- `prepare-tuning`: create one immutable Agent artifact run per train/validation session.
- `tune`: search deterministic ranking parameters on train, freeze the winner, and consume
  the chronological validation holdout once.

## Train/validation tuning

The study universe is the 100-stock subset `data/research100_v1`. It keeps the parent
archive's company filings and the expanded macro file. A session is one regular New York
trading day. The 752 sessions are split into 676 training days and a final 76-day test.
Training ranks `(high - low) / low` and does not score those 76 days:

```bash
uv run python -m scripts.train_expanded
```

After that report is frozen, score the 76 days once without refitting:

```bash
uv run python -m scripts.train_expanded --holdout
```

Do not run `financial-strategy tune` before that holdout command. `tune` also reads the
same 76 sessions and records them as consumed.

Historical macro rows from current-view calendar endpoints use their verified observation time,
while explicitly labeling historical availability as an assumption. The expanded fixture covers
2023-07-01 through 2026-10-30 and contains 144 released observations plus 27 scheduled FOMC
records. Research replay assumes actual is available at `event_time` and consensus one second
earlier; `macro/audit.json` records that these timestamps are assumptions, not archived-source
proof. Four rows absent from Nasdaq were filled from the TradingView economic calendar API and
are counted separately in the audit. The earlier v2 Agent artifacts and `train_validation_v1.json` report
remain invalidated audit history; new tuning uses the v3 artifact index in `configs/tuning.yaml`.

The project is research software. It does not place trades and its heuristic rank score is
not a probability.

## Current study

Ranking and adaptive range both forecast `(high - low) / low` for the same 100 stocks.
The parent archive `data/2023-11-01_2026-10-30` supplies company filings and the expanded
macro records, including CPI, payrolls, unemployment, FOMC, DGS2, DGS10, and VIX. Residual
scoring stays disabled, so the published ranking order is the market score. LLM research
is kept for audit and process scores. `final_weights.company_event` and
`final_weights.macro` stay at zero. Earlier open-to-close, close-to-close, and 200-stock
candidate configs have been removed. Keep `.env`, `data/`, `runs/`, and `cache/` local.
