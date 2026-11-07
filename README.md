# Point-in-Time Multi-Agent Volatility Forecasting

This repository contains the prototype for a multi-agent research system that forecasts
which U.S. stocks will experience the largest intraday price ranges in the forthcoming
regular trading session. At 9:00 a.m. New York time, the system ranks an eligible stock
universe using only information available before that cutoff and publishes a top-10
watchlist.

The research question is whether verified multi-agent research adds forecasting value over
a frozen quantitative market control. The market model is therefore a no-agent baseline,
not the main contribution.

## System design

The prototype separates the research process into explicit roles:

1. **Market control (M)** builds the initial ranking from historical price, volatility, and
   volume features.
2. **Orchestrator** proposes dated research questions for candidate stocks.
3. **Allocator** prioritizes questions that may affect the top-10 decision.
4. **Research workers** search the frozen company and macro evidence corpus.
5. **Independent verifier** checks entity identity, publication time, exact quotations, and
   whether the evidence supports the claimed volatility mechanism.
6. **Forecast revision** may update a stock only when it cites accepted evidence. Live
   revisions are code-bounded to 20% above or below M's forecast.
7. **Fail-closed execution** retains M whenever planning, retrieval, validation, or revision
   fails.

The target is the regular-session range `(high - low) / low`. It measures realized
intraday volatility, not price direction, return, or trading profit.

## Prediction designs

- **Prediction A — verified ranking correction:** verified research supplies a signed
  correction to M's unitless ranking score. A learned residual model is used only after
  chronological validation; while it is disabled, a bounded direct rule maps verified
  assessments to score adjustments.
- **Prediction B — daily direct revision:** the multi-agent system researches current
  premarket events and directly adjusts M's next-session range forecasts after independent
  verification.

Both designs are evaluated by paired comparison with the same frozen M control.

## Run the bundled prototype

Requirements:

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)

Install the project and run the test suite:

```bash
git clone https://github.com/zewenli98/Stanford-CS329Z-project-Financial_Strategy.git
cd Stanford-CS329Z-project-Financial_Strategy
uv sync --extra dev
uv run pytest
```

Run the bundled point-in-time multi-agent prototype. This configuration uses the checked-in
integration fixture and recorded model responses, so it requires no API key:

```bash
uv run financial-strategy screen --config configs/adaptive_range_v1.yaml
```

The command prints a run ID and publishes an immutable run under `runs/`. Inspect:

- `watchlist.json` for the top-10 selection;
- `range_predictions.json` for baseline and revised range forecasts;
- `research_questions.jsonl` and `research_cards.json` for agent research;
- `rejected_cards.json` for failed verification;
- `tool_traces.jsonl` and `model_calls.jsonl` for the audit trail;
- `manifest.json` and `manifest.json.sha256` for artifact integrity.

Replay the published run without making a live provider call:

```bash
uv run financial-strategy replay RUN_ID
```

Replace `RUN_ID` with the identifier printed by the screening command.

## One-day live demo on Windows

Run both prediction heads with fresh OpenRouter model calls for one session from the
frozen historical archive. First run `uv sync --extra dev` and restore the local archive
listed below under **Run the 100-stock research experiment**; the full demo dataset is not
bundled with GitHub. Run these commands from the repository root:

```powershell
.\scripts\run_live_demo.ps1 -DryRun
.\scripts\run_live_demo.ps1
.\scripts\run_live_demo.ps1 -Date 2026-08-27 -RequireDifferences
.\scripts\run_live_demo.ps1 -Head A -Date 2026-08-27
.\scripts\run_live_demo.ps1 -Head B -Date 2026-08-27
```

If Windows blocks direct script execution, use this form for a fresh live run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_live_demo.ps1" -RequireDifferences
```

Append `-Date 2026-07-31` to choose another archived session. Each invocation starts fresh
model calls. To rebuild the final comparison from an existing saved run without inference:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_live_demo.ps1" -ReportRunId YOUR_RUN_ID
```

Replace `YOUR_RUN_ID` with the launch ID printed by your earlier run. The ground-truth
column appears automatically in normal runs, regardless of `-RequireDifferences`.

The default session is **2026-08-27**, with **three Head B batches** and both heads.
Put `OPENROUTER_API_KEY` in an untracked `.env` in this checkout or the shared main
repository, or pass `-EnvFile C:\path\to\.env` explicitly. Existing environment
credentials take precedence. Credential files are read without copying their contents.
The script uses this checkout's `.venv` and `src` code, enables UTF-8, and prepares the 100-stock
subset when needed. Preparation leaves source data and tracked configurations unchanged
and skips outcome labels. `-DryRun` checks input coverage and prints the no-agent baseline
without a provider call or an API key.

`-Head Both` is the default. Head A runs the existing company/macro research,
verification, and verified-correction ranking workflow. Head B runs adaptive research
and direct range revision. They share the session, cutoff, data, and frozen market weights.
The learned ridge model retains its configured promotion status. When it is disabled,
the bounded direct correction uses verified non-neutral assessments. Distinct filing
exhibits and retrieved body chunks are preserved. Exact, unique claim/event references
are normalized to their existing document IDs before deterministic and semantic
verification, so a separate citation-repair script is not needed. A's unitless scores
are displayed separately from B's percentage-range forecasts.

The default budget settings are $0.80 for A and $0.80 for B. `-HeadABudgetUsd` controls
A's existing post-response usage guard; it is not a strict pre-dispatch spending cap.
`-BudgetUsd` controls B's reserved, configured-price research cap, calculated from the dated
model prices in `configs/adaptive_range_100_compact_live.yaml`. Adjust research coverage
with `-MaxBatches`; its default is three. This runs live inference on an archived session; it does
not fetch today's prices or news. Dates without complete historical inputs stop before
live inference. A three-minute recording may need edits around model waiting time.

Each invocation uses separate response caches for A and B and a new run ID. It prints
both top tens, A's market score/residual/final score and verification counts, B's baseline
and revised ranges, and a side-by-side M/A/B/ground-truth membership table.
Ground truth ranks the same full prediction universe by realized session (high-low)/low,
with actual ranges shown as percentages. Target-session bars are opened only after
published predictions pass integrity checks. Missing high/low coverage is reported
explicitly rather than displaying a partial universe as the true top ten. A's cost display covers
successful responses with reported usage; failed/unreported usage may also be billed.
Full audit files stay under `runs/`. The launch configurations, per-head status and
summary, and `comparison.csv` stay under `runs/demo_launches/`. The CSV includes A's market
score and correction and B's baseline and forecast ranges. `summary.json` records changed
forecasts and whether all three top-ten orders differ. A's run ID has an `a_`
prefix; B retains the launch run ID. Each head has its own replay command. A failed head
is labeled and leaves its comparison column empty, while the other head can still finish.
Reusing a supplied `-RunId` is refused.
To add ground truth to an existing saved presentation without paid inference, run
`.\scripts\run_live_demo.ps1 -ReportRunId YOUR_RUN_ID`. This refreshes only
`comparison.csv` and `ground_truth.json`; published predictions and the original
summary stay unchanged. New full runs also include ground truth in `summary.json`.
To rehearse without paid inference, use `-Offline`. Baseline fallback is explicitly
identified if no live provider response completes.

`-RequireDifferences` validates the demonstration without changing model outputs. It exits
with code 3 if completed live research does not produce three distinct M/A/B orders;
the audit is retained. Live model judgments can vary, so August 27's earlier ranking is
not hardcoded or guaranteed. An unchanged result can still be a valid research outcome.

## Run the 100-stock research experiment

The full historical study uses local datasets that are intentionally excluded from GitHub.
The expected local paths are:

```text
data/2023-11-01_2026-10-30/
data/research100_v1/
```

Given the parent archive, create the fixed 100-stock subset:

```bash
uv run python scripts/prepare_research100.py --compact
```

For live Prediction B, create an untracked `.env` file:

```dotenv
OPENROUTER_API_KEY=your-key
```

Then run the matched 30-session comparison between M and M plus the full multi-agent direct
revision workflow:

```bash
uv run python scripts/run_live_range_study.py \
  configs/adaptive_range_first30_live_evaluation.yaml \
  --receipt runs/prediction_b_first30/receipt.json
```

The runner refuses to overwrite an existing receipt to prevent accidental paid reruns.
Provider, model, concurrency, point-in-time inputs, frozen M weights, and the 20% revision
bound are defined in `configs/adaptive_range_100_compact_live.yaml`.

## Evaluation

The primary ranking metric is NDCG@10. The report also includes:

- mean realized range among the selected top 10;
- precision@10 for stocks whose realized range exceeds 4%;
- mean absolute forecast error;
- top-k overlap and regret;
- accepted and rejected evidence cards;
- changed forecasts and final forecast revisions;
- per-session paired differences between the multi-agent arm and M.

Labels are opened only after prediction artifacts are published and hash-verified. Missing
high/low labels are reported rather than replaced with another target.

## Repository layout

```text
src/financial_strategy/   Core gateway, ranking, agent, verification, and evaluation code
configs/                  Maintained prototype and research configurations
scripts/                  Data preparation, training, live evaluation, and audit entry points
tests/                    Unit tests plus a self-contained integration fixture
docs/                     Detailed adaptive workflow documentation
reports/                  ICLR-style project report and bibliography
```

## Reproducibility and safety

- `.env`, `.env.*`, `data/`, `runs/`, and `cache/` are ignored and must remain local.
- The gateway admits only market bars and evidence available before the daily cutoff.
- The evaluator opens target-session highs and lows only after prediction publication.
- Every accepted claim is tied to a source span and verification decision.
- Failed agent operations fall back to the quantitative control.
- The software is a research prototype. It does not place trades or estimate portfolio
  profitability.

For implementation details, see [the adaptive workflow guide](docs/adaptive-range-v1.md).
