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

The expanded snapshot contains 752 XNYS sessions split chronologically into 676 train and
76 validation sessions. Parameter search reuses frozen Research Cards and never calls the
model once per parameter trial:

```bash
uv run financial-strategy prepare-tuning --config configs/tuning.yaml --limit 5
# Remove --limit after the cost/failure smoke test succeeds.
uv run financial-strategy prepare-tuning --config configs/tuning.yaml
uv run financial-strategy tune --config configs/tuning.yaml
```

The tuning command optimizes NDCG@10 on train only. It writes an immutable candidate before
loading validation labels, then produces a one-shot validation comparison against the original
configuration, four deterministic baselines, and three ablations. The candidate is promoted to
the configured `screening_tuned` path only when both paired block-bootstrap acceptance tests
pass.

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

## LLM residual v2

The v2 research path keeps the promoted market ranker frozen and permits the LLM to
contribute only a bounded, signed residual. SEC document text is chunked and hashed;
numeric fields must quote an exact chunk, and both deterministic and semantic verification
must pass before the residual can be nonzero.

```bash
# Build a document-complete snapshot for every SEC evidence record.
uv run python -m scripts.enrich_evidence \
  --all-documents \
  --preserve-macro \
  --resume-documents cache/sec_documents

# Deterministically assign the canonical multi-label company event taxonomy.
uv run python -m scripts.classify_company_events

# Live calls are written to both the atomic request cache and immutable recordings.
uv run python -m scripts.generate_llm_artifacts \
  --mode live --maximum-cards 200

# Rebuild the same cards without network model calls, then fit/evaluate the residual.
uv run python -m scripts.generate_llm_artifacts \
  --mode recorded --maximum-cards 200
uv run python -m scripts.train_llm_residual
```

The trainer opens labels only after verifying the recorded artifact manifest. It reports
market-only, company, macro, and full residual variants. If the paired NDCG confidence
interval does not clear the promotion gate, the generated candidate keeps residual scoring
disabled and the LLM remains explanation-only.

## Agentic ranking v4 experiment

The v4 path binds every artifact bundle to the dataset manifest, chronological split,
prompt versions, and response schemas. It also attaches point-in-time macro interactions,
uses the same percentile implementation as the market champion, and tunes residual
shrinkage and scale only inside nested chronological cross-validation.

```bash
uv run python -m scripts.generate_llm_artifacts \
  --mode live --maximum-sessions 40 \
  --maximum-model-calls 4000 \
  --maximum-input-tokens 40000000 \
  --maximum-output-tokens 5000000 \
  --maximum-cost-usd 25 \
  --output-directory runs/agentic_v4_train40_prompt4

uv run python -m scripts.generate_llm_artifacts \
  --mode recorded --maximum-sessions 40 \
  --maximum-model-calls 4000 \
  --output-directory runs/agentic_v4_train40_prompt4

uv run python -m scripts.train_llm_residual \
  --artifacts runs/agentic_v4_train40_prompt4/artifacts_recorded.json
```

On the current frozen snapshot, live and recorded artifacts match exactly. The nested-CV
mean NDCG@10 delta is positive, but its confidence interval still crosses zero and the fold
stability gates fail. The candidate therefore remains disabled and the 76-session validation
holdout has not been consumed.

The follow-up 120-session experiment is stored under `runs/agentic_v5_train120`.
Event citations are normalized against the deterministic taxonomy before verification,
bounded catalyst forecasts are treated as advisory inferences while factual failures remain
blocking, and cards with an unknown impact bucket retain their other verified features.
This increased eligible, encodable training rows to 1,387 across all 120 sampled sessions.
The nested-CV NDCG@10 delta remained statistically indistinguishable from zero, so residual
scoring and validation consumption remain disabled.

The calibrated follow-up is stored under `runs/agentic_v6_train120_calibrated`. Before
fitting the residual, each session's frozen market scores are converted to cross-sectional
percentiles without using labels; production scoring applies the same transform. This
preserves the market ordering while putting the baseline and realized percentile target on
the same scale. Nested CV selected residual scale zero in every fold, so the calibrated
candidate also remains disabled and the validation holdout remains untouched.

The direction-semantics correction is evaluated under
`runs/agentic_v7_train120_direction_fixed`. It prevents negative price-direction labels from
demoting an absolute-movement catalyst and updates the synthesis prompt to keep return
direction separate from absolute-movement rank correction. This corrected 98 legacy cards,
but nested CV still selected residual scale zero in all 32 folds; the candidate remains
disabled.

## Close-to-close experiment

The independent close-to-close pipeline uses split-adjusted absolute previous-close to
current-close labels while retaining the frozen 20-feature schema. Its market model is stored
under `runs/expanded_training_c2c`; nested CV promoted the train-only winner with an NDCG@10
delta of `+0.024543` and a 95% interval of `[0.019684, 0.029289]`. The resulting Top-50
candidate set is stored under `runs/expanded_agent_c2c`.

The 120-session live and recorded Agent bundles under `runs/agentic_c2c_train120` contain
2,112 cards and have identical artifact hashes. Live generation stayed within the approved
budget at an estimated `$28.76`. On 112 outer-test sessions, market NDCG@10 was `0.436746`
and Agent NDCG@10 was `0.434675`, a delta of `-0.002070` with 95% interval
`[-0.004141, 0.0]`. The residual candidate therefore remains disabled; the 76-session
validation holdout was not consumed.

The same-day feature ablation under `runs/agentic_c2c_same_day_features` retains only
`before_open` and `during_session` cards and adds event-family, release/schedule proximity,
and matched reported-versus-consensus surprise features. This reduces the training set from
1,391 to 174 rows. Across 102 outer-test sessions, the paired NDCG@10 delta improves to
`+0.002692`, but its 95% interval `[-0.003178, 0.010909]` still crosses zero and only 9 of
29 folds win. Numeric surprise is unavailable in this snapshot because no same-day card has
a matched reported/consensus metric pair. The candidate remains disabled.

The shared expanded snapshot is `data/2023-11-01_2026-10-30`. Ranking and adaptive range
both use its market bars, company filings, and macro records. FRED daily DGS2, DGS10, and
VIXCLS observations are market-state inputs on that same snapshot; they are not additional
CPI or employment releases. Residual scoring stays disabled unless a later promotion gate
passes. See [the architecture experiment guide](docs/architecture-experiments.md) for the
100-stock range study. Keep `.env`, `data/`, `runs/`, and `cache/` local.
