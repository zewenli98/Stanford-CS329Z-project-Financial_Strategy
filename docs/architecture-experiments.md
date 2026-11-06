# Reproducing the 100-stock architecture experiments

This package contains the adaptive workflow, tests, configuration examples, and experiment
scripts. It does not add datasets, credentials, model recordings, source documents, caches,
generated watchlists, or run artifacts. Data already tracked in the repository's base
history is unaffected. The source distribution also excludes datasets and local outputs.

## Install and prepare local inputs

Use Python 3.12 or later and install the project with its development dependencies:

```bash
uv sync --extra dev
```

If using another environment manager, install the editable project with `.[dev]`.
Keep provider credentials in a local `.env` file; the code reads `OPENROUTER_API_KEY`
for paid OpenRouter calls. Never add that file to version control.

The preparation script expects the separately supplied dataset at
`data/2023-11-01_2026-10-30/`, with universe, market, company, and macro subdirectories.
It writes the derived dataset to `data/research100_v1/` without modifying source files:

```bash
python scripts/prepare_research100.py --compact
```

The fixed universe uses the 100 highest median dollar-volume stocks over the first
60 pre-evaluation sessions. Price values remain unchanged. The script checks every
recorded split before declaring the already-adjusted vendor series split-adjusted.
Study runs read the expanded macro archive, including CPI, payrolls, unemployment, FOMC, and the added DGS2, DGS10, and VIX series. The research subset's own macro file only keeps high-confidence records and is empty.

Run deterministic checks before using a provider:

```bash
python -m pytest
financial-strategy screen --config configs/adaptive_range_100_compact_offline.yaml
```

Integration tests require the historical snapshots referenced by their fixtures. A code-only
installation without those snapshots cannot run the full integration suite. The `baseline`
policy and scripted examples make no paid model requests.

## Experimental variants

All variants forecast the complete eligible universe and retain a ten-name watchlist.
The 100-stock configurations bound expensive research to 20 candidates. The packet
candidate runs two workers over four batches, for up to eight research assignments.

| Option | Behavior |
|---|---|
| `research_protocol: legacy` | Original planner and tool-using researchers |
| `research_protocol: guided` | Explicit tool contracts, substantive default reads, repair slot |
| `research_protocol: packet` | Deterministic dated source passages before model extraction |
| `candidate_method: hybrid` | Mix historical volatility and recent-event candidates |
| `baseline_method: ewma10` | Recency-weighted historical range baseline |
| `evidence_previous_day_only: true` | Publication and availability strictly before target-day New York midnight |

The optional `configs/adaptive_range_lab_candidate.yaml` uses Luna for extraction and
Terra for verification and forecasting. It retains the original mean-20 history baseline.
`configs/adaptive_range_lab_ewma.yaml` is the inference-free weighted-history comparator.
These are opt-in configurations; the stronger model is not promoted to the default.
Their saved task dates and prices describe the historical experiment: deliberately update
the task and verify current provider prices before a new paid run. A per-run cap is not
an aggregate authorization for an arbitrary comparison grid.

## Registered study and local records

The lab scripts use `runs/architecture_lab_20261106/`. The setup step refuses to overwrite
an existing plan. It freezes development/holdout dates, the initial source snapshot, and
provider model prices. It contacts the model-catalog endpoint but does not run inference:

```bash
python scripts/setup_architecture_lab.py
python scripts/snapshot_architecture_runtime.py
python scripts/train_range_baselines.py
python scripts/run_architecture_lab.py --stage smoke
```

The remaining stages make paid requests and share a durable $10 reservation ledger:

```bash
python scripts/run_architecture_lab.py --stage dev
python scripts/run_architecture_lab.py --stage strong
python scripts/run_architecture_lab.py --stage holdout
python scripts/run_architecture_lab.py --stage confirm
```

Run stages sequentially. The ledger protects parallel jobs within one controller process;
do not launch two controllers against the same ledger. The plan sets a 55-minute launch
window, and jobs reserve their full caps before dispatch. Existing registered jobs are
skipped. The holdout stage freezes the best complete development agent arm before scoring
holdout dates. Do not rerun selection after viewing holdout results.

Local records contain complete configurations, frozen inputs, exact source reads, research
questions, accepted/rejected cards, forecast revisions, model attempts, cost settlements,
and post-publication evaluation metrics. The report builder can summarize a fresh lab:

```bash
python scripts/report_architecture_lab.py
```

The finalization and audit scripts additionally expect the original study's source hashes,
runtime snapshot, final test XML, and post-audit change records. They are review/reproduction
utilities for that completed study, not a single-command launcher for a new experiment.
For an individual run, use `financial-strategy replay RUN_ID` with its local artifacts.
Recorded replay makes no paid requests; it is not an independent live replicate.

## Recorded assessment

The completed study had 52 successful daily outputs and $2.8381104 in recorded charges,
including conservative bounds for failed requests. Three dates were used for development;
six other dates were reserved for evaluation. On those six dates, original-agent NDCG@10
was 0.69632 versus 0.69513 for the stronger-model candidate, with 35% top-10 overlap for
both. Candidate research acceptance improved from 3/36 to 28/48 assignments, but this
did not establish a ranking improvement. The full local report and raw outputs remain
outside version control.

The final full suite passed 154 tests. The audit checked all 52 daily outputs and exact
recorded replays of the first and last candidate holdout dates. A replay-only DST defect
was corrected by preserving named cutoff timezones in new snapshot metadata and restoring
older New York runs from their stored TTL-clamped expiries. Saved live results were unchanged.

Known limitations include six holdout sessions, one live trajectory per setup, current-
constituent survivorship bias, vendor history revisions, incomplete identity masking,
uncalibrated event effects, and one synthesis input-limit fallback. Timestamp and quotation
checks validate the recorded information boundary, not every archive fact or impact inference.
