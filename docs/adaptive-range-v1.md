# Adaptive daily-range research, version 1

`adaptive_range_v1` is an opt-in experiment alongside the ranking workflow. Both forecast
the regular-session range, `(high - low) / low`.
At a 9:00 a.m. America/New_York cutoff, it forecasts each eligible stock's same-day
regular-session relative range and returns a configurable top-`k` watchlist. It does not
execute trades. Working execution, bounded cost, evidence provenance, and reproducibility
are the acceptance criteria for this version; a forecasting performance win is not.

## Run the fixture

From the repository root, install the declared environment and run the scripted example:

```bash
uv sync --extra dev
uv run financial-strategy screen --config configs/adaptive_range_v1.yaml
uv run financial-strategy replay RUN_ID
uv run financial-strategy prepare-range-labels --config configs/adaptive_range_evaluation.yaml
uv run financial-strategy evaluate --config configs/adaptive_range_evaluation.yaml
```

Use the screening output's run ID for `RUN_ID`. On Windows, after synchronization, the
equivalent executable is `.venv\Scripts\financial-strategy.exe`. For example:

```powershell
.venv\Scripts\financial-strategy.exe screen --config configs/adaptive_range_v1.yaml
```

The sample uses `agents.mode: scripted`; no model-provider credential is needed. The
scripted responder tests planning, permitted tool use, verification, memory, allocation,
and replay. Its forecasts intentionally preserve the historical baseline. Identical
watchlists across policies in this mode therefore do not establish that research is
ineffective, or that the adaptive allocator is better than another allocator.

## Target, inputs, and label coverage

For a supplied stock `s` and target session `t`, let `H[s,t]` be its realized regular-session
high and `L[s,t] > 0` its realized regular-session low. The realized target is
`R[s,t] = (H[s,t] - L[s,t]) / L[s,t]`, a nonnegative, dimensionless ratio. A high of $103
and low of $100 give `(103 - 100) / 100 = 0.03`, or 3%. This measures the session's range;
it does not establish the order of those prices or an achievable trading return.

Forecasts are finite, nonnegative expected ranges. Initially each forecast is the mean
relative high/low range over the last 20 completed sessions. At least 21 complete sessions
are required because the accompanying market-context features also compute 20
close-to-close returns. Those returns are feature inputs. The forecast target remains
the high-low range. No lognormal predictive-distribution assumption is imposed, and
zero ranges are allowed. A baseline forecast of zero remains zero in hypothetical
multiplicative scenarios.

The workflow consumes the repository's frozen universe, daily OHLCV bars, corporate
actions, company evidence, macro evidence, and local source documents. It retains the
point-in-time gateway and its documented availability assumptions. Only completed,
eligible bars enter features. Research cannot read future bars or target-day outcomes.
Yahoo history is already split-adjusted and must not be adjusted again.

These are retrospectively collected sources: declared historical availability is not the
same as a contemporaneously observed data feed. Yahoo can revise history, and some source
timestamps rely on conservative assumptions recorded in the snapshot provenance. Passing
the gateway enforces those assumptions; it does not independently prove that all source
contents were known in that exact form at the historical cutoff.

The current snapshot is an engineering fixture: 20 stocks and ten configured evaluation
sessions from October 1 through October 14, 2026. Its frozen market bars contain target-day
high/low for nine sessions, through October 13. October 14 is screening-only until real
target-day high/low are supplied. Label preparation reports missing and invalid coverage
explicitly and never substitutes absolute open-to-close returns. Source files stay intact.
The evaluator opens outcome bars for scoring only after a completed, manifested forecast
run has been published and integrity checked.

## Research and forecasting

LangGraph runs a repeating plan, allocate, and research loop. Initial shallow company
searches and a shared macro search provide broad coverage before the orchestrator proposes
bounded factual questions. Each assignment specifies affected companies, starting evidence,
permitted tools, hypothetical answers, and a reserved investigation cost.

Workers are general researchers using the same protocol, rather than debate characters
or agents obliged to defend stocks. Each can choose a query, inspect source spans, retrieve
historical features, inspect event timelines, or retrieve accepted cards. Tools operate
only on frozen eligible evidence and documents. They do not provide arbitrary filesystem
access or live web search. A question about a future release cannot earn credit for
retrieving that future result before the cutoff.

Cards separate source-supported observations from impact interpretations. They carry
source spans, availability and event timing, affected entities, uncertainty, counterevidence,
and version/supersession information. Deterministic checks validate citations, identity,
time boundaries, source spans, and duplicates. A semantic verifier evaluates factual support
and unsupported premises. A worker can receive one targeted repair request. A verified
event does not make its anticipated market impact a verified fact, and finite search
cannot establish that no significant information exists outside the permitted archive.

After accepted new evidence or a correction, the orchestrator may revise expected ranges.
Revisions must cite relevant accepted cards, retain the original historical baseline, and
explain a mechanism affecting the regular-session range. An overnight opening gap alone
does not imply a wide range after the open. Reusing unchanged cards does not justify repeated
numeric revisions. With no accepted new information, previous forecasts are retained.
Execution stops at its budget/batch limits or when it has no eligible research action.
Research failure preserves the latest valid forecasts, initially the historical baseline.

## Mean-preserving scenario allocation

The following quantities are scheduling hypotheses, not observed facts or calibrated
probabilities. The actual forecasting agent can respond to accepted evidence differently.
Consequently this is a **scenario-based acquisition score**, not calibrated expected value
of information.

Fix the current eligible universe of `n >= 1` stocks and a requested positive integer `k`.
Let `m[s] >= 0` be the current expected-range forecast for stock `s`, and let `m` denote
the vector of all these fixed forecasts. Selections contain `min(k,n)` stocks. Let `S0`
be the current selection of stocks with the largest forecasts, using stable ID tie breaks.
For any forecast vector `x`, define `T_k(x)` as the sum of its `min(k,n)` largest entries.

For an investigable question `q`, let `a` index its two or three mutually exclusive proposed
answers conditional on useful resolution. Let `p[a] >= 0` be their hypothetical probabilities,
with `sum_a p[a] = 1`. Let `d[s,a]` be its proposed log-multiplicative forecast shift for
an affected stock; an omitted shift is zero. Let `r[q]` in `[0,1]` be the estimated
probability of useful resolution, and let `c[q] > 0` be the reserved investigation cost in
US dollars. The function `exp` is the natural exponential function. All are fixed inputs
to this calculation, not claims about actual future draws.

1. Clip each proposed shift to `[-1.5,1.5]`, then shrink it by one half:
   `b[s,a] = 0.5 * max(-1.5, min(1.5, d[s,a]))`.
2. Define `Z[s] = sum_a p[a] * exp(b[s,a])`. Each exponential is positive, and at least
   one probability is positive because the probabilities sum to one, so `Z[s] > 0`.
3. Set `m[s,a] = m[s] * exp(b[s,a]) / Z[s]`. Unaffected stocks retain `m[s]`.
   Do not clip the resulting means afterward: that would generally break normalization.
4. Verify preservation of each affected current mean by substituting that definition:

   ```text
   sum_a p[a] * m[s,a]
     = sum_a p[a] * m[s] * exp(b[s,a]) / Z[s]
     = (m[s] / Z[s]) * sum_a p[a] * exp(b[s,a])
     = (m[s] / Z[s]) * Z[s]
     = m[s].
   ```

   Factoring out `m[s]/Z[s]` is valid because it does not depend on the answer. The last
   cancellation is valid because `Z[s] > 0`. If `m[s] = 0`, every scenario mean is zero
   and the same identity holds. This establishes first-moment consistency only.
5. Let `m^a` be the full forecast vector under answer `a`. Compute:

   ```text
   conditional_value(q)
     = sum_a p[a] * (T_k(m^a) - sum_{s in S0} m[s,a])

   priority(q) = r[q] * conditional_value(q) / c[q].
   ```

   Every bracket is nonnegative: `T_k(m^a)` maximizes the sum among all allowed selections,
   and retaining `S0` is one such selection. Nonnegative probabilities preserve this sign;
   multiplying by `r[q] >= 0` and dividing by `c[q] > 0` also preserve it. If the optimal
   selection never changes, the conditional value is zero even if forecasts change.

For example, suppose `k=1`, current forecasts are `A=0.025` and `B=0.020`, and only `B`
is affected by two equally likely proposed shifts of `+1` and `-1`. After shrinkage the
shifts are `+0.5` and `-0.5`. Then `Z[B]=(exp(0.5)+exp(-0.5))/2`, approximately `1.127626`.
The scenario means for `B` are approximately `0.0292423` and `0.0107577`; their average
is `0.0200000`. Only the first answer replaces `A` in the selection, so conditional value
is `0.5*(0.0292423-0.025)+0.5*0`, approximately `0.00212117`. With resolution probability
`0.8` and reserved cost `$0.10`, priority is approximately `0.0169694` range units per dollar.
These numbers illustrate scheduling, not a predicted financial gain from research.

Code logs the original shifts, clipping, shrinkage, normalizers, candidate scores, and
selection probabilities. Positive scenario scores are preferred with stable question-ID
tie breaks. No minimum value-per-dollar stopping threshold is enabled in this version.

At each dispatch, exploration occurs with probability `0.2` by default. Exploration weights
are `1 / (1 + sum of affected companies' prior inspection counts)`, normalized over eligible
questions. Discovery questions without scenarios enter through exploration. The logged
selection propensity includes both the exploration and exploitation branches. If there is
no positive exploitation action and exploration is enabled, remaining actions are sampled
entirely through exploration. Both its effective probability and the configured `0.2` are
recorded; a zero hypothetical score does not establish archive exhaustion.

The default two-worker batch excludes duplicate questions and shared event investigations.
Both questions are scored against the current forecast state; this is a batch approximation.
It does not claim exact joint answer enumeration or additive information value.

## Configuration and cost accounting

Select the workflow with `workflow: adaptive_range_v1`. Existing task and data fields retain
their original meanings. The `adaptive` configuration includes:

| Setting | Default or purpose |
|---|---|
| `policy` | `scenario`; comparison policies are described below |
| `daily_budget_usd` | `2.0`, normalized uncached model cost per forecast run |
| `max_workers`, `max_batches` | `2` concurrent assignments; `8` research batches |
| `max_worker_model_calls`, `max_worker_tool_calls` | `3` worker model calls and `4` tools per assignment |
| `exploration_probability` | `0.2` per dispatch |
| `seed`, `replicate_id` | Reproducible selection and separate generation-replicate cache namespaces |
| `memory_days`, `memory_path` | Seven-day active-card lifetime; optional persistent memory destination |
| `max_input_bytes` | Bounded request payload, with a separate schema-size guard |
| `prices`, `pricing_as_of` | Manually frozen rates for every exact live model ID and their reference date |
| `offline_price` | Explicit synthetic reference rates for offline accounting |

The inherited task limits also apply, including 170 model calls and 110 local tool calls
in the example. Planning, research, verification, forecast updates, repairs, and technical
retries all count toward the model budget. Finalization funds are held before research;
concurrent calls reserve conservative bounds atomically, then settle/refund unused funds.
Unknown live failure usage is charged at the full bound. Hidden SDK retries are disabled.
Unbounded or oversized requests and missing live pricing are refused.

To use live inference, change `agents.mode`, configure the provider/credential as for the
legacy workflow, and supply `adaptive.prices` entries covering `agents.model` and every
role override. Rates contain `input_per_million`, `output_per_million`, and optional
`request_usd`. Supply `adaptive.pricing_as_of` explicitly. No automatic price lookup or
silent fallback from missing live rates to offline reference prices is performed. The
inference-free `baseline` policy does not require live model pricing.

Cached or recorded responses remain charged at normalized uncached cost for policy
comparison, while actual provider-request cost is reported separately. Known token usage
is used when available; otherwise estimates are identified. Offline reference charges are
simulation/accounting values, not paid bills or published provider prices. Uncertain failed
live-call costs are conservative estimates, not reconciled invoices.

Identical complete requests may share responses within a stochastic replicate; differing
state/history inputs are different requests. Separate `replicate_id` namespaces prevent
such sharing between generation replicates. Each comparison arm pays its own normalized
charge even if the experiment avoids a duplicate actual request. Live evaluation also
requires an explicit aggregate budget sufficient for its full requested comparison grid.
Running the documented scripted commands or tests does not make paid inference requests.
Scripted and mocked tests do not validate current provider availability, model quality,
or the correctness of manually entered live prices.

## Masking, memory, and replay

Model-visible companies use opaque aliases. Evidence identifiers, source text, and tool
responses are redacted; a private mapping restores the original symbols for published
watchlists. This reduces obvious historical identification but does not prevent it:
sector, numerical histories, dates, distinctive events, or text can reveal an entity or
market episode. Masking is a contamination control, not a proof that model pretraining
contains no relevant future knowledge. Source archives and private audit artifacts retain
identifying information; the repository is not an anonymized dataset.

Memory exposes accepted cards only, with a seven-day default lifetime shortened by expiry
or supersession. Cards are append-only versions grouped by experiment arm and day; expiry
changes active retrieval rather than deleting the audit history. Historical dates are
processed chronologically. Policy, budget, and replicate arms have isolated memory.
Outcome labels never enter research memory. For separate `screen` invocations, set an
explicit `memory_path` to retain memory across runs; otherwise memory is local to the run.

Exact replay verifies hashes, restores the original frozen inputs and initial memory,
and consumes recorded model responses for that trajectory. It requires the rebuilt
watchlist, forecasts, allocations, research results, revisions, and tool traces to match.
Replay writes a new audited run and leaves the original unchanged; it needs no live model
request. An unseen state-dependent request is an error, not an invitation to invent an
answer from another trajectory's final cards.

Comparative evaluation instead executes each policy's actual trajectory. Previously unseen
queries or forecasting prompts must be executed against the permitted corpus/model.
Replaying one collection of complete research cards can compare aggregation under fixed
information, but cannot stand in for evaluating unrestricted adaptive research policies.

## Comparisons and subsequent study

The evaluation grid supports seven policies: historical-range `baseline` without inference,
`history_only` orchestrator forecasts, least-inspected `uniform` research, seeded `random`
research, current-ranking `boundary` research, `lexicographic` research, and `scenario`
allocation. The lexicographic comparator prefers scenario selection changes, then rank
displacement, uninspected evidence coverage, lower cost, and stable ID. Other components
are shared. Budget curves use `$0.50`, `$1`, and `$2` when configured and affordable.

The primary metric is realized top-`k` regret: the mean realized range of the best `k`
eligible stocks minus the mean realized range of the selected `k`. Smaller is better.
With partial labels, evaluation uses `min(k, labelled eligible stock count)`, preserves
selected rank positions, and assigns zero contribution to missing selected labels. Read
these incomplete-coverage scores alongside the coverage report rather than as complete
measurements of the actual day's regret.
Secondary diagnostics include selected range, overlap with the realized top group, NDCG,
forecast error, and revisions, together with label coverage, verification, evidence yield,
resolution, cost, and failure information. Insufficient labels produce explicit missing
coverage, and missing selected labels are not silently dropped to improve a ranking.

`evaluation.report_top_ks` defaults to `[3, 5, 10]`. The values must fit within
`task.top_k`; the configured watchlist size is also always reported. Each cutoff uses
the first stocks in the same predicted ordering. Its hindsight comparator is the best
group of that size in the full eligible universe, not just in the published watchlist.
The backward-compatible `metrics` field retains the primary watchlist-size metrics;
`metrics_by_k` adds each reporting cutoff to daily records and arm summaries. MAE uses
all labeled forecasts, so it is the same at every cutoff. Reports include
`daily_costs.csv`, `daily_metrics.csv`, and `daily_watchlists.csv`. Outcomes appear in
evaluation artifacts only after the prediction run has been published.

### Five-day live-model pilot

The bounded pilot configuration uses October 7, 8, 9, 12, and 13, 2026, all with complete
high/low labels for the 20-stock universe. It compares live `scenario` research with
the no-inference `baseline`, using one chronological replicate and top-3/5/10 reports.
It uses frozen historical evidence, not a prospective five-day market deployment.

```powershell
.venv\Scripts\python.exe scripts\run_live_range_study.py configs\adaptive_range_live_five_day_evaluation.yaml --receipt runs\NEW_RECEIPT.json
```

This command makes paid model calls. The runner refuses to overwrite an existing receipt
or progress log, writes per-day progress, and preserves all run audits. The evaluation
configuration must reserve the full grid under its aggregate cap. For the original pilot,
five sandbox connection failures were conservatively charged at $0.056768 per date.
The retry configuration therefore caps each successful attempt at $1.943232, with an
aggregate retry reservation of $9.71616, so original failures plus retries remain below
the authorized $2 per day and $10 total. Prices are frozen in the screening configuration;
verify model availability and refresh prices before a subsequent study.

The saved `actual_usd` field is a cost-accounting estimate: reported tokens priced at
the frozen rates, plus full bounds for failed or usage-missing calls. It is not a
reconciled provider invoice. The results-brief builder separates token-priced amounts
from uncertain-charge bounds and includes prior failed attempts:

```powershell
.venv\Scripts\python.exe scripts\build_live_range_report.py runs\live_five_day_20261104_retry_receipt.json --prior-receipt runs\live_five_day_20261104_receipt.json
```

The builder checks run manifests and daily/aggregate spending caps before writing the
offline HTML brief and consolidated cost, metrics, and watchlist CSVs. The pilot is
descriptive development evidence; it does not establish statistical superiority.

The present scripted fixture demonstrates engineering behavior. It does not calibrate
event effects, prove the scenario probabilities or resolution estimates accurate, or
establish a superiority claim. A forecast revision is not evidence of a forecasting gain.

Subsequent research needs a larger frozen archive with explicit event/earnings coverage,
a model/date protocol addressing historical knowledge, and an untouched test period.
Estimate event effects and acquisition calibration using earlier outcomes only. Check
that accessible evidence offers predictive headroom before funding large allocation
comparisons. Plan paired, dependence-aware analysis and sample sizes from meaningful
effect sizes and precision requirements. A nonsignificant result alone does not establish
that research has no value. Input-order permutation tests invariance; shuffled event
assignments are a negative control without a guarantee of random-policy performance.
