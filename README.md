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

- **Prediction A — learned correction:** verified research is encoded as structured event
  features and used to train a residual correction to M. The correction is enabled only if
  it passes chronological validation.
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
