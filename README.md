# Financial Strategy

A point-in-time research workflow for ranking US common stocks by expected absolute
open-to-close movement. The MVP combines deterministic features and ranking with bounded
LangGraph research agents and independent verification.

## Quick start

```bash
uv sync --extra dev
uv run pytest
```

## Real-data snapshot

The 2026-10-01 through 2026-10-14 snapshot freezes real inputs for 20 stocks from Yahoo
Finance, SEC EDGAR, verified company news pages, BLS, and pre-release Dow Jones consensus
reports from CNBC. The model only receives evidence that passed the point-in-time gateway.

```bash
uv run python scripts/fetch_real_data.py
uv run financial-strategy screen --config configs/screening_2026-10-14.yaml
uv run financial-strategy evaluate --config configs/evaluation_2026-10-01_2026-10-14.yaml
```

Live runs use OpenRouter structured output. Configure the shared model with `agents.model`
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

The project is research software. It does not place trades and its heuristic rank score is
not a probability.
