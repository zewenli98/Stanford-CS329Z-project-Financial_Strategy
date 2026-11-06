"""Freeze the starting code, experiment dates, and current provider prices."""

import json
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

ROOT = Path("runs/architecture_lab_20261106")
ROOT.mkdir(parents=True, exist_ok=True)
if (ROOT / "plan.json").exists():
    raise SystemExit("Lab already initialized; preserve original plan")
now = datetime.now(UTC)
plan = {
    "started_at": now.isoformat(),
    "deadline": (now + timedelta(minutes=55)).isoformat(),
    "new_api_cap_usd": 10.0,
    "development_dates": ["2026-08-03", "2026-09-01", "2026-10-01"],
    "holdout_dates": [
        "2026-07-20",
        "2026-08-14",
        "2026-09-10",
        "2026-09-22",
        "2026-10-14",
        "2026-10-27",
    ],
    "deterministic_training_end": "2026-07-15",
    "universe": "data/research100_v1/universe/symbols.json",
    "universe_count": 100,
    "watchlist_count": 10,
    "evidence_rule": "published_at and available_at strictly before target-day New York midnight",
    "primary_metric": "mean daily NDCG@10",
    "secondary_metrics": [
        "overlap@3",
        "overlap@5",
        "overlap@10",
        "watchlist10_recall_actual_top3",
        "completion",
        "cost",
    ],
    "selection_rule": (
        "Select architecture on development NDCG@10; freeze before evaluating holdout. "
        "Include failures in the registry."
    ),
    "planned_arms": [
        "legacy compact control",
        "guided repair and passage retrieval",
        "deterministic evidence packets with hybrid shortlist",
        "training-selected historical baseline",
        "optional alternate model within remaining time and budget",
    ],
}
(ROOT / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
with zipfile.ZipFile(ROOT / "starting_source.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for directory in ["src", "scripts", "configs", "tests"]:
        for path in Path(directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                z.write(path, path.as_posix())
with urlopen("https://openrouter.ai/api/v1/models", timeout=30) as response:
    catalog = json.load(response)
(ROOT / "model_catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
matches = [
    {"id": m["id"], "pricing": m["pricing"]}
    for m in catalog["data"]
    if any(s in m["id"] for s in ["gpt-5.6", "gemini-3", "claude-haiku", "qwen3.5"])
]
print(json.dumps({"plan": plan, "available_models": matches}, indent=2))
