"""Generate versioned example configurations without copying source datasets."""

from pathlib import Path

import yaml

root = Path(__file__).resolve().parents[1]
base = yaml.safe_load((root / "configs/screening_2026-10-14.yaml").read_text())
base["workflow"] = "adaptive_range_v1"
base["agents"]["mode"] = "scripted"
base["agents"]["model"] = "scripted"
base["agents"]["roles"]["macro_evidence"]["max_tokens"] = 8192
base["agents"]["roles"]["synthesis"]["max_tokens"] = 4096
for role, settings in base["agents"]["roles"].items():
    settings["prompt_version"] = f"adaptive_{role}_v1"
base["adaptive"] = {
    "policy": "scenario",
    "daily_budget_usd": 2.0,
    "max_workers": 2,
    "max_batches": 8,
    "exploration_probability": 0.2,
    "max_worker_model_calls": 3,
    "max_worker_tool_calls": 4,
    "seed": 0,
    "replicate_id": "0",
    "memory_days": 7,
    "max_input_bytes": 60000,
    "offline_price": {"input_per_million": 0.10, "output_per_million": 0.40},
}
(root / "configs/adaptive_range_v1.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
evaluation = yaml.safe_load((root / "configs/evaluation_2026-10-01_2026-10-14.yaml").read_text())
evaluation["screening_config"] = "configs/adaptive_range_v1.yaml"
body = evaluation["evaluation"]
for key in ("labels", "variants", "ndcg"):
    body.pop(key, None)
body.update(
    {
        "policies": [
            "baseline",
            "history_only",
            "uniform",
            "random",
            "boundary",
            "lexicographic",
            "scenario",
        ],
        "budgets_usd": [0.5, 1.0, 2.0],
        "replicates": 1,
        # This is an explicit ceiling, not an instruction to spend it. Scripted mode spends $0.
        "aggregate_budget_usd": 2.0,
    }
)
(root / "configs/adaptive_range_evaluation.yaml").write_text(
    yaml.safe_dump(evaluation, sort_keys=False)
)
