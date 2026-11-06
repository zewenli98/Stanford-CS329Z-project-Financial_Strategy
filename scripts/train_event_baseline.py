"""Test whether a simple dated-event uplift adds value before paying for agents."""

import bisect
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import polars as pl

from financial_strategy.adaptive.evaluation import evaluate_range_cutoffs
from financial_strategy.adaptive.research_v2 import event_priority

ROOT = Path("runs/architecture_lab_20261106")
training = json.loads((ROOT / "deterministic_training.json").read_text())
plan = {
    "registered_at": datetime.now().isoformat(),
    "dates": "same133 training dates ending2026-07-15",
    "base_method": "ewma10",
    "event_uplift_grid": [0, 0.25, 0.5, 1.0],
    "formula": "forecast=history_baseline*(1+alpha*max_typed_event_recency_score)",
    "rule": "SEC records published and available before target midnight within prior7 days; "
    "select alpha on training NDCG@10 only",
}
(ROOT / "event_baseline_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
bars = pl.read_parquet("data/research100_v1/market/daily_bars.parquet")
history = {}
for symbol in bars["symbol"].unique():
    rows = bars.filter(pl.col("symbol") == symbol).sort("session").to_dicts()
    history[symbol] = [(r["session"], (r["high"] - r["low"]) / r["low"]) for r in rows]
events = []
for line in Path("data/2023-11-01_2026-10-30/company/evidence.jsonl").read_text().splitlines():
    r = json.loads(line)
    if r["symbol"] in history:
        events.append(
            SimpleNamespace(
                symbol=r["symbol"],
                available_at=datetime.fromisoformat(r["available_at"].replace("Z", "+00:00")),
                published_at=datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")),
                event_types=r["event_types"],
                event_type=None,
            )
        )
days = sorted({r["session"] for r in training["daily"]})
output = []
for day in days:
    boundary = datetime.fromisoformat(day).replace(tzinfo=ZoneInfo("America/New_York"))
    cutoff = datetime.combine(boundary.date(), time(9), ZoneInfo("America/New_York"))
    eligible = [
        r
        for r in events
        if cutoff - timedelta(days=7) <= r.available_at < boundary and r.published_at < boundary
    ]
    scores = event_priority(eligible, cutoff)
    labels, bases = {}, {}
    for symbol, rows in history.items():
        i = bisect.bisect_left([d for d, _ in rows], boundary.date())
        values = [v for _, v in rows[i - 60 : i]]
        average = values[0]
        for v in values[1:]:
            average = (2 / 11) * v + (9 / 11) * average
        bases[symbol] = average
        labels[symbol] = rows[i][1]
    for alpha in plan["event_uplift_grid"]:
        predictions = [
            {"symbol": s, "predicted_range": v * (1 + alpha * scores.get(s, 0))}
            for s, v in bases.items()
        ]
        output.append(
            {
                "session": day,
                "alpha": alpha,
                "metrics": evaluate_range_cutoffs(predictions, labels, top_ks=[3, 5, 10]),
            }
        )
summary = {
    str(a): {
        str(k): {
            m: sum(r["metrics"][str(k)][m] for r in output if r["alpha"] == a) / len(days)
            for m in ["ndcg_at_k", "top_k_overlap"]
        }
        for k in [3, 5, 10]
    }
    for a in plan["event_uplift_grid"]
}
selected = max(plan["event_uplift_grid"], key=lambda a: summary[str(a)]["10"]["ndcg_at_k"])
body = {"plan": plan, "selected_alpha": selected, "summary": summary, "daily": output}
(ROOT / "event_baseline_training.json").write_text(json.dumps(body, indent=2), encoding="utf-8")
print(json.dumps({"selected_alpha": selected, "summary": summary}, indent=2))
