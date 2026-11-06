"""Select a historical range baseline only on the supplied training partition."""

import bisect
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from financial_strategy.adaptive.evaluation import evaluate_range_cutoffs

ROOT = Path("runs/architecture_lab_20261106")
bars = pl.read_parquet("data/research100_v1/market/daily_bars.parquet")
history = defaultdict(list)
for row in bars.sort(["symbol", "session"]).to_dicts():
    history[row["symbol"]].append((row["session"], (row["high"] - row["low"]) / row["low"]))
sessions = sorted(d for d in bars["session"].unique() if date(2026, 1, 2) <= d <= date(2026, 7, 15))
methods = ["mean20", "mean5", "mean10", "median20", "ewma10"]
daily = []
for day in sessions:
    predictions = {m: [] for m in methods}
    labels = {}
    for symbol, rows in history.items():
        dates = [d for d, _ in rows]
        index = bisect.bisect_left(dates, day)
        values = [v for _, v in rows[max(0, index - 60) : index]]
        assert len(values) == 60 and rows[index][0] == day
        ewma = values[0]
        for value in values[1:]:
            ewma = (2 / 11) * value + (9 / 11) * ewma
        estimates = {
            "mean20": float(np.mean(values[-20:])),
            "mean5": float(np.mean(values[-5:])),
            "mean10": float(np.mean(values[-10:])),
            "median20": float(np.median(values[-20:])),
            "ewma10": ewma,
        }
        for method, value in estimates.items():
            predictions[method].append({"symbol": symbol, "predicted_range": value})
        labels[symbol] = rows[index][1]
    for method in methods:
        metrics = evaluate_range_cutoffs(predictions[method], labels, top_ks=[3, 5, 10])
        daily.append({"session": str(day), "method": method, "metrics": metrics})
summary = {}
for method in methods:
    rows = [r for r in daily if r["method"] == method]
    summary[method] = {
        str(k): {
            metric: sum(r["metrics"][str(k)][metric] for r in rows) / len(rows)
            for metric in ["ndcg_at_k", "top_k_overlap", "top_k_regret", "mae"]
        }
        for k in [3, 5, 10]
    }
winner = max(methods, key=lambda m: summary[m]["10"]["ndcg_at_k"])
output = {
    "selection": "highest mean training NDCG@10; no holdout labels used",
    "start": str(sessions[0]),
    "end": str(sessions[-1]),
    "session_count": len(sessions),
    "selected_method": winner,
    "summary": summary,
    "daily": daily,
}
(ROOT / "deterministic_training.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in output.items() if k != "daily"}, indent=2))
