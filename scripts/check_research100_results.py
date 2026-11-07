"""Independently audit a research100 evaluation and write a compact result summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import polars as pl

from financial_strategy.audit.writer import verify_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    report_path = Path(receipt["report_path"])
    report = json.loads(report_path.read_text())
    verify_manifest(report_path.parent)
    preparation = json.loads(Path("data/research100_v1/preparation.json").read_text())
    for name, expected in preparation["source_sha256"].items():
        actual = hashlib.sha256((Path(preparation["source"]) / name).read_bytes()).hexdigest()
        assert actual == expected, f"Source changed: {name}"
    bars = pl.read_parquet("data/research100_v1/market/daily_bars.parquet")
    sessions = []
    for row in report["sessions"]:
        assert row["status"] == "completed", row
        root = report_path.parent.parent / row["run_id"]
        assert verify_manifest(root)["status"] == "completed"
        predictions = json.loads((root / "range_predictions.json").read_text())
        screening = json.loads((root / "research_screening.json").read_text())
        task = json.loads((root / "task.json").read_text())
        frozen = pl.read_parquet(root / "snapshot/market.parquet")
        features = json.loads((root / "range_features.json").read_text())
        assert len(predictions) == len(features) == 100
        assert len({p["symbol"] for p in predictions}) == 100
        assert len(row["watchlist"]) == 10
        assert screening["eligible_count"] == 100 and screening["candidate_count"] == 20
        assert str(frozen["session"].max()) < row["session"]
        assert not (root / "snapshot/macro_evidence.jsonl").read_text().strip()
        unresearched = set(features) - set(screening["candidate_symbols"])
        assert len(unresearched) == 80
        assert all(
            p["predicted_range"] == features[p["symbol"]]["mean_range_20d"]
            for p in predictions
            if p["symbol"] in unresearched
        )
        target = bars.filter(pl.col("session").cast(pl.String) == row["session"])
        labels = {r["symbol"]: (r["high"] - r["low"]) / r["low"] for r in target.to_dicts()}
        assert len(labels) == 100
        ordered = sorted(predictions, key=lambda p: (-p["predicted_range"], p["symbol"]))
        for k in [3, 5, 10]:
            values = [labels[p["symbol"]] for p in ordered[:k]]
            oracle = sorted(labels.values(), reverse=True)[:k]
            dcg = sum(v / math.log2(i + 2) for i, v in enumerate(values))
            ideal = sum(v / math.log2(i + 2) for i, v in enumerate(oracle))
            metrics = row["metrics_by_k"][str(k)]
            assert abs(metrics["ndcg_at_k"] - dcg / ideal) < 1e-12
            assert abs(metrics["top_k_regret"] - (sum(oracle) - sum(values)) / k) < 1e-12
            assert metrics["label_count"] == 100
        calls = [json.loads(s) for s in (root / "model_calls.jsonl").read_text().splitlines() if s]
        budget = json.loads((root / "budget.json").read_text())
        assert budget["actual_usd"] <= budget["cap_usd"] + 1e-12
        assert budget["reserved_usd"] == 0
        if row["policy"] == "baseline":
            assert not calls and budget["actual_usd"] == 0
        # Audit recorded request size and distinguish actual live responses from scripted ones.
        sources = {}
        failures = []
        for call in calls:
            result = call.get("result", {})
            source = result.get("source", call["status"])
            sources[source] = sources.get(source, 0) + 1
            if call["status"] != "completed":
                failures.append({"role": call["role"], "error": call.get("error_code")})
        replay_check = "not_requested"
        if args.replay and row["policy"] != "baseline":
            from financial_strategy.adaptive.replay import replay_adaptive

            expected_watchlist = json.loads((root / "watchlist.json").read_text())
            assert replay_adaptive(root) == expected_watchlist
            replay_check = "exact_match_without_live_requests"
        sessions.append(
            {
                "session": row["session"],
                "policy": row["policy"],
                "run_id": row["run_id"],
                "status": row["status"],
                "forecast_count": len(predictions),
                "research_candidate_count": screening["candidate_count"],
                "final_changed_forecast_count": sum(
                    p["predicted_range"] != p["baseline_range"] for p in predictions
                ),
                "ndcg": {k: row["metrics_by_k"][k]["ndcg_at_k"] for k in ["3", "5", "10"]},
                "mean_selected_range": row["metrics"]["mean_selected_range"],
                "top_k_regret": row["metrics"]["top_k_regret"],
                "actual_usd": budget["actual_usd"],
                "model_response_sources": sources,
                "model_failures": failures,
                "integrity": row["integrity"],
                "max_payload_bytes": max(
                    [
                        0,
                        *[
                            len(
                                json.dumps(
                                    c["payload"], sort_keys=True, separators=(",", ":")
                                ).encode()
                            )
                            for c in calls
                        ],
                    ]
                ),
                "decision_cutoff": task["decision_cutoff"],
                "recorded_replay": replay_check,
            }
        )
    summary = {
        "report_path": str(report_path.resolve()),
        "source_files_unchanged": True,
        "all_checks_passed": True,
        "session_count": len(sessions),
        "universe_count": 100,
        "research_candidate_limit": 20,
        "macro_policy": "exclude_low_confidence_from_primary_live_inputs",
        "source_limitations": preparation["limitations"],
        "total_actual_usd": sum(r["actual_usd"] for r in sessions),
        "sessions": sessions,
        "summary": report["summary"],
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ["all_checks_passed", "session_count", "total_actual_usd"]}
        )
    )


if __name__ == "__main__":
    main()
