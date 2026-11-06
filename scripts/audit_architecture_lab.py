"""Independently verify all completed lab forecasts, costs, timing, and citations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from financial_strategy.audit.writer import verify_manifest

ROOT = Path("runs/architecture_lab_20261106")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def instant(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    state = read(ROOT / "registry.json")
    source = pl.read_parquet("data/research100_v1/market/daily_bars.parquet")
    by_day = defaultdict(dict)
    for r in source.to_dicts():
        by_day[str(r["session"])][r["symbol"]] = (r["high"] - r["low"]) / r["low"]
    universe = {
        s["symbol"] for s in read(Path("data/research100_v1/universe/symbols.json"))["securities"]
    }
    audits = []
    for job in state["jobs"]:
        if job["status"] != "completed":
            continue
        root = Path(job["run_dir"])
        verify_manifest(root)
        preds = read(root / "range_predictions.json")
        assert len(preds) == 100 and {p["symbol"] for p in preds} == universe
        task = read(root / "task.json")
        cutoff = instant(task["decision_cutoff"])
        boundary = datetime.fromisoformat(job["session"]).replace(
            tzinfo=ZoneInfo("America/New_York")
        )
        bars = pl.read_parquet(root / "snapshot/market.parquet")
        assert str(bars["session"].max()) < job["session"] and bars["available_at"].max() <= cutoff
        evidence = rows(root / "snapshot/company_evidence.jsonl") + rows(
            root / "snapshot/macro_evidence.jsonl"
        )
        assert all(
            instant(e["available_at"]) < boundary and instant(e["published_at"]) < boundary
            for e in evidence
        )
        mapping = read(root / "private_mapping.json")
        sources = {e["evidence_id"]: e for e in evidence}
        research = rows(root / "research_results.jsonl")
        read_spans = [
            t["result"]["span"] for q in research for t in q["tool_traces"] if "span" in t["result"]
        ]
        cards = read(root / "research_cards.json")
        for card in cards:
            for span in card["source_spans"]:
                assert any(
                    span["evidence_id"] == s["evidence_id"]
                    and s["start"] <= span["start"] < span["end"] <= s["end"]
                    and span["quote"]
                    == s["quote"][span["start"] - s["start"] : span["end"] - s["start"]]
                    for s in read_spans
                )
                evidence_id = mapping["sources"][span["evidence_id"]]["evidence_id"]
                assert (
                    instant(sources[evidence_id]["available_at"])
                    <= instant(card["available_at"])
                    < boundary
                )
        labels = by_day[job["session"]]
        ordered = sorted(preds, key=lambda p: (-p["predicted_range"], p["symbol"]))
        actual = sorted(labels, key=lambda s: (-labels[s], s))
        capture = {}
        for k in [3, 5, 10]:
            values = [labels[p["symbol"]] for p in ordered[:k]]
            ideal = [labels[s] for s in actual[:k]]
            ndcg = sum(v / math.log2(i + 2) for i, v in enumerate(values)) / sum(
                v / math.log2(i + 2) for i, v in enumerate(ideal)
            )
            overlap = len({p["symbol"] for p in ordered[:k]} & set(actual[:k])) / k
            assert abs(ndcg - job["metrics"][str(k)]["ndcg_at_k"]) < 1e-12
            assert abs(overlap - job["metrics"][str(k)]["top_k_overlap"]) < 1e-12
            capture[str(k)] = len({p["symbol"] for p in ordered[:10]} & set(actual[:k])) / k
        calls = rows(root / "model_calls.jsonl")
        attempts = rows(root / "model_attempts.jsonl")
        budget = read(root / "budget.json")
        assert abs(sum(c["actual_usd"] for c in calls) - job["charged_usd"]) < 1e-9
        assert budget["reserved_usd"] == 0 and job["charged_usd"] <= job["reserved_usd"] + 1e-9
        legit_abstain = sum(
            q["stop_reason"] == "archive_insufficient"
            and not any("error_code" in t["result"] for t in q["tool_traces"])
            for q in research
        )
        audits.append(
            {
                "id": job["id"],
                "phase": job["phase"],
                "arm": job["arm"],
                "session": job["session"],
                "manifest": "verified",
                "forecast_count": len(preds),
                "frozen_evidence_count": len(evidence),
                "accepted_cards": len(cards),
                "verified_citation_spans": sum(len(c["source_spans"]) for c in cards),
                "source_timing_violations": 0,
                "independent_metrics": "exact_match",
                "watchlist10_capture_actual_top_k": capture,
                "resolved_or_cleanly_abstained": sum(
                    q["stop_reason"] == "accepted" for q in research
                )
                + legit_abstain,
                "research_attempts": len(research),
                "targeted_repair_calls": sum(c["attempt_type"] == "targeted_repair" for c in calls),
                "paid_live_responses": sum(
                    c.get("result", {}).get("source") == "live" for c in calls
                ),
                "failed_model_calls": sum(c["status"] != "completed" for c in calls),
                "failed_workflow_attempts": sum(a["status"] == "failed" for a in attempts),
                "model_failure_reasons": dict(
                    Counter(
                        a.get("error_code", "unknown") for a in attempts if a["status"] == "failed"
                    )
                ),
                "estimated_cost_calls": sum(c.get("usage_estimated", False) for c in calls),
                "replay": "not_requested",
            }
        )
    spend = sum(j.get("charged_usd", j["reserved_usd"]) for j in state["jobs"])
    assert spend <= 10 + 1e-9
    if args.replay and state["selection"]:
        import financial_strategy.adaptive.budget as budget_module
        from financial_strategy.adaptive.replay import replay_adaptive

        def forbidden(*a, **k):
            raise AssertionError("Audit replay must never construct a live client")

        budget_module.LiveClient = forbidden
        chosen = [
            j
            for j in state["jobs"]
            if j["phase"] == "holdout"
            and j["arm"] == state["selection"]["arm"]
            and j["status"] == "completed"
        ]
        for job in [chosen[0], chosen[-1]] if chosen else []:
            original = Path(job["run_dir"])
            assert replay_adaptive(original) == read(original / "watchlist.json")
            next(a for a in audits if a["id"] == job["id"])["replay"] = "exact_match_no_live_client"
    initial = read(Path("data/research100_v1/preparation.json"))
    for name, digest in initial["source_sha256"].items():
        assert hashlib.sha256((Path(initial["source"]) / name).read_bytes()).hexdigest() == digest
    runtime = read(ROOT / "runtime_sha256.json")
    post_audit_changes = read(ROOT / "post_audit_source_changes.json")
    for name, digest in runtime.items():
        expected = post_audit_changes.get(name, {}).get("sha256", digest)
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    output = {
        "all_checks_passed": True,
        "completed_jobs_audited": len(audits),
        "new_api_spending_usd": spend,
        "cap_usd": 10,
        "source_files_unchanged": True,
        "tested_runtime_files_unchanged": len(runtime) - len(post_audit_changes),
        "post_audit_source_changes": post_audit_changes,
        "jobs": audits,
        "limitations": [
            "Source publication metadata and exact recorded quotations were checked; "
            "source timing is not a complete independent archival PIT certification.",
            "Current-constituent survivorship bias and vendor historical "
            "adjustment limitations remain.",
        ],
    }
    (ROOT / "audit.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k != "jobs"}, indent=2))


if __name__ == "__main__":
    main()
