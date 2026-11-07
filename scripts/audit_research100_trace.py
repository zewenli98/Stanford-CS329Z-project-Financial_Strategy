"""Read-only audit of coverage, research completion, costs, and source timing."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from financial_strategy.adaptive.agents import _numbers
from financial_strategy.audit.writer import verify_manifest


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def main():
    base = Path("runs")
    results = read_json(base / "research100_results.json")
    report = read_json(Path(results["report_paths"]["corrected"]))
    days, questions, accepted_cards = [], [], []
    totals = Counter()
    role_cost = defaultdict(float)
    cost_by_outcome = defaultdict(float)
    temporal_violations = []
    for row in report["sessions"]:
        root = base / row["run_id"]
        verify_manifest(root)
        task = read_json(root / "task.json")
        cutoff = timestamp(task["decision_cutoff"])
        midnight = datetime.combine(cutoff.date(), time(), ZoneInfo("America/New_York"))
        mapping = read_json(root / "private_mapping.json")
        frozen_records = read_jsonl(root / "snapshot/company_evidence.jsonl")
        frozen_records += read_jsonl(root / "snapshot/macro_evidence.jsonl")
        by_id = {r["evidence_id"]: r for r in frozen_records}
        records = {opaque: by_id[m["evidence_id"]] for opaque, m in mapping["sources"].items()}
        temporal = {}
        for label, items in [("frozen", frozen_records), ("model_visible", list(records.values()))]:
            temporal[label] = {
                "record_count": len(items),
                "latest_published": max((r["published_at"] for r in items), default=None),
                "after_cutoff": sum(timestamp(r["available_at"]) > cutoff for r in items),
                "published_after_cutoff": sum(timestamp(r["published_at"]) > cutoff for r in items),
                "available_on_target_day": sum(
                    timestamp(r["available_at"]) >= midnight for r in items
                ),
            }
            for r in items:
                if timestamp(r["published_at"]) > cutoff or timestamp(r["available_at"]) > cutoff:
                    temporal_violations.append(
                        {"session": row["session"], "source": r["evidence_id"]}
                    )
        calls = read_jsonl(root / "model_calls.jsonl")
        question_costs = defaultdict(float)
        for call in calls:
            cost = call["actual_usd"]
            op = call["payload"]["op"]
            role_cost[op] += cost
            question = call["payload"].get("question", {}).get("question_id")
            question = question or call["payload"].get("card", {}).get("question_id")
            if question:
                question_costs[question] += cost
        question_defs = {
            r["question"]["question_id"]: r["question"]
            for r in read_jsonl(root / "research_questions.jsonl")
        }
        outcome_counts, tool_errors, tool_count = Counter(), [], 0
        for q in read_jsonl(root / "research_results.jsonl"):
            qid = q["question_id"]
            definition = question_defs[qid]
            outcome = q["stop_reason"]
            outcome_counts[outcome] += 1
            totals[outcome] += 1
            cost_by_outcome[outcome] += question_costs[qid]
            errors = []
            for trace in q["tool_traces"]:
                tool_count += 1
                if "error_code" in trace["result"]:
                    errors.append(
                        {
                            "tool": trace["tool"],
                            "arguments": trace["arguments"],
                            "error": trace["result"]["error_code"],
                        }
                    )
                if "span" in trace["result"]:
                    r = records[trace["result"]["span"]["evidence_id"]]
                    if timestamp(r["available_at"]) > cutoff:
                        temporal_violations.append({"session": row["session"], "question": qid})
            tool_errors.extend(errors)
            questions.append(
                {
                    "session": row["session"],
                    "question_id": qid,
                    "symbols": [mapping["companies"][s] for s in definition["symbols"]],
                    "discovery": definition["discovery"],
                    "text": q["text"],
                    "outcome": outcome,
                    "accepted_cards": q["accepted_card_ids"],
                    "tool_attempts": len(q["tool_traces"]),
                    "tool_errors": errors,
                    "direct_worker_and_verifier_cost_usd": question_costs[qid],
                }
            )
        cards = read_json(root / "research_cards.json")
        for card in cards:
            sources = []
            for span in card["source_spans"]:
                r = records[span["evidence_id"]]
                sources.append(
                    {
                        "opaque_id": span["evidence_id"],
                        "source_id": r["evidence_id"],
                        "symbol": r["symbol"],
                        "published_at": r["published_at"],
                        "available_at": r["available_at"],
                        "retrieved_at": r["retrieved_at"],
                        "source_url": r["source_url"],
                        "content_path": r["content_path"],
                        "age_hours": (cutoff - timestamp(r["available_at"])).total_seconds() / 3600,
                        "before_target_day_midnight": timestamp(r["available_at"]) < midnight,
                        "span_start": span["start"],
                        "span_end": span["end"],
                        "event_type": r.get("event_type"),
                        "event_types": r.get("event_types"),
                    }
                )
            accepted_cards.append(
                {
                    "session": row["session"],
                    "card_id": card["card_id"],
                    "symbols": [mapping["companies"][s] for s in card["symbols"]],
                    "observations": card["observations"],
                    "inferences": card["inferences"],
                    "range_mechanism": card["range_mechanism"],
                    "sources": sources,
                }
            )
        bars = pl.read_parquet(root / "snapshot/market.parquet")
        errors = Counter(e["error"] for e in tool_errors)
        coverage = {}
        source_bars = pl.read_parquet("data/research100_v1/market/daily_bars.parquet")
        target_bars = source_bars.filter(pl.col("session").cast(pl.String) == row["session"])
        realized = sorted(
            [(r["symbol"], (r["high"] - r["low"]) / r["low"]) for r in target_bars.to_dicts()],
            key=lambda pair: (-pair[1], pair[0]),
        )
        watchlist = {r["symbol"] for r in row["watchlist"]}
        candidates = set(read_json(root / "research_screening.json")["candidate_symbols"])
        recall, screening_recall = {}, {}
        for k in [3, 5, 10]:
            metric = row["metrics_by_k"][str(k)]
            coverage[str(k)] = {
                "hits": round(metric["top_k_overlap"] * k),
                "denominator": k,
                "rate": metric["top_k_overlap"],
            }
            truth = {symbol for symbol, _ in realized[:k]}
            recall[str(k)] = len(watchlist & truth)
            screening_recall[str(k)] = len(candidates & truth)
        same_day = [
            {"id": r["evidence_id"], "symbol": r["symbol"], "published_at": r["published_at"]}
            for r in records.values()
            if timestamp(r["available_at"]) >= midnight
        ]
        rejected_numbers = []
        for card in read_json(root / "rejected_cards.json"):
            supported = _numbers("\n".join(s["quote"] for s in card["source_spans"]))
            rejected_numbers.append(
                {
                    "id": card["card_id"],
                    "issues": card["verification_issues"],
                    "unsupported_numbers": [
                        {"text": obs, "numbers": sorted(_numbers(obs) - supported)}
                        for obs in card["observations"]
                        if _numbers(obs) - supported
                    ],
                }
            )
        predictions = read_json(root / "range_predictions.json")
        days.append(
            {
                "session": row["session"],
                "run_id": row["run_id"],
                "cutoff": str(cutoff),
                "coverage": coverage,
                "outcomes": dict(outcome_counts),
                "tool_attempts": tool_count,
                "tool_errors": tool_errors,
                "tool_error_counts": dict(errors),
                "cost_usd": sum(c["actual_usd"] for c in calls),
                "temporal": temporal,
                "latest_market_session": str(bars["session"].max()),
                "rejection_codes": row["integrity"]["rejection_code_distribution"],
                "model_calls": len(calls),
                "model_call_failures": sum(c["status"] != "completed" for c in calls),
                "watchlist10_capture_actual_top_k": recall,
                "screen20_capture_actual_top_k": screening_recall,
                "same_day_model_visible_sources": same_day,
                "rejected_numeric_analysis": rejected_numbers,
                "changed_forecasts": [
                    {
                        "symbol": p["symbol"],
                        "baseline": p["baseline_range"],
                        "updated": p["predicted_range"],
                    }
                    for p in predictions
                    if p["predicted_range"] != p["baseline_range"]
                ],
            }
        )
    output = {
        "scope": "corrected 3-day 100-stock live pilot; no new live model calls",
        "coverage": {
            str(k): {
                "hits": sum(d["coverage"][str(k)]["hits"] for d in days),
                "denominator": k * len(days),
                "rate": sum(d["coverage"][str(k)]["hits"] for d in days) / (k * len(days)),
            }
            for k in [3, 5, 10]
        },
        "outcomes": dict(totals),
        "question_count": len(questions),
        "cost_by_operation_usd": dict(role_cost),
        "direct_cost_by_outcome_usd": dict(cost_by_outcome),
        "total_cost_usd": sum(d["cost_usd"] for d in days),
        "total_tool_attempts": sum(d["tool_attempts"] for d in days),
        "total_tool_errors": sum(len(d["tool_errors"]) for d in days),
        "temporal_violations": temporal_violations,
        "days": days,
        "questions": questions,
        "accepted_cards": accepted_cards,
    }
    (base / "research100_process_audit.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in output.items() if k not in {"questions", "accepted_cards"}}, indent=2
        )
    )


if __name__ == "__main__":
    main()
