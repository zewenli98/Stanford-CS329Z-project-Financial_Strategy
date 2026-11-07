"""Auditable process scores for each LM-assisted screening run.

These scores use deterministic contracts and snapshot metadata.  They are useful
proxies for step correctness, but are deliberately not presented as human-label
semantic accuracy.
"""

from __future__ import annotations

import json
from collections import Counter
from collections import defaultdict as DefaultDict
from pathlib import Path
from typing import Any

from financial_strategy.retrieval.corrective import scripted_retrieval_judgment

JUDGMENT_FIELDS = ("company_match", "event_match", "sufficient", "action")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _rate(hits: int, count: int) -> float | None:
    return hits / count if count else None


def score_run_process(run_dir: Path) -> dict[str, Any]:
    """Score model execution, retrieval decisions, and final verification."""

    attempts = _read_jsonl(run_dir / "model_attempts.jsonl")
    traces = _read_jsonl(run_dir / "retrieval_traces.jsonl")
    evidence = _read_jsonl(run_dir / "snapshot/company_evidence.jsonl")
    cards_path = run_dir / "research_cards.json"
    cards = json.loads(cards_path.read_text()) if cards_path.exists() else []
    task_path = run_dir / "task.json"
    task = json.loads(task_path.read_text()) if task_path.exists() else {}

    completed = sum(row.get("status") == "completed" for row in attempts)
    retries = sum(
        str(row.get("attempt_type", "")).startswith(("technical_retry", "structured_repair"))
        for row in attempts
    )
    role_counts: dict[str, Counter] = DefaultDict(Counter)
    for row in attempts:
        role_counts[str(row.get("role", "unknown"))][str(row.get("status", "unknown"))] += 1

    evidence_by_id = {row["evidence_id"]: row for row in evidence}
    company_traces = [row for row in traces if row.get("scope") == "company"]
    field_hits = Counter()
    judged_count = 0
    identity_hits = 0
    identity_count = 0
    accepted_valid = 0
    accepted_count = 0
    decision_rows = []
    for trace in company_traces:
        returned_ids = list(dict.fromkeys(trace.get("returned_evidence_ids") or []))
        returned = [evidence_by_id[item] for item in returned_ids if item in evidence_by_id]
        symbol = trace.get("symbol")
        if returned:
            identity_count += 1
            identity_hits += all(item.get("symbol") == symbol for item in returned)
        judgment = trace.get("judgment")
        if not isinstance(judgment, dict):
            continue
        judged_count += 1
        oracle = scripted_retrieval_judgment(
            {
                "symbol": symbol,
                "question": trace.get("question", ""),
                "attempt": trace.get("attempt", 0),
                "decision_cutoff": task.get("decision_cutoff"),
                "news_lookback_days": task.get("news_lookback_days", 7),
                "evidence": returned,
            }
        )
        matches = {}
        for field in JUDGMENT_FIELDS:
            matches[field] = judgment.get(field) == oracle.get(field)
            field_hits[field] += matches[field]
        accepted_ids = judgment.get("accepted_evidence_ids") or []
        for evidence_id in accepted_ids:
            accepted_count += 1
            accepted_valid += evidence_id in returned_ids and evidence_id in evidence_by_id
        decision_rows.append(
            {
                "symbol": symbol,
                "attempt": trace.get("attempt", 0),
                "action": judgment.get("action"),
                "oracle_action": oracle.get("action"),
                "field_matches": matches,
            }
        )

    deterministic_pass = 0
    semantic_pass = 0
    eligible = 0
    for card in cards:
        deterministic_pass += card.get("deterministic_verification", {}).get("verdict") == "PASS"
        semantic_pass += card.get("semantic_verification", {}).get("verdict") == "PASS"
        eligible += bool(card.get("eligibility", {}).get("eligible"))

    judgment_accuracy = {
        f"{field}_accuracy": _rate(field_hits[field], judged_count) for field in JUDGMENT_FIELDS
    }
    available_judgment_rates = [value for value in judgment_accuracy.values() if value is not None]
    retrieval_score = (
        sum(available_judgment_rates) / len(available_judgment_rates)
        if available_judgment_rates
        else None
    )
    verification_rates = [
        value
        for value in (
            _rate(deterministic_pass, len(cards)),
            _rate(semantic_pass, len(cards)),
        )
        if value is not None
    ]
    verification_score = (
        sum(verification_rates) / len(verification_rates) if verification_rates else None
    )
    execution_score = _rate(completed, len(attempts))
    components = [
        value
        for value in (execution_score, retrieval_score, verification_score)
        if value is not None
    ]

    return {
        "score_scale": "0_to_1",
        "overall_process_score": sum(components) / len(components) if components else None,
        "overall_weighting": "equal_mean_of_available_execution_retrieval_verification_scores",
        "model_execution": {
            "score": execution_score,
            "attempt_count": len(attempts),
            "completed_attempt_count": completed,
            "failed_attempt_count": len(attempts) - completed,
            "retry_or_repair_attempt_count": retries,
            "retry_or_repair_rate": _rate(retries, len(attempts)),
            "by_role": {
                role: {
                    **dict(counts),
                    "score": _rate(counts["completed"], sum(counts.values())),
                }
                for role, counts in sorted(role_counts.items())
            },
        },
        "retrieval_and_tool_decisions": {
            "score": retrieval_score,
            "score_type": "deterministic_snapshot_proxy_not_human_gold",
            "tool_invocation_control": "workflow_managed; LM generates questions and judgments",
            "company_trace_count": len(company_traces),
            "judged_trace_count": judged_count,
            **judgment_accuracy,
            "returned_company_identity_accuracy": _rate(identity_hits, identity_count),
            "accepted_id_contract_accuracy": _rate(accepted_valid, accepted_count),
            "decision_audit": decision_rows,
        },
        "evidence_and_verification": {
            "score": verification_score,
            "card_count": len(cards),
            "deterministic_pass_rate": _rate(deterministic_pass, len(cards)),
            "semantic_verifier_pass_rate": _rate(semantic_pass, len(cards)),
            "eligible_card_rate": _rate(eligible, len(cards)),
        },
        "limitations": [
            "Retrieval judgment accuracy is compared with a deterministic metadata oracle, "
            "not human labels.",
            "Question quality and passage-level relevance require an independently labeled "
            "benchmark.",
            "Verifier pass rate measures pipeline acceptance, not independent factual correctness.",
        ],
    }
