import json
from pathlib import Path

from financial_strategy.evaluation.process_scores import score_run_process


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_score_run_process_reports_execution_retrieval_and_verification(tmp_path: Path) -> None:
    _jsonl(
        tmp_path / "model_attempts.jsonl",
        [
            {"role": "company_evidence", "attempt_type": "company_question", "status": "completed"},
            {"role": "company_evidence", "attempt_type": "structured_repair", "status": "failed"},
        ],
    )
    evidence = {
        "evidence_id": "ev1",
        "symbol": "ABC",
        "scope": "company",
        "published_at": "2026-01-09T12:00:00+00:00",
        "event_type": "earnings",
        "event_types": ["earnings"],
        "title": "ABC earnings",
        "excerpt": "earnings",
    }
    _jsonl(tmp_path / "snapshot/company_evidence.jsonl", [evidence])
    judgment = {
        "company_match": True,
        "event_match": True,
        "sufficient": True,
        "action": "accept",
        "gap": None,
        "rewritten_query": None,
        "accepted_evidence_ids": ["ev1"],
    }
    _jsonl(
        tmp_path / "retrieval_traces.jsonl",
        [
            {
                "scope": "company",
                "symbol": "ABC",
                "question": "latest earnings",
                "attempt": 0,
                "returned_evidence_ids": ["ev1"],
                "judgment": judgment,
            }
        ],
    )
    (tmp_path / "task.json").write_text(
        json.dumps({"decision_cutoff": "2026-01-10T09:00:00+00:00", "news_lookback_days": 7})
    )
    (tmp_path / "research_cards.json").write_text(
        json.dumps(
            [
                {
                    "deterministic_verification": {"verdict": "PASS"},
                    "semantic_verification": {"verdict": "PASS"},
                    "eligibility": {"eligible": True},
                }
            ]
        )
    )

    report = score_run_process(tmp_path)

    assert report["model_execution"]["score"] == 0.5
    assert report["model_execution"]["retry_or_repair_attempt_count"] == 1
    assert report["retrieval_and_tool_decisions"]["score"] == 1.0
    assert report["retrieval_and_tool_decisions"]["accepted_id_contract_accuracy"] == 1.0
    assert report["evidence_and_verification"]["score"] == 1.0
