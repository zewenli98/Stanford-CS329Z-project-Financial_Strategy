"""Accuracy of the company retrieval judgment, separate from watchlist rank."""

from __future__ import annotations

import json
from pathlib import Path

from financial_strategy.retrieval.corrective import scripted_retrieval_judgment

FIELDS = ("company_match", "event_match", "sufficient", "action")


def retrieval_accuracy(pairs: list[tuple[dict, dict | None]]) -> dict:
    """Score gold labels against judgments. Missing judgments stay out of the rates."""

    matched = [(gold, predicted) for gold, predicted in pairs if predicted is not None]
    missing = len(pairs) - len(matched)

    def rate(field: str) -> float | None:
        if not matched:
            return None
        hits = sum(predicted[field] == gold[field] for gold, predicted in matched)
        return hits / len(matched)

    return {
        "count": len(pairs),
        "matched": len(matched),
        "missing": missing,
        **{f"{field}_accuracy": rate(field) for field in FIELDS},
    }


def evaluate_retrieval_labels(labels: list[dict], traces: list[dict] | None = None) -> dict:
    """Score a label file against recorded traces, or against the scripted judge."""

    predictions = None if traces is None else _prediction_index(traces)
    pairs: list[tuple[dict, dict | None]] = []
    for label in labels:
        gold = {field: label[field] for field in FIELDS}
        if predictions is None:
            judged = scripted_retrieval_judgment(
                {
                    "symbol": label["symbol"],
                    "question": label["question"],
                    "attempt": label.get("attempt", 0),
                    "decision_cutoff": label["decision_cutoff"],
                    "news_lookback_days": label.get("news_lookback_days", 7),
                    "evidence": label.get("evidence", []),
                }
            )
            predicted = {field: judged[field] for field in FIELDS}
        else:
            found = predictions.get(_label_key(label))
            predicted = None if found is None else {field: found[field] for field in FIELDS}
        pairs.append((gold, predicted))
    return retrieval_accuracy(pairs)


def load_retrieval_labels(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("retrieval labels must be a list")
    return value


def load_retrieval_traces(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _prediction_index(traces: list[dict]) -> dict[tuple, dict]:
    index: dict[tuple, dict] = {}
    for trace in traces:
        judgment = trace.get("judgment")
        if trace.get("scope") != "company" or not isinstance(judgment, dict):
            continue
        key = (
            trace.get("symbol"),
            trace.get("question"),
            int(trace.get("attempt", 0)),
            tuple(sorted(trace.get("returned_evidence_ids") or [])),
        )
        index[key] = judgment
    return index


def _label_key(label: dict) -> tuple:
    evidence_ids = [item["evidence_id"] for item in label.get("evidence", [])]
    return (
        label["symbol"],
        label["question"],
        int(label.get("attempt", 0)),
        tuple(sorted(evidence_ids)),
    )
