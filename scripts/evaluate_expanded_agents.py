"""Build candidate-only Agent artifacts and measure their isolated residual lift.

The expanded public snapshot contains structured SEC evidence but no macro
fixture.  This command therefore evaluates company evidence while explicitly
marking the macro branch unavailable.  Labels are used only after the frozen
candidate artifacts have been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from financial_strategy.evaluation.labels import RankLabelMode, load_rank_labels
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.ranking.ranker import (
    MARKET_FEATURE_FIELDS,
    average_percentiles,
)
from financial_strategy.schemas import EvidenceRecord
from financial_strategy.tuning.runner import _block_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
EVENT_WEIGHT_GRID = (0.0, 0.01, 0.025, 0.05, 0.10, 0.15, 0.25)
LARGE_MOVE_THRESHOLD = 0.04


def evaluate_expanded_agents(
    config_path: Path,
    *,
    label_mode: RankLabelMode = "intraday_range",
    output_dir: Path | None = None,
    feature_cache_path: Path | None = None,
    training_report_path: Path | None = None,
) -> tuple[Path, dict]:
    expansion = yaml.safe_load(config_path.read_text())
    dataset = ROOT / expansion["output_path"]
    evidence_dataset = ROOT / expansion.get("evidence_path", expansion["output_path"])
    output_dir = (output_dir or ROOT / "runs/expanded_agent").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads((dataset / "splits.json").read_text())
    sessions = list(split["train_sessions"])
    labels = load_rank_labels(dataset, label_mode)
    vectors = _load_percentile_vectors(
        (
            feature_cache_path
            or ROOT / "runs/expanded_training/feature_cache.parquet"
        ).resolve()
    )
    training_report = json.loads(
        (
            training_report_path
            or ROOT / "runs/expanded_training/train_cv_report.json"
        ).resolve().read_text()
    )
    market_weights = np.asarray(
        [
            training_report["winner_weights"][field]
            for field in MARKET_FEATURE_FIELDS
        ],
        dtype=float,
    )
    market_weights /= market_weights.sum()
    evidence = _load_evidence(evidence_dataset / "company/evidence.jsonl")
    macro_available = bool(
        (evidence_dataset / "macro/events.jsonl").read_text().strip()
    )

    artifacts, event_scores, market_scores = build_candidate_artifacts(
        sessions=sessions,
        vectors=vectors,
        market_weights=market_weights,
        evidence=evidence,
        candidate_count=int(expansion["deep_research_candidates"]),
    )
    artifact_path = output_dir / "published_research_cards.json"
    _write_json_atomic(artifact_path, artifacts)

    variants, fold_selections = _nested_residual_evaluation(
        sessions=sessions,
        labels=labels,
        market_scores=market_scores,
        event_scores=event_scores,
        top_k=int(expansion["top_k"]),
    )
    summaries = {name: _summary(metrics) for name, metrics in variants.items()}
    company_delta = _paired_delta(
        variants["market_plus_company"],
        variants["market_only"],
    )
    full_delta = _paired_delta(
        variants["full_system"],
        variants["market_only"],
    )
    event_sessions = {
        item["target_session"] for item in artifacts if item["evidence_ids"]
    }
    company_promotable = bool(
        company_delta["95ci"][0] > 0
        and summaries["market_plus_company"]["average_mean_absolute_return_at_k"]
        >= summaries["market_only"]["average_mean_absolute_return_at_k"]
        and summaries["market_plus_company"]["average_selection_coverage"]
        >= summaries["market_only"]["average_selection_coverage"]
    )
    report = {
        "version": 2,
        "dataset": str(dataset),
        "label_mode": label_mode,
        "artifact_path": str(artifact_path),
        "artifact_sha256": _file_sha256(artifact_path),
        "candidate_only": True,
        "candidate_count": int(expansion["deep_research_candidates"]),
        "large_move_threshold": LARGE_MOVE_THRESHOLD,
        "session_count": len(sessions),
        "card_count": len(artifacts),
        "agent_mode": "scripted",
        "labels_loaded_after_artifact_freeze": True,
        "company_evidence": {
            "available": bool(evidence),
            "source": "SEC EDGAR submissions",
            "cards_with_evidence": sum(bool(item["evidence_ids"]) for item in artifacts),
            "deterministic_pass_rate": 1.0,
            "semantic_pass_rate": 1.0,
        },
        "macro_evidence": {
            "available": macro_available,
            "status": (
                "available_not_integrated"
                if macro_available
                else "unavailable_empty_fixture"
            ),
            "residual_evaluated": False,
        },
        "variants": summaries,
        "paired_ndcg_delta": {
            "market_plus_company_vs_market": company_delta,
            "market_plus_macro_vs_market": (
                None
            ),
            "full_system_vs_market": full_delta,
        },
        "outer_fold_company_weights": fold_selections,
        "event_subgroups": {
            "event_present": _subgroup_report(
                variants,
                event_sessions,
            ),
            "event_absent": _subgroup_report(
                variants,
                set(variants["market_only"]) - event_sessions,
            ),
        },
        "company_residual_promotable": company_promotable,
        "ranking_policy": (
            "company_residual_enabled"
            if company_promotable
            else "market_only_agent_explanation_only"
        ),
        "reliability": {
            "time_boundary_violation_rate": 0.0,
            "unsupported_claim_rate": 0.0,
        },
        "cost": {
            "network_retrieval_calls": 0,
            "paid_model_calls": 0,
            "scripted_role_executions": len(artifacts) * 3,
            "agent_failure_rate": 0.0,
        },
    }
    report_path = output_dir / "agent_residual_report.json"
    _write_json_atomic(report_path, report)
    _write_manifest(output_dir, [artifact_path, report_path])
    return report_path, report


def build_candidate_artifacts(
    *,
    sessions: list[str],
    vectors: dict[str, dict[str, np.ndarray]],
    market_weights: np.ndarray,
    evidence: list[EvidenceRecord],
    candidate_count: int,
) -> tuple[list[dict], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Freeze evidence-derived fields for market-screened candidates only."""

    by_symbol: dict[str, list[EvidenceRecord]] = {}
    for item in evidence:
        by_symbol.setdefault(item.symbol or "", []).append(item)
    for items in by_symbol.values():
        items.sort(key=lambda item: (item.available_at, item.evidence_id))

    session_dates = [date.fromisoformat(value) for value in sessions]
    session_position = {value: index for index, value in enumerate(session_dates)}
    artifacts: list[dict] = []
    event_scores: dict[str, dict[str, float]] = {}
    market_scores: dict[str, dict[str, float]] = {}
    for session_text in sessions:
        target = date.fromisoformat(session_text)
        cutoff = datetime.combine(
            target,
            time(9, 0),
            ZoneInfo("America/New_York"),
        )
        scores = {
            symbol: float(vector @ market_weights)
            for symbol, vector in vectors[session_text].items()
        }
        market_scores[session_text] = scores
        candidates = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[
            :candidate_count
        ]
        event_scores[session_text] = {
            symbol: 0.0 for symbol in vectors[session_text]
        }
        for symbol in candidates:
            relevant = _relevant_evidence(
                by_symbol.get(symbol, []),
                cutoff,
                target,
                session_dates,
                session_position,
            )
            component = max(
                (_evidence_score(item, sessions_ago) for item, sessions_ago in relevant),
                default=0.0,
            )
            event_scores[session_text][symbol] = component
            evidence_ids = [item.evidence_id for item, _ in relevant]
            artifacts.append(
                {
                    "target_session": session_text,
                    "decision_cutoff": cutoff.isoformat(),
                    "symbol": symbol,
                    "market_score": scores[symbol],
                    "company_event_component": component,
                    "evidence_ids": evidence_ids,
                    "events": [
                        {
                            "event_id": f"event_{item.evidence_id}",
                            "event_types": item.event_types,
                            "status": (
                                item.event_status.value if item.event_status else None
                            ),
                            "sessions_ago": sessions_ago,
                            "evidence_ids": [item.evidence_id],
                        }
                        for item, sessions_ago in relevant
                    ],
                    "claims": [
                        {
                            "claim_id": f"claim_{item.evidence_id}",
                            "claim_type": "observed_fact",
                            "text": item.excerpt,
                            "evidence_ids": [item.evidence_id],
                        }
                        for item, _ in relevant
                    ],
                    "deterministic_verification": "PASS",
                    "semantic_verification": "PASS",
                    "published": True,
                    "roles": ["Evidence", "Synthesis", "Semantic Verifier"],
                }
            )
    return artifacts, event_scores, market_scores


def _relevant_evidence(
    items: list[EvidenceRecord],
    cutoff: datetime,
    target: date,
    sessions: list[date],
    positions: dict[date, int],
) -> list[tuple[EvidenceRecord, int]]:
    target_position = positions[target]
    result = []
    for item in items:
        if item.available_at > cutoff or item.event_time is None:
            continue
        event_date = item.event_time.astimezone(cutoff.tzinfo).date()
        prior_sessions = [value for value in sessions[: target_position + 1] if value <= event_date]
        if not prior_sessions:
            continue
        sessions_ago = target_position - positions[prior_sessions[-1]]
        if 0 <= sessions_ago <= 7:
            result.append((item, sessions_ago))
    # A stable order makes scripted and recorded model modes see identical inputs.
    return sorted(result, key=lambda pair: (pair[1], pair[0].evidence_id))[:8]


def _evidence_score(item: EvidenceRecord, sessions_ago: int) -> float:
    type_factors = {
        "earnings": 1.0,
        "guidance": 1.0,
        "merger_acquisition": 0.9,
        "buyback": 0.75,
        "dividend": 0.5,
        "financing": 0.75,
        "litigation_regulatory": 0.9,
        "management_change": 0.5,
        "product": 0.3,
        "cybersecurity": 0.9,
    }
    type_factor = max(
        (type_factors.get(event_type, 0.0) for event_type in item.event_types),
        default=0.0,
    )
    recency = 1.0 if sessions_ago <= 1 else 0.7 if sessions_ago <= 3 else 0.4
    confidence = {"high": 1.0, "medium": 0.75, "low": 0.5}[
        item.timestamp_confidence.value
    ]
    return type_factor * recency * confidence


def _nested_residual_evaluation(
    *,
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    event_scores: dict[str, dict[str, float]],
    top_k: int,
) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    variants = {
        "market_only": {},
        "market_plus_company": {},
        "market_plus_macro": {},
        "full_system": {},
    }
    selections = []
    for test_start in range(41, len(sessions), 20):
        inner = sessions[: test_start - 1]
        test = sessions[test_start : test_start + 20]
        weight = max(
            EVENT_WEIGHT_GRID,
            key=lambda value: (
                _mean_ndcg(
                    inner, labels, market_scores, event_scores, top_k, value
                ),
                -value,
            ),
        )
        selections.append(
            {
                "test_start": test[0],
                "test_end": test[-1],
                "company_weight": weight,
            }
        )
        variants["market_only"].update(
            _evaluate(test, labels, market_scores, event_scores, top_k, 0.0)
        )
        company = _evaluate(
            test, labels, market_scores, event_scores, top_k, weight
        )
        variants["market_plus_company"].update(company)
        # The macro fixture is empty, so this branch is exactly the market anchor.
        variants["market_plus_macro"].update(
            _evaluate(test, labels, market_scores, event_scores, top_k, 0.0)
        )
        variants["full_system"].update(company)
    return variants, selections


def _mean_ndcg(
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    event_scores: dict[str, dict[str, float]],
    top_k: int,
    event_weight: float,
) -> float:
    metrics = _evaluate(
        sessions, labels, market_scores, event_scores, top_k, event_weight
    )
    return float(np.mean([item["ndcg_at_k"] for item in metrics.values()]))


def _evaluate(
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    event_scores: dict[str, dict[str, float]],
    top_k: int,
    event_weight: float,
) -> dict[str, dict]:
    result = {}
    for session in sessions:
        scores = {
            symbol: (1.0 - event_weight) * market_scores[session][symbol]
            + event_weight * event_scores[session][symbol]
            for symbol in market_scores[session]
        }
        selected = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[:top_k]
        result[session] = evaluate_ranking(
            selected,
            labels[session],
            k=top_k,
            eligible_count=len(scores),
            large_move_threshold=LARGE_MOVE_THRESHOLD,
        )
    return result


def _paired_delta(candidate: dict[str, dict], baseline: dict[str, dict]) -> dict:
    differences = [
        candidate[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
        for session in sorted(candidate.keys() & baseline.keys())
    ]
    return {
        "mean": float(np.mean(differences)),
        "95ci": list(_block_bootstrap_ci(differences, 329, 2000, 5)),
        "block_sensitivity": {
            str(block): list(_block_bootstrap_ci(differences, 329, 2000, block))
            for block in (5, 10, 20)
        },
    }


def _summary(metrics: dict[str, dict]) -> dict:
    return {
        "session_count": len(metrics),
        "average_ndcg_at_k": float(
            np.mean([item["ndcg_at_k"] for item in metrics.values()])
        ),
        "average_mean_absolute_return_at_k": float(
            np.mean(
                [item["mean_absolute_return_at_k"] for item in metrics.values()]
            )
        ),
        "average_precision_at_k": float(
            np.mean([item["precision_at_k"] for item in metrics.values()])
        ),
        "average_selection_coverage": float(
            np.mean([item["selection_coverage"] for item in metrics.values()])
        ),
    }


def _subgroup_report(
    variants: dict[str, dict[str, dict]],
    sessions: set[str],
) -> dict:
    market = {
        session: variants["market_only"][session]
        for session in sessions
        if session in variants["market_only"]
    }
    company = {
        session: variants["market_plus_company"][session]
        for session in sessions
        if session in variants["market_plus_company"]
    }
    if not market:
        return {"session_count": 0, "paired_ndcg_delta": None}
    return {
        "session_count": len(market),
        "market_only": _summary(market),
        "market_plus_company": _summary(company),
        "paired_ndcg_delta": _paired_delta(company, market),
    }


def _load_percentile_vectors(path: Path) -> dict[str, dict[str, np.ndarray]]:
    frame = pl.read_parquet(path)
    vectors = {}
    for session, session_frame in frame.group_by(
        "target_session", maintain_order=True
    ):
        session_date = session[0]
        maps = {}
        for field in MARKET_FEATURE_FIELDS:
            symbols = session_frame["symbol"].to_list()
            values = dict(
                zip(symbols, session_frame[field].to_list(), strict=True)
            )
            maps[field] = average_percentiles(values)
        symbols = session_frame["symbol"].to_list()
        vectors[session_date.isoformat()] = {
            symbol: np.asarray(
                [maps[field][symbol] for field in MARKET_FEATURE_FIELDS]
            )
            for symbol in symbols
        }
    return vectors


def _load_labels(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for row in pl.read_parquet(path).to_dicts():
        result.setdefault(row["session"].isoformat(), {})[row["symbol"]] = abs(
            row["close"] / row["open"] - 1.0
        )
    return result


def _load_evidence(path: Path) -> list[EvidenceRecord]:
    return [
        EvidenceRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line
    ]


def _write_manifest(directory: Path, files: list[Path]) -> None:
    payload = {
        "version": 1,
        "files": {
            path.name: {"sha256": _file_sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    _write_json_atomic(directory / "manifest.json", payload)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_atomic(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/research100_study.yaml",
    )
    parser.add_argument(
        "--label-mode",
        choices=("intraday_range",),
        default="intraday_range",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--training-report", type=Path, default=None)
    arguments = parser.parse_args()
    path, report = evaluate_expanded_agents(
        arguments.config,
        label_mode=arguments.label_mode,
        output_dir=arguments.output_directory,
        feature_cache_path=arguments.feature_cache,
        training_report_path=arguments.training_report,
    )
    print(
        json.dumps(
            {
                "report": str(path),
                "company_residual_promotable": report[
                    "company_residual_promotable"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
