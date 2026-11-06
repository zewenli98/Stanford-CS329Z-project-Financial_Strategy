"""Evaluate a separate point-in-time macro residual on train sessions only."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from financial_strategy.config import load_mvp_config
from financial_strategy.evaluation.labels import load_rank_labels
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.macro import build_macro_interactions
from financial_strategy.ranking.ranker import MARKET_FEATURE_FIELDS
from financial_strategy.schemas import (
    MacroEvidenceRecord,
    SymbolMacroExposure,
)
from financial_strategy.tuning.runner import _block_bootstrap_ci
from scripts.train_expanded import _load_feature_cache

ROOT = Path(__file__).resolve().parents[1]
EASTERN = ZoneInfo("America/New_York")
WEIGHTS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
LARGE_MOVE_THRESHOLD = 0.04


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _percentiles(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values, key=lambda symbol: (values[symbol], symbol))
    result = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = ((start + end - 1) / 2) / max(1, len(ordered) - 1)
        for symbol in ordered[start:end]:
            result[symbol] = rank
        start = end
    return result


def _metrics(
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    macro_scores: dict[str, dict[str, float]],
    weight: float,
) -> dict[str, dict]:
    result = {}
    for session in sessions:
        if session not in macro_scores:
            continue
        scores = {
            symbol: (1 - weight) * market_scores[session][symbol]
            + weight * macro_scores[session][symbol]
            for symbol in labels[session]
        }
        selected = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[:10]
        result[session] = evaluate_ranking(
            selected,
            labels[session],
            k=10,
            eligible_count=len(scores),
            large_move_threshold=LARGE_MOVE_THRESHOLD,
        )
    return result


def _summary(metrics: dict[str, dict]) -> dict:
    if not metrics:
        return {
            "session_count": 0,
            "average_ndcg_at_k": None,
            "average_mean_absolute_return_at_k": None,
            "average_precision_at_k": None,
            "average_selection_coverage": None,
        }
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


def _paired_delta(candidate: dict[str, dict], baseline: dict[str, dict]) -> dict:
    shared = sorted(candidate.keys() & baseline.keys())
    values = [
        candidate[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
        for session in shared
    ]
    if not values:
        return {"mean": None, "95ci": None}
    return {
        "mean": float(np.mean(values)),
        "95ci": _block_bootstrap_ci(
            values,
            seed=329,
            samples=2_000,
            block_size=min(5, len(values)),
        ),
    }


def _select_inner_weight(
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    macro_scores: dict[str, dict[str, float]],
) -> tuple[float, dict[str, float | None]]:
    scores = {}
    for weight in WEIGHTS:
        deltas = []
        for test_start in range(9, len(sessions), 4):
            inner_test = sessions[test_start : test_start + 4]
            if not sessions[: test_start - 1]:
                continue
            baseline = _metrics(
                inner_test, labels, market_scores, macro_scores, 0.0
            )
            challenger = _metrics(
                inner_test, labels, market_scores, macro_scores, weight
            )
            deltas.extend(
                challenger[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
                for session in sorted(challenger.keys() & baseline.keys())
            )
        scores[str(weight)] = float(np.mean(deltas)) if deltas else None
    available = {
        float(name): value for name, value in scores.items() if value is not None
    }
    selected = (
        max(available, key=lambda value: (available[value], -value))
        if available
        else 0.0
    )
    return selected, scores


def _nested_cv(
    event_sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    macro_scores: dict[str, dict[str, float]],
) -> dict:
    market_metrics = {}
    macro_metrics = {}
    folds = []
    for test_start in range(13, len(event_sessions), 5):
        outer_train = event_sessions[: test_start - 1]
        outer_test = event_sessions[test_start : test_start + 5]
        weight, inner_scores = _select_inner_weight(
            outer_train, labels, market_scores, macro_scores
        )
        baseline = _metrics(
            outer_test, labels, market_scores, macro_scores, 0.0
        )
        challenger = _metrics(
            outer_test, labels, market_scores, macro_scores, weight
        )
        shared = sorted(baseline.keys() & challenger.keys())
        if not shared:
            continue
        market_metrics.update(baseline)
        macro_metrics.update(challenger)
        fold_delta = float(
            np.mean(
                [
                    challenger[session]["ndcg_at_k"]
                    - baseline[session]["ndcg_at_k"]
                    for session in shared
                ]
            )
        )
        folds.append(
            {
                "test_sessions": shared,
                "selected_weight": weight,
                "inner_scores": inner_scores,
                "paired_ndcg_delta": fold_delta,
            }
        )
    delta = _paired_delta(macro_metrics, market_metrics)
    shared_sessions = sorted(market_metrics.keys() & macro_metrics.keys())
    overlap = [
        len(
            set(market_metrics[session]["selected_symbols"])
            & set(macro_metrics[session]["selected_symbols"])
        )
        / 10
        for session in shared_sessions
    ]
    return {
        "fold_count": len(folds),
        "folds": folds,
        "market_only": _summary(market_metrics),
        "market_plus_macro": _summary(macro_metrics),
        "paired_ndcg_delta": delta,
        "top_k_overlap_with_market": (
            float(np.mean(overlap)) if overlap else None
        ),
        "winning_folds": sum(item["paired_ndcg_delta"] > 0 for item in folds),
        "minimum_fold_delta": min(
            (item["paired_ndcg_delta"] for item in folds), default=None
        ),
    }


def run_experiment(
    macro_fixture: Path,
    output: Path,
    *,
    minimum_history: int = 8,
) -> tuple[Path, dict]:
    if output.exists():
        shutil.rmtree(output)
    events = [
        MacroEvidenceRecord.model_validate_json(line)
        for line in (macro_fixture / "events.jsonl").read_text().splitlines()
        if line
    ]
    dataset = ROOT / "data/research100_v1"
    vectors, labels, _ = _load_feature_cache(
        dataset,
        ROOT / "runs/research100_training/feature_cache.parquet",
    )
    split = json.loads((dataset / "splits.json").read_text())
    train_sessions = list(split["train_sessions"])
    train_set = set(train_sessions)
    signed_returns = {
        session: values
        for session, values in load_rank_labels(dataset).items()
        if session in train_set
    }

    universe = json.loads((dataset / "universe/symbols.json").read_text())
    sectors = {item["symbol"]: item["sector"] for item in universe["securities"]}
    exposure_config = json.loads(
        (ROOT / "configs/sector_exposures.json").read_text()
    )
    exposures = {
        symbol: SymbolMacroExposure(
            symbol=symbol,
            sector=sector,
            exposure_method="sector_heuristic_v1",
            template_version=exposure_config["version"],
            exposures=exposure_config["sectors"][sector],
        )
        for symbol, sector in sectors.items()
    }
    config = load_mvp_config(ROOT / "configs/screening_research100_candidate.yaml")
    if config.ranking.market_feature_weights is None:
        raise ValueError(
            "macro residual evaluation requires market_feature_weights; "
            "use configs/screening_research100_candidate.yaml"
        )
    weights = np.asarray(
        [
            config.ranking.market_feature_weights[field]
            for field in MARKET_FEATURE_FIELDS
        ]
    )
    market_scores = {
        session: {
            symbol: float(vector @ weights)
            for symbol, vector in vectors[session].items()
        }
        for session in train_sessions
    }

    by_session: dict[str, list[MacroEvidenceRecord]] = {}
    for event in events:
        if event.event_time is None:
            continue
        session = event.event_time.date().isoformat()
        if session not in train_set:
            continue
        cutoff = datetime.combine(
            event.event_time.date(), time(9, 0), EASTERN
        ).astimezone(event.event_time.tzinfo)
        if event.available_at <= cutoff:
            by_session.setdefault(session, []).append(event)

    macro_scores = {}
    macro_scores_by_group: dict[str, dict[str, dict[str, float]]] = {
        "cpi": {},
        "employment": {},
        "fomc": {},
    }
    artifacts = []
    interaction_symbols = 0
    total_symbols = 0
    for session in sorted(by_session):
        interactions = build_macro_interactions(
            target_session=date.fromisoformat(session),
            target_events=by_session[session],
            history_events=events,
            returns=signed_returns,
            sectors=sectors,
            exposures=exposures,
            surprise_scales=config.ranking.macro.surprise_scales,
            minimum_symbol_events=minimum_history,
            minimum_sector_events=minimum_history,
            minimum_global_events=minimum_history,
        )
        raw = {
            symbol: max(
                (item.expected_abs_impact for item in values),
                default=0.0,
            )
            for symbol, values in interactions.items()
        }
        macro_scores[session] = _percentiles(raw)
        grouped_types = {
            "cpi": {"cpi_yoy"},
            "employment": {"nonfarm_payrolls", "unemployment"},
            "fomc": {"fed_rate"},
        }
        for group, event_types in grouped_types.items():
            if not any(
                event.event_type in event_types for event in by_session[session]
            ):
                continue
            grouped_raw = {
                symbol: max(
                    (
                        item.expected_abs_impact
                        for item in values
                        if item.event_type in event_types
                    ),
                    default=0.0,
                )
                for symbol, values in interactions.items()
            }
            macro_scores_by_group[group][session] = _percentiles(grouped_raw)
        total_symbols += len(raw)
        interaction_symbols += sum(
            any(item.sensitivity.fallback_level != "zero" for item in values)
            for values in interactions.values()
        )
        artifacts.append(
            {
                "target_session": session,
                "evidence_ids": [
                    item.evidence_id for item in by_session[session]
                ],
                "assessment_mode": "deterministic_normalized_macro_contract",
                "interactions": {
                    symbol: [
                        item.model_dump(mode="json") for item in values
                    ]
                    for symbol, values in interactions.items()
                    if values
                },
            }
        )

    event_sessions = sorted(macro_scores)
    nested = _nested_cv(
        event_sessions, labels, market_scores, macro_scores
    )
    event_group_cv = {
        group: _nested_cv(
            sorted(scores),
            labels,
            market_scores,
            scores,
        )
        for group, scores in macro_scores_by_group.items()
    }
    audit = json.loads((macro_fixture / "audit.json").read_text())
    counts = Counter(
        item.event_type
        for item in events
        if item.status == "released_before_cutoff"
    )
    interaction_coverage = (
        interaction_symbols / total_symbols if total_symbols else 0.0
    )
    released_count = sum(counts.values())
    primary_pit_coverage = (
        audit["primary_eligible_count"] / released_count
        if released_count
        else 0.0
    )
    data_gates = {
        "cpi_yoy_at_least_24": counts["cpi_yoy"] >= 24,
        "employment_at_least_24": min(
            counts["nonfarm_payrolls"], counts["unemployment"]
        )
        >= 24,
        "fed_rate_at_least_18": counts["fed_rate"] >= 18,
        "event_sessions_at_least_40": len(event_sessions) >= 40,
        "interaction_coverage_at_least_90pct": interaction_coverage >= 0.90,
        "primary_pit_coverage_at_least_95pct": primary_pit_coverage >= 0.95,
    }
    ci = nested["paired_ndcg_delta"]["95ci"]
    performance_gates = {
        "ndcg_ci_lower_above_zero": ci is not None and ci[0] > 0,
        "mean_absolute_return_non_degrading": (
            nested["fold_count"] > 0
            and nested["market_plus_macro"]["average_mean_absolute_return_at_k"]
            >= nested["market_only"]["average_mean_absolute_return_at_k"]
        ),
        "coverage_non_degrading": (
            nested["fold_count"] > 0
            and nested["market_plus_macro"]["average_selection_coverage"]
            >= nested["market_only"]["average_selection_coverage"]
        ),
        "majority_winning_folds": (
            nested["fold_count"] > 0
            and nested["winning_folds"] > nested["fold_count"] / 2
        ),
        "no_fold_below_minus_002": (
            nested["minimum_fold_delta"] is not None
            and nested["minimum_fold_delta"] >= -0.02
        ),
    }
    report = {
        "version": 1,
        "macro_fixture": str(macro_fixture),
        "labels_scope": "train_only",
        "future_holdout_consumed": False,
        "released_event_counts": dict(sorted(counts.items())),
        "event_session_count": len(event_sessions),
        "interaction_coverage": interaction_coverage,
        "primary_pit_coverage": primary_pit_coverage,
        "minimum_history_releases": minimum_history,
        "fixture_audit": audit,
        "data_quality_gates": data_gates,
        "nested_cv": nested,
        "event_group_cv": event_group_cv,
        "performance_gates": performance_gates,
        "secondary_assumed_pit_promotable": all(performance_gates.values()),
        "primary_promotable": minimum_history >= 8
        and all(data_gates.values())
        and all(performance_gates.values()),
        "company_channel_status": "not_retrained_in_macro_isolation_experiment",
    }
    output.mkdir(parents=True)
    artifact_path = output / "macro_interactions.json"
    _atomic_json(artifact_path, artifacts)
    report["artifact_sha256"] = hashlib.sha256(
        artifact_path.read_bytes()
    ).hexdigest()
    report_path = output / "macro_residual_report.json"
    _atomic_json(report_path, report)
    return report_path, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--macro-fixture",
        type=Path,
        default=ROOT / "data/macro_current",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/macro_residual_current",
    )
    parser.add_argument("--minimum-history", type=int, default=8)
    arguments = parser.parse_args()
    report_path, report = run_experiment(
        arguments.macro_fixture.resolve(),
        arguments.output.resolve(),
        minimum_history=arguments.minimum_history,
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "primary_promotable": report["primary_promotable"],
                "secondary_assumed_pit_promotable": report[
                    "secondary_assumed_pit_promotable"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
