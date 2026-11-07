"""Train and evaluate a signed LLM residual over the frozen market champion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from financial_strategy.config import canonical_json, load_mvp_config
from financial_strategy.evaluation.labels import RankLabelMode
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.ranking.ranker import (
    LLM_RESIDUAL_FEATURE_FIELDS,
    MARKET_FEATURE_FIELDS,
    calibrate_market_scores,
    llm_residual_features,
)
from financial_strategy.schemas import ResearchCard
from financial_strategy.tuning.learning import fit_signed_ridge_residual
from financial_strategy.tuning.runner import _block_bootstrap_ci
from scripts.train_expanded import _load_feature_cache

ROOT = Path(__file__).resolve().parents[1]
SHRINKAGES = (0.1, 1.0, 10.0)
RESIDUAL_SCALES = (0.0, 0.25, 0.5, 0.75, 1.0)
LARGE_MOVE_THRESHOLD = 0.04
MARKET_SCORE_CALIBRATION = "cross_sectional_percentile"


def train_residual(
    artifact_path: Path,
    output: Path | None = None,
    candidate_path: Path | None = None,
    *,
    label_mode: RankLabelMode = "intraday_range",
    base_config_path: Path | None = None,
    feature_cache_path: Path | None = None,
) -> tuple[Path, dict]:
    artifact_path = artifact_path.resolve()
    output = (output or artifact_path.parent).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "residual_cv_report.json"
    candidate_path = (
        candidate_path or output / "screening_llm_v3_candidate.yaml"
    ).resolve()
    report_path.unlink(missing_ok=True)
    candidate_path.unlink(missing_ok=True)
    manifest_path = (
        artifact_path.parent
        / f"manifest_{artifact_path.stem.removeprefix('artifacts_')}.json"
    )
    _verify_artifact_manifest(manifest_path, artifact_path)
    raw_artifacts = json.loads(artifact_path.read_text())
    artifacts = [
        {
            **item,
            "card_model": ResearchCard.model_validate(item["card"]),
        }
        for item in raw_artifacts
    ]
    # Labels are intentionally opened only after the immutable artifact hash check.
    dataset = ROOT / "data/2023-11-01_2026-10-30"
    base_config_path = (
        base_config_path or ROOT / "configs/screening_research100_candidate.yaml"
    ).resolve()
    feature_cache_path = (
        feature_cache_path
        or ROOT / "runs/expanded_training/feature_cache.parquet"
    ).resolve()
    vectors, labels, _diagnostics = _load_feature_cache(
        dataset,
        feature_cache_path,
        label_mode,
    )
    split = json.loads((dataset / "splits.json").read_text())
    sectors = {
        item["symbol"]: item["sector"]
        for item in json.loads(
            (dataset / "universe/symbols.json").read_text()
        )["securities"]
    }
    sessions = list(split["train_sessions"])
    config = load_mvp_config(base_config_path)
    if config.ranking.market_feature_weights is None:
        raise ValueError(
            "residual training requires market_feature_weights; "
            "use configs/screening_research100_candidate.yaml"
        )
    market_weights = np.asarray(
        [config.ranking.market_feature_weights[field] for field in MARKET_FEATURE_FIELDS]
    )
    raw_market_scores = {
        session: {
            symbol: float(vector @ market_weights)
            for symbol, vector in vectors[session].items()
        }
        for session in sessions
    }
    market_scores = {
        session: calibrate_market_scores(scores, MARKET_SCORE_CALIBRATION)
        for session, scores in raw_market_scores.items()
    }
    by_session: dict[str, dict[str, dict]] = {}
    for item in artifacts:
        by_session.setdefault(item["target_session"], {})[item["symbol"]] = item

    variants = {}
    for shrinkage in SHRINKAGES:
        market_metrics = {}
        residual_metrics = {
            "market_plus_company": {},
            "market_plus_macro": {},
            "full_system": {},
        }
        fold_deltas = []
        for test_start in range(41, len(sessions), 20):
            inner = sessions[: test_start - 1]
            test = sessions[test_start : test_start + 20]
            rows, targets = _training_rows(
                inner,
                by_session,
                labels,
                market_scores,
            )
            if len(rows) < 20:
                continue
            intercept, coefficients = fit_signed_ridge_residual(
                rows,
                targets,
                shrinkage=shrinkage,
                coefficient_bound=0.25,
            )
            baseline = _evaluate_sessions(
                test,
                labels,
                market_scores,
                by_session,
                intercept=0.0,
                coefficients=np.zeros(len(LLM_RESIDUAL_FEATURE_FIELDS)),
                channel="none",
            )
            challengers = {
                name: _evaluate_sessions(
                    test,
                    labels,
                    market_scores,
                    by_session,
                    intercept=intercept,
                    coefficients=coefficients,
                    channel=channel,
                )
                for name, channel in (
                    ("market_plus_company", "company"),
                    ("market_plus_macro", "macro"),
                    ("full_system", "full"),
                )
            }
            challenger = challengers["full_system"]
            if not challenger:
                continue
            market_metrics.update(baseline)
            for name, metrics in challengers.items():
                residual_metrics[name].update(metrics)
            fold_deltas.append(
                float(
                    np.mean(
                        [
                            challenger[session]["ndcg_at_k"]
                            - baseline[session]["ndcg_at_k"]
                            for session in challenger
                        ]
                    )
                )
            )
        variants[str(shrinkage)] = {
            "shrinkage": shrinkage,
            "market_only": _summary(market_metrics),
            **{
                name: _summary(metrics)
                for name, metrics in residual_metrics.items()
            },
            "paired_ndcg_delta": {
                name: _paired_delta(metrics, market_metrics)
                for name, metrics in residual_metrics.items()
            },
            "top_k_overlap_with_market": _top_k_overlap(
                residual_metrics["full_system"],
                market_metrics,
            ),
            "industry_subgroups": _industry_subgroups(
                residual_metrics["full_system"],
                labels,
                sectors,
            ),
            "fold_deltas": fold_deltas,
            "winning_folds": sum(value > 0 for value in fold_deltas),
            "fold_count": len(fold_deltas),
        }
    eligible_variants = {
        name: value
        for name, value in variants.items()
        if value["fold_count"] > 0
        and value["paired_ndcg_delta"]["full_system"]["95ci"] is not None
    }
    diagnostic_winner = (
        max(
            eligible_variants,
            key=lambda name: (
                eligible_variants[name]["winning_folds"],
                min(eligible_variants[name]["fold_deltas"]),
                eligible_variants[name]["paired_ndcg_delta"]["full_system"]["mean"],
            ),
        )
        if eligible_variants
        else None
    )
    nested_cv = _nested_cv(
        sessions,
        by_session,
        labels,
        market_scores,
    )
    selected_counts = Counter(
        (
            str(item["selected_shrinkage"]),
            str(item["selected_scale"]),
        )
        for item in nested_cv["folds"]
    )
    winner_pair = (
        max(
            selected_counts,
            key=lambda pair: (
                selected_counts[pair],
                -float(pair[1]),
                -float(pair[0]),
            ),
        )
        if selected_counts
        else (
            (diagnostic_winner, "1.0")
            if diagnostic_winner is not None
            else None
        )
    )
    winner = winner_pair[0] if winner_pair is not None else None
    winner_scale = float(winner_pair[1]) if winner_pair is not None else 0.0
    all_rows, all_targets = _training_rows(
        sessions,
        by_session,
        labels,
        market_scores,
    )
    if winner is not None and len(all_rows) >= 20:
        final_intercept, final_coefficients = fit_signed_ridge_residual(
            all_rows,
            all_targets,
            shrinkage=float(winner),
            coefficient_bound=0.25,
        )
        final_intercept *= winner_scale
        final_coefficients *= winner_scale
    else:
        final_intercept = 0.0
        final_coefficients = np.zeros(len(LLM_RESIDUAL_FEATURE_FIELDS))
    coefficients = dict(
        zip(
            LLM_RESIDUAL_FEATURE_FIELDS,
            final_coefficients.tolist(),
            strict=True,
        )
    )
    schema_hash = hashlib.sha256(
        canonical_json(list(LLM_RESIDUAL_FEATURE_FIELDS))
    ).hexdigest()
    model_hash = hashlib.sha256(
        canonical_json(
            {
                "intercept": final_intercept,
                "coefficients": coefficients,
                "market_score_calibration": MARKET_SCORE_CALIBRATION,
                "feature_schema_hash": schema_hash,
            }
        )
    ).hexdigest()
    promotable = False
    promotion_gates = {
        "has_outer_folds": nested_cv["fold_count"] > 0,
        "ndcg_ci_lower_above_zero": False,
        "mean_absolute_return_non_degrading": False,
        "coverage_non_degrading": False,
        "majority_winning_folds": False,
        "no_fold_below_minus_002": False,
    }
    if nested_cv["fold_count"] > 0:
        ci = nested_cv["paired_ndcg_delta"]["full_system"]["95ci"]
        promotion_gates.update(
            {
                "ndcg_ci_lower_above_zero": bool(
                    ci is not None and ci[0] > 0
                ),
                "mean_absolute_return_non_degrading": bool(
                    nested_cv["full_system"][
                        "average_mean_absolute_return_at_k"
                    ]
                    >= nested_cv["market_only"][
                        "average_mean_absolute_return_at_k"
                    ]
                ),
                "coverage_non_degrading": bool(
                    nested_cv["full_system"]["average_selection_coverage"]
                    >= nested_cv["market_only"]["average_selection_coverage"]
                ),
                "majority_winning_folds": (
                    nested_cv["winning_folds"] > nested_cv["fold_count"] / 2
                ),
                "no_fold_below_minus_002": (
                    min(
                        item["paired_ndcg_delta"]
                        for item in nested_cv["folds"]
                    )
                    > -0.02
                ),
            }
        )
        promotable = all(promotion_gates.values())
    encodable_cards = [
        item
        for item in artifacts
        if item["card_model"].eligibility.eligible
        and llm_residual_features(item["card_model"])
    ]
    report = {
        "version": 2,
        "label_mode": label_mode,
        "base_config_path": str(base_config_path),
        "feature_cache_path": str(feature_cache_path),
        "artifact_path": str(artifact_path),
        "artifact_sha256": _file_hash(artifact_path),
        "labels_loaded_after_artifact_freeze": True,
        "artifact_card_count": len(artifacts),
        "verified_card_count": sum(
            item["card_model"].eligibility.eligible for item in artifacts
        ),
        "encodable_card_count": len(encodable_cards),
        "event_card_count": sum(
            bool(item["card_model"].events) for item in artifacts
        ),
        "absolute_move_direction_correction_count": sum(
            item["card_model"].catalyst_assessment is not None
            and item["card_model"].catalyst_assessment.surprise_direction == "negative"
            and item[
                "card_model"
            ].catalyst_assessment.signed_residual_bucket
            in {"demote", "strong_demote"}
            for item in artifacts
        ),
        "macro_interaction_card_count": sum(
            bool(item["card_model"].macro_interactions) for item in artifacts
        ),
        "agent_signal_session_count": len(
            {item["target_session"] for item in encodable_cards}
        ),
        "card_funnel": _card_funnel(artifacts),
        "residual_training_row_count": len(all_rows),
        "winner": winner,
        "winner_scale": winner_scale,
        "diagnostic_global_winner": diagnostic_winner,
        "variants": variants,
        "nested_cv": nested_cv,
        "final_model": {
            "intercept": final_intercept,
            "coefficients": coefficients,
            "market_score_calibration": MARKET_SCORE_CALIBRATION,
            "feature_schema_hash": schema_hash,
            "model_hash": model_hash,
        },
        "macro_status": (
            "available"
            if any(item.get("macro_evidence_ids") for item in artifacts)
            else "unavailable_or_not_extracted"
        ),
        "promotable": promotable,
        "promotion_gates": promotion_gates,
        "ranking_policy": (
            "market_plus_signed_llm_residual"
            if promotable
            else "market_only_llm_explanation_only"
        ),
    }
    _atomic_json(report_path, report)
    _write_candidate_config(
        candidate_path,
        config,
        report,
        artifact_path.parent,
    )
    return report_path, report


def _card_funnel(artifacts: list[dict]) -> dict:
    verdicts: Counter[str] = Counter()
    deterministic_issues: Counter[str] = Counter()
    semantic_issues: Counter[str] = Counter()
    untrainable_reasons: Counter[str] = Counter()
    for item in artifacts:
        card = item["card_model"]
        deterministic = card.deterministic_verification
        semantic = card.semantic_verification
        verdicts[
            "|".join(
                (
                    deterministic.verdict.value
                    if deterministic is not None
                    else "NONE",
                    semantic.verdict.value if semantic is not None else "NONE",
                    "eligible" if card.eligibility.eligible else "ineligible",
                )
            )
        ] += 1
        if deterministic is not None:
            deterministic_issues.update(
                issue.code.value for issue in deterministic.issues
            )
        if semantic is not None:
            semantic_issues.update(issue.code.value for issue in semantic.issues)
        if card.eligibility.eligible and not llm_residual_features(card):
            assessment = card.catalyst_assessment
            if assessment is None:
                untrainable_reasons["assessment_none"] += 1
            else:
                if not assessment.evidence_ids:
                    untrainable_reasons["no_evidence_ids"] += 1
                if assessment.target_session_relevance == "none":
                    untrainable_reasons["relevance_none"] += 1
    return {
        "verdicts": dict(sorted(verdicts.items())),
        "deterministic_issue_counts": dict(
            sorted(deterministic_issues.items())
        ),
        "semantic_issue_counts": dict(sorted(semantic_issues.items())),
        "eligible_untrainable_reason_counts": dict(
            sorted(untrainable_reasons.items())
        ),
    }


def _training_rows(
    sessions: list[str],
    by_session: dict[str, dict[str, dict]],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for session in sessions:
        session_cards = by_session.get(session, {})
        if not session_cards:
            continue
        symbols = sorted(labels[session])
        values = np.asarray([labels[session][symbol] for symbol in symbols])
        order = values.argsort(kind="stable")
        percentiles = np.empty(len(values), dtype=float)
        percentiles[order] = np.arange(len(values)) / max(1, len(values) - 1)
        target_by_symbol = dict(zip(symbols, percentiles, strict=True))
        for symbol, artifact in sorted(session_cards.items()):
            card = artifact["card_model"]
            if not card.eligibility.eligible:
                continue
            encoded = llm_residual_features(card)
            if not encoded:
                continue
            rows.append([encoded[name] for name in LLM_RESIDUAL_FEATURE_FIELDS])
            targets.append(target_by_symbol[symbol] - market_scores[session][symbol])
    return (
        np.asarray(rows, dtype=float).reshape(-1, len(LLM_RESIDUAL_FEATURE_FIELDS)),
        np.asarray(targets, dtype=float),
    )


def _nested_cv(
    sessions: list[str],
    by_session: dict[str, dict[str, dict]],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
) -> dict:
    """Select shrinkage inside each outer train window, then score future sessions."""

    market_metrics = {}
    residual_metrics = {
        "market_plus_company": {},
        "market_plus_macro": {},
        "full_system": {},
    }
    folds = []
    for test_start in range(41, len(sessions), 20):
        # The immediately preceding session is an embargo and belongs to
        # neither the expanding outer train window nor its future test block.
        outer_train = sessions[: test_start - 1]
        outer_test = sessions[test_start : test_start + 20]
        rows, targets = _training_rows(
            outer_train,
            by_session,
            labels,
            market_scores,
        )
        if len(rows) < 20:
            continue
        selected_shrinkage, selected_scale, inner_scores = (
            _select_inner_hyperparameters(
                outer_train,
                by_session,
                labels,
                market_scores,
            )
        )
        intercept, coefficients = fit_signed_ridge_residual(
            rows,
            targets,
            shrinkage=selected_shrinkage,
            coefficient_bound=0.25,
        )
        intercept *= selected_scale
        coefficients *= selected_scale
        baseline = _evaluate_sessions(
            outer_test,
            labels,
            market_scores,
            by_session,
            intercept=0.0,
            coefficients=np.zeros(len(LLM_RESIDUAL_FEATURE_FIELDS)),
            channel="none",
        )
        challengers = {
            name: _evaluate_sessions(
                outer_test,
                labels,
                market_scores,
                by_session,
                intercept=intercept,
                coefficients=coefficients,
                channel=channel,
            )
            for name, channel in (
                ("market_plus_company", "company"),
                ("market_plus_macro", "macro"),
                ("full_system", "full"),
            )
        }
        if not challengers["full_system"]:
            continue
        market_metrics.update(baseline)
        for name, metrics in challengers.items():
            residual_metrics[name].update(metrics)
        shared = sorted(
            challengers["full_system"].keys() & baseline.keys()
        )
        folds.append(
            {
                "test_start": outer_test[0],
                "test_end": outer_test[-1],
                "evaluated_sessions": shared,
                "selected_shrinkage": selected_shrinkage,
                "selected_scale": selected_scale,
                "inner_scores": inner_scores,
                "paired_ndcg_delta": float(
                    np.mean(
                        [
                            challengers["full_system"][session]["ndcg_at_k"]
                            - baseline[session]["ndcg_at_k"]
                            for session in shared
                        ]
                    )
                ),
            }
        )
    return {
        "market_only": _summary(market_metrics),
        **{
            name: _summary(metrics)
            for name, metrics in residual_metrics.items()
        },
        "paired_ndcg_delta": {
            name: _paired_delta(metrics, market_metrics)
            for name, metrics in residual_metrics.items()
        },
        "top_k_overlap_with_market": _top_k_overlap(
            residual_metrics["full_system"],
            market_metrics,
        ),
        "folds": folds,
        "fold_count": len(folds),
        "winning_folds": sum(
            item["paired_ndcg_delta"] > 0 for item in folds
        ),
    }


def _select_inner_hyperparameters(
    sessions: list[str],
    by_session: dict[str, dict[str, dict]],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
) -> tuple[float, float, dict[str, float | None]]:
    scores: dict[str, float | None] = {}
    for shrinkage in SHRINKAGES:
        for scale in RESIDUAL_SCALES:
            deltas = []
            for test_start in range(21, len(sessions), 20):
                inner_train = sessions[: test_start - 1]
                inner_test = sessions[test_start : test_start + 20]
                rows, targets = _training_rows(
                    inner_train,
                    by_session,
                    labels,
                    market_scores,
                )
                if len(rows) < 20:
                    continue
                intercept, coefficients = fit_signed_ridge_residual(
                    rows,
                    targets,
                    shrinkage=shrinkage,
                    coefficient_bound=0.25,
                )
                baseline = _evaluate_sessions(
                    inner_test,
                    labels,
                    market_scores,
                    by_session,
                    intercept=0.0,
                    coefficients=np.zeros(len(LLM_RESIDUAL_FEATURE_FIELDS)),
                    channel="none",
                )
                challenger = _evaluate_sessions(
                    inner_test,
                    labels,
                    market_scores,
                    by_session,
                    intercept=intercept * scale,
                    coefficients=coefficients * scale,
                    channel="full",
                )
                shared = sorted(challenger.keys() & baseline.keys())
                if shared:
                    deltas.extend(
                        challenger[session]["ndcg_at_k"]
                        - baseline[session]["ndcg_at_k"]
                        for session in shared
                    )
            scores[f"{shrinkage}:{scale}"] = (
                float(np.mean(deltas)) if deltas else None
            )
    available = {
        tuple(map(float, name.split(":"))): value
        for name, value in scores.items()
        if value is not None
    }
    selected = (
        max(
            available,
            key=lambda pair: (
                available[pair],
                -pair[1],
                -pair[0],
            ),
        )
        if available
        else (1.0, 0.0)
    )
    return selected[0], selected[1], scores


def _evaluate_sessions(
    sessions: list[str],
    labels: dict[str, dict[str, float]],
    market_scores: dict[str, dict[str, float]],
    by_session: dict[str, dict[str, dict]],
    *,
    intercept: float,
    coefficients: np.ndarray,
    channel: str,
) -> dict[str, dict]:
    result = {}
    for session in sessions:
        if session not in by_session:
            continue
        scores = dict(market_scores[session])
        for symbol, artifact in by_session[session].items():
            card = artifact["card_model"]
            if not card.eligibility.eligible:
                continue
            if channel == "none":
                continue
            company_residual = 0.0
            if channel in {"company", "full"} and artifact.get(
                "company_evidence_ids"
            ):
                encoded = llm_residual_features(card)
                # Keep evaluation identical to production scoring: a card
                # without a complete company assessment contributes no residual.
                if encoded:
                    vector = np.asarray(
                        [
                            encoded.get(name, 0.0)
                            for name in LLM_RESIDUAL_FEATURE_FIELDS
                        ]
                    )
                    company_residual = float(
                        np.clip(intercept + vector @ coefficients, -0.15, 0.15)
                    )
            # Production has no trained signed macro residual yet. Keep this at
            # zero so a nonnegative heuristic cannot bypass residual calibration.
            macro_residual = 0.0
            residual = company_residual + macro_residual
            if residual == 0.0:
                continue
            scores[symbol] = float(np.clip(scores[symbol] + residual, 0.0, 1.0))
        selected = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[:10]
        result[session] = evaluate_ranking(
            selected,
            labels[session],
            k=10,
            eligible_count=len(scores),
            large_move_threshold=LARGE_MOVE_THRESHOLD,
        )
    return result


def _paired_delta(candidate: dict[str, dict], baseline: dict[str, dict]) -> dict:
    differences = [
        candidate[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
        for session in sorted(candidate.keys() & baseline.keys())
    ]
    if not differences:
        return {"mean": None, "95ci": None, "block_sensitivity": {}}
    return {
        "mean": float(np.mean(differences)),
        "95ci": list(_block_bootstrap_ci(differences, 329, 2_000, 5)),
        "block_sensitivity": {
            str(block): list(
                _block_bootstrap_ci(differences, 329, 2_000, block)
            )
            for block in (5, 10, 20)
        },
    }


def _summary(metrics: dict[str, dict]) -> dict:
    names = (
        "ndcg_at_k",
        "mean_absolute_return_at_k",
        "precision_at_k",
        "selection_coverage",
    )
    result = {"session_count": len(metrics), "monthly": {}}
    for name in names:
        values = [item[name] for item in metrics.values() if item[name] is not None]
        result[f"average_{name}"] = float(np.mean(values)) if values else None
    for month in sorted({session[:7] for session in metrics}):
        monthly = [
            item for session, item in metrics.items() if session.startswith(month)
        ]
        result["monthly"][month] = {
            name: float(
                np.mean([item[name] for item in monthly if item[name] is not None])
            )
            for name in names
        }
    return result


def _top_k_overlap(candidate: dict[str, dict], market: dict[str, dict]) -> dict:
    values = []
    for session in sorted(candidate.keys() & market.keys()):
        left = set(candidate[session]["selected_symbols"])
        right = set(market[session]["selected_symbols"])
        values.append(len(left & right) / max(1, len(left)))
    return {
        "mean_fraction": float(np.mean(values)) if values else None,
        "95ci": (
            list(_block_bootstrap_ci(values, 329, 2_000, 5))
            if values
            else None
        ),
    }


def _industry_subgroups(
    metrics: dict[str, dict],
    labels: dict[str, dict[str, float]],
    sectors: dict[str, str],
) -> dict:
    grouped: dict[str, list[float]] = {}
    for session, item in metrics.items():
        for symbol in item["selected_symbols"]:
            grouped.setdefault(sectors[symbol], []).append(labels[session][symbol])
    return {
        sector: {
            "selection_count": len(values),
            "mean_absolute_return": float(np.mean(values)),
            "large_move_rate": float(
                np.mean([value >= LARGE_MOVE_THRESHOLD for value in values])
            ),
        }
        for sector, values in sorted(grouped.items())
    }


def _verify_artifact_manifest(manifest_path: Path, artifact_path: Path) -> None:
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest["files"].items():
        path = manifest_path.parent / relative
        if not path.exists() or expected != _file_hash(path):
            raise ValueError(f"LLM artifact manifest mismatch: {relative}")
    if artifact_path.name not in manifest["files"]:
        raise ValueError("LLM artifact manifest omits card artifact")


def _write_candidate_config(
    path: Path,
    config: object,
    report: dict,
    artifact_directory: Path,
) -> None:
    payload = config.model_dump(mode="json")
    model = report["final_model"]
    payload["ranking"]["residual_model"] = {
        "enabled": report["promotable"],
        "version": "llm_residual_v3_calibrated",
        "market_score_calibration": (
            model["market_score_calibration"] if report["promotable"] else "none"
        ),
        "coefficients": model["coefficients"] if report["promotable"] else {},
        "intercept": model["intercept"] if report["promotable"] else 0.0,
        "max_abs_residual": 0.15,
        "feature_schema_hash": (
            model["feature_schema_hash"] if report["promotable"] else None
        ),
        "model_hash": model["model_hash"] if report["promotable"] else None,
    }
    payload["agents"]["mode"] = "recorded"
    payload["agents"]["model"] = "openai/gpt-5.6-luna"
    payload["agents"]["recordings_path"] = str(
        artifact_directory.relative_to(ROOT) / "recordings"
    )
    for name in payload["agents"]["roles"]:
        version = 5 if name == "synthesis" else 4
        payload["agents"]["roles"][name]["prompt_version"] = f"{name}_v{version}"
    snapshot = Path("data/2023-11-01_2026-10-30")
    payload["task"]["universe_snapshot_id"] = snapshot.name
    payload["data"].update(
        {
            "universe": str(snapshot / "universe/symbols.json"),
            "market_bars": str(snapshot / "market/daily_bars.parquet"),
            "corporate_actions": str(
                snapshot / "market/corporate_actions.jsonl"
            ),
            "company_evidence": str(snapshot / "company/evidence.jsonl"),
            "macro_events": str(snapshot / "macro/events.jsonl"),
            "macro_documents": str(snapshot / "macro/documents"),
        }
    )
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "runs/llm_residual_session40/artifacts_recorded.json",
    )
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--candidate-path", type=Path, default=None)
    parser.add_argument(
        "--label-mode",
        choices=("intraday_range",),
        default="intraday_range",
    )
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    arguments = parser.parse_args()
    path, report = train_residual(
        arguments.artifacts,
        arguments.output_directory,
        arguments.candidate_path,
        label_mode=arguments.label_mode,
        base_config_path=arguments.base_config,
        feature_cache_path=arguments.feature_cache,
    )
    print(
        json.dumps(
            {
                "report": str(path),
                "promotable": report["promotable"],
                "winner": report["winner"],
            }
        )
    )


if __name__ == "__main__":
    main()
