"""Deterministic parameter search over frozen Agent artifacts."""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from financial_strategy.audit.writer import file_sha256
from financial_strategy.config import (
    AnomalyWeightConfig,
    MVPConfig,
    PairWeightConfig,
    RankingConfig,
    TuningConfig,
    WeightConfig,
    content_hash,
    load_mvp_config,
)
from financial_strategy.errors import ConfigError
from financial_strategy.evaluation.baselines import baseline_rankings
from financial_strategy.evaluation.labels import range_ratio
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.features.engine import compute_features
from financial_strategy.ranking.ranker import (
    MARKET_FEATURE_FIELDS,
    market_components,
    market_feature_percentiles,
    rank_cards,
)
from financial_strategy.schemas import (
    EvidenceRecord,
    FeatureRecord,
    MacroContext,
    MacroEvidenceRecord,
    ResearchCard,
    SymbolMacroExposure,
    Verdict,
)
from financial_strategy.tuning.artifacts import (
    _screening_config_hash,
    _screening_data_hashes,
    _validate_indexed_run,
    load_splits,
)
from financial_strategy.tuning.learning import (
    fit_nonnegative_linear_ranker,
    fit_pairwise_ranker,
    transform_rank_target,
)


@dataclass(frozen=True)
class SessionArtifact:
    session: str
    decision_cutoff: datetime
    features: dict[str, FeatureRecord]
    cards: list[ResearchCard]
    initial_cards: list[ResearchCard]
    macro: MacroContext
    exposures: dict[str, SymbolMacroExposure]
    company_evidence: list[EvidenceRecord]
    macro_evidence: list[MacroEvidenceRecord]


@dataclass(frozen=True)
class CandidateParameters:
    final_weights: tuple[float, float, float, float]
    volatility_weights: tuple[float, float]
    anomaly_weights: tuple[float, float, float]
    minimum_score: float
    candidate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "final_weights": {
                "volatility": self.final_weights[0],
                "anomaly": self.final_weights[1],
                "company_event": self.final_weights[2],
                "macro": self.final_weights[3],
            },
            "volatility_weights": {
                "median_abs_oc_return_20d": self.volatility_weights[0],
                "realized_volatility_20d": self.volatility_weights[1],
            },
            "anomaly_weights": {
                "abs_return_5d": self.anomaly_weights[0],
                "previous_session_relative_volume": self.anomaly_weights[1],
                "abs_distance_to_ma20": self.anomaly_weights[2],
            },
            "minimum_score": self.minimum_score,
            "candidate_count": self.candidate_count,
        }


def run_tuning(config: TuningConfig) -> tuple[Path, dict]:
    """Search train only, freeze the winner, then consume validation exactly once."""

    if config.output_path.exists() or config.tuned_config_path.exists():
        raise ConfigError(
            "VALIDATION_ALREADY_CONSUMED",
            "tuning output or tuned config already exists",
        )
    splits = load_splits(config.splits)
    index = json.loads(config.artifact_index.read_text())
    if index.get("version") != 2 or index["split_hash"] != splits["content_hash"]:
        raise ConfigError("ARTIFACT_INDEX_MISMATCH", str(config.artifact_index))
    base = load_mvp_config(config.screening_config)
    if (
        base.ranking.market_feature_weights is not None
        or base.ranking.volatility_weights is None
        or base.ranking.anomaly_weights is None
    ):
        raise ConfigError(
            "TUNING_REQUIRES_COMPONENT_RANKER",
            "tune starts from volatility_weights and anomaly_weights, "
            "with no market_feature_weights",
        )
    if index["screening_config_hash"] != _screening_config_hash(base):
        raise ConfigError("ARTIFACT_INDEX_MISMATCH", "screening config hash differs")
    if index.get("data_hashes") != _screening_data_hashes(base):
        raise ConfigError("ARTIFACT_INDEX_MISMATCH", "prediction input hash differs")
    if any(count < base.task.top_k for count in config.search.candidate_counts):
        raise ConfigError("INVALID_CANDIDATE_COUNT", "candidate counts must be >= top_k")

    expected_train = list(splits["train_sessions"])
    expected_validation = list(splits["validation_sessions"])
    available = {
        session
        for session, metadata in index["sessions"].items()
        if metadata["status"] == "completed"
    }
    train_coverage = len(available & set(expected_train)) / len(expected_train)
    failed_count = sum(
        metadata["status"] != "completed"
        for session, metadata in index["sessions"].items()
        if session in set(expected_train)
    )
    failed_rate = failed_count / len(expected_train)
    if train_coverage < config.search.minimum_session_coverage:
        raise ConfigError("INSUFFICIENT_TRAIN_COVERAGE", str(train_coverage))
    if failed_rate > config.search.maximum_failed_run_rate:
        raise ConfigError("EXCESSIVE_FAILED_RUN_RATE", str(failed_rate))
    if not set(expected_validation) <= available:
        raise ConfigError(
            "INCOMPLETE_VALIDATION_ARTIFACTS",
            "all validation artifacts must exist before one-shot validation",
        )

    train_artifacts = _load_artifacts(config, index, expected_train)
    train_reliability = _artifact_reliability(train_artifacts)
    if train_reliability["time_boundary_violation_rate"] != 0:
        raise ConfigError("TIME_BOUNDARY_VIOLATION", "train artifacts contain future evidence")
    if train_reliability["unsupported_claim_rate"] != 0:
        raise ConfigError("UNSUPPORTED_CLAIMS", "train artifacts contain unsupported facts")
    label_hash = file_sha256(config.labels)
    train_labels = _load_and_validate_labels(
        config.labels,
        expected_train,
        train_artifacts,
        base.data.market_bars,
    )
    candidates = _parameter_candidates(base, config)
    scored_metrics: dict[str, tuple[CandidateParameters, dict[str, dict]]] = {}
    for parameters in candidates:
        metrics = _evaluate_parameters(
            train_artifacts,
            train_labels,
            base,
            parameters,
            config.large_move_threshold,
        )
        scored_metrics[json.dumps(parameters.as_dict(), sort_keys=True)] = (
            parameters,
            metrics,
        )
    baseline_train_metrics = _evaluate_no_agent_baseline(
        train_artifacts,
        train_labels,
        base,
        config.large_move_threshold,
    )
    best, best_train_metrics, nested_cv = _select_with_nested_cv(
        scored_metrics,
        baseline_train_metrics,
        expected_train,
    )
    best_objective = (
        _metric_mean(best_train_metrics, "ndcg_at_k"),
        _metric_mean(best_train_metrics, "mean_absolute_return_at_k"),
        _metric_mean(best_train_metrics, "precision_at_k"),
        _metric_mean(best_train_metrics, "selection_coverage"),
    )
    if not all(math.isfinite(value) for value in best_objective):
        raise ConfigError("UNDEFINED_TRAIN_OBJECTIVE", "train metrics are undefined")
    tuned = _apply_parameters(base, best)
    candidate_path = config.tuned_config_path.with_suffix(".candidate.yaml")
    _write_yaml_atomic(candidate_path, tuned.model_dump(mode="json"))

    consumption_path = _begin_validation_consumption(
        config,
        splits,
        index,
        label_hash,
        candidate_path,
    )
    validation_artifacts = _load_artifacts(config, index, expected_validation)
    validation_labels = _load_and_validate_labels(
        config.labels,
        expected_validation,
        validation_artifacts,
        base.data.market_bars,
    )
    validation_variants = _evaluate_validation_variants(
        validation_artifacts,
        validation_labels,
        base,
        tuned,
        best.candidate_count,
        config.large_move_threshold,
    )
    validation_summaries = {
        name: _summarize_metrics(
            metrics,
            config.search.seed,
            config.search.bootstrap_samples,
            config.search.bootstrap_block_sessions,
        )
        for name, metrics in validation_variants.items()
    }
    tuned_ndcg = validation_summaries["tuned_full_system"]["average_ndcg_at_k"]
    original_ndcg = validation_summaries["original_full_system"]["average_ndcg_at_k"]
    no_agent_ndcg = validation_summaries["no_agent_ranker"]["average_ndcg_at_k"]
    ndcg_deltas = {
        baseline: _paired_metric_delta(
            validation_variants["tuned_full_system"],
            validation_variants[baseline],
            "ndcg_at_k",
            config.search.seed,
            config.search.bootstrap_samples,
            config.search.bootstrap_block_sessions,
        )
        for baseline in ("original_full_system", "no_agent_ranker")
    }
    validation_reliability = _artifact_reliability(validation_artifacts)
    accepted = (
        tuned_ndcg is not None
        and original_ndcg is not None
        and no_agent_ndcg is not None
        and tuned_ndcg > original_ndcg
        and tuned_ndcg > no_agent_ndcg
        and all(
            delta["95ci"] is not None and delta["95ci"][0] > 0
            for delta in ndcg_deltas.values()
        )
        and validation_reliability["time_boundary_violation_rate"] == 0
        and validation_reliability["unsupported_claim_rate"] == 0
    )
    if accepted:
        _write_yaml_atomic(config.tuned_config_path, tuned.model_dump(mode="json"))
    report = {
        "version": 1,
        "split_hash": splits["content_hash"],
        "artifact_index_hash": content_hash(index),
        "labels_sha256": label_hash,
        "screening_config_hash": _screening_config_hash(base),
        "search": config.search.model_dump(mode="json"),
        "searchable_components": {
            "company_event_weight": False,
            "macro_weight": False,
            "macro_observation_count": sum(
                len(artifact.macro_evidence) for artifact in train_artifacts
            ),
        },
        "train": {
            "expected_sessions": len(expected_train),
            "completed_sessions": len(train_artifacts),
            "coverage": train_coverage,
            "failed_run_rate": failed_rate,
            "reliability": train_reliability,
            "best_objective": {
                "average_ndcg_at_k": best_objective[0],
                "average_mean_absolute_return_at_k": best_objective[1],
                "average_precision_at_k": best_objective[2],
                "average_selection_coverage": best_objective[3],
            },
            "best_parameters": best.as_dict(),
            "best_metrics": _summarize_metrics(
                best_train_metrics,
                config.search.seed,
                config.search.bootstrap_samples,
                config.search.bootstrap_block_sessions,
            ),
            "baseline_metrics": _summarize_metrics(
                baseline_train_metrics,
                config.search.seed,
                config.search.bootstrap_samples,
                config.search.bootstrap_block_sessions,
            ),
            "nested_cv": nested_cv,
            "trial_count": len(scored_metrics),
        },
        "validation": {
            "session_count": len(validation_artifacts),
            "reliability": validation_reliability,
            "accept_tuned_config": accepted,
            "acceptance_rule": (
                "tuned average NDCG@10 and the lower bound of each paired 95% "
                "block-bootstrap delta must exceed original full system and "
                "no-agent ranker"
            ),
            "paired_ndcg_deltas": ndcg_deltas,
            "variants": validation_summaries,
        },
        "candidate_config": str(candidate_path),
        "tuned_config": str(config.tuned_config_path) if accepted else None,
    }
    _write_json_atomic(config.output_path, report)
    _complete_validation_consumption(consumption_path, config.output_path, accepted)
    return config.output_path, report


def run_train_cross_validation(config: TuningConfig) -> tuple[Path, dict]:
    """Select a challenger using train-only nested CV without touching validation."""

    splits = load_splits(config.splits)
    index = json.loads(config.artifact_index.read_text())
    base = load_mvp_config(config.screening_config)
    if (
        index.get("version") != 2
        or index.get("split_hash") != splits["content_hash"]
        or index.get("screening_config_hash") != _screening_config_hash(base)
        or index.get("data_hashes") != _screening_data_hashes(base)
    ):
        raise ConfigError("ARTIFACT_INDEX_MISMATCH", str(config.artifact_index))
    sessions = list(splits["train_sessions"])
    artifacts = _load_artifacts(config, index, sessions)
    labels = _load_and_validate_labels(
        config.labels, sessions, artifacts, base.data.market_bars
    )
    reliability = _artifact_reliability(artifacts)
    if reliability["time_boundary_violation_rate"] != 0:
        raise ConfigError("TIME_BOUNDARY_VIOLATION", "train artifacts contain future evidence")
    candidates = _parameter_candidates(
        base,
        config,
        any(artifact.macro_evidence for artifact in artifacts),
    )
    candidate_metrics = {
        json.dumps(parameters.as_dict(), sort_keys=True): (
            parameters,
            _evaluate_parameters(
                artifacts,
                labels,
                base,
                parameters,
                config.large_move_threshold,
            ),
        )
        for parameters in candidates
    }
    baseline = _evaluate_no_agent_baseline(
        artifacts,
        labels,
        base,
        config.large_move_threshold,
    )
    learned_weights, learned_metrics, learned_diagnostics = _learned_market_cv(
        artifacts,
        labels,
        base,
        baseline,
        config.large_move_threshold,
        config.search.seed,
        config.search.bootstrap_samples,
        config.search.bootstrap_block_sessions,
    )
    best, metrics, nested_cv = _select_with_nested_cv(
        candidate_metrics,
        baseline,
        sessions,
    )
    candidate_path = config.tuned_config_path.with_suffix(".train_cv_candidate.yaml")
    output_path = config.output_path.with_name(
        f"{config.output_path.stem}_train_cv{config.output_path.suffix}"
    )
    residual_delta = _paired_metric_delta(
        metrics,
        baseline,
        "ndcg_at_k",
        config.search.seed,
        config.search.bootstrap_samples,
        config.search.bootstrap_block_sessions,
    )
    # Company and macro final weights are fixed at zero: they do not enter the
    # published score. Promotion is the learned feature-weight ranker, or the
    # original component ranker when that model does not clear its gate.
    learned_promotable = learned_diagnostics["promotable"]
    market_total = (
        base.ranking.final_weights.volatility
        + base.ranking.final_weights.anomaly
    )
    component_volatility = base.ranking.volatility_weights
    component_anomaly = base.ranking.anomaly_weights
    assert component_volatility is not None and component_anomaly is not None
    if learned_promotable:
        promoted = CandidateParameters(
            final_weights=(
                base.ranking.final_weights.volatility / market_total,
                base.ranking.final_weights.anomaly / market_total,
                0.0,
                0.0,
            ),
            volatility_weights=tuple(component_volatility.model_dump().values()),
            anomaly_weights=tuple(component_anomaly.model_dump().values()),
            minimum_score=base.ranking.minimum_score,
            candidate_count=max(config.search.candidate_counts),
        )
        promoted_metrics = learned_metrics
    else:
        promoted = CandidateParameters(
            final_weights=(
                base.ranking.final_weights.volatility / market_total,
                base.ranking.final_weights.anomaly / market_total,
                0.0,
                0.0,
            ),
            volatility_weights=tuple(component_volatility.model_dump().values()),
            anomaly_weights=tuple(component_anomaly.model_dump().values()),
            minimum_score=0.0,
            candidate_count=max(config.search.candidate_counts),
        )
        promoted_metrics = baseline
    candidate = _apply_parameters(base, promoted)
    if learned_promotable:
        learned_mapping = {
            field: float(weight)
            for field, weight in zip(
                MARKET_FEATURE_FIELDS,
                learned_weights,
                strict=True,
            )
        }
        candidate = candidate.model_copy(
            update={
                "ranking": candidate.ranking.model_copy(
                    update={
                        "market_feature_weights": learned_mapping,
                        "volatility_weights": None,
                        "anomaly_weights": None,
                    }
                )
            }
        )
    _write_yaml_atomic(candidate_path, candidate.model_dump(mode="json"))
    report = {
        "version": 1,
        "mode": "train_only_nested_cv",
        "split_hash": splits["content_hash"],
        "artifact_index_hash": content_hash(index),
        "labels_sha256": file_sha256(config.labels),
        "session_count": len(sessions),
        "reliability": reliability,
        "best_parameters": best.as_dict(),
        "promoted_parameters": promoted.as_dict(),
        "candidate_metrics": _summarize_metrics(
            promoted_metrics,
            config.search.seed,
            config.search.bootstrap_samples,
            config.search.bootstrap_block_sessions,
        ),
        "baseline_metrics": _summarize_metrics(
            baseline,
            config.search.seed,
            config.search.bootstrap_samples,
            config.search.bootstrap_block_sessions,
        ),
        "paired_ndcg_delta": residual_delta,
        "agent_residual_delta": None,
        "agent_residual_promotable": False,
        "learned_market_ranker": learned_diagnostics,
        "diagnostic_breakdowns": _diagnostic_breakdowns(
            metrics,
            baseline,
            artifacts,
            labels,
        ),
        "nested_cv": nested_cv,
        "candidate_config": str(candidate_path),
        "validation_consumed": False,
    }
    _write_json_atomic(output_path, report)
    return output_path, report


def _load_artifacts(
    config: TuningConfig,
    index: dict,
    sessions: list[str],
) -> list[SessionArtifact]:
    result: list[SessionArtifact] = []
    for session in sessions:
        metadata = index["sessions"].get(session)
        if metadata is None or metadata["status"] != "completed":
            continue
        run_dir = config.runs_path / metadata["run_id"]
        _validate_indexed_run(
            run_dir,
            session,
            metadata,
            index["screening_config_hash"],
        )
        task = json.loads((run_dir / "task.json").read_text())
        stored_features = {
            row["symbol"]: FeatureRecord.model_validate(row)
            for row in pl.read_parquet(run_dir / "features.parquet").to_dicts()
        }
        market_frame = pl.read_parquet(run_dir / "snapshot/market.parquet")
        available_sessions = sorted(market_frame["session"].unique().to_list())
        exposure_records = [
            SymbolMacroExposure.model_validate(value)
            for value in json.loads(
                (run_dir / "snapshot/exposures.json").read_text()
            )
        ]
        features, rejected = compute_features(
            market_frame,
            available_sessions[-21:],
            required_sessions=21,
            symbol_sectors={item.symbol: item.sector for item in exposure_records},
        )
        if rejected or set(features) != set(stored_features):
            raise ConfigError(
                "ENHANCED_FEATURE_RECOMPUTE_FAILED",
                f"{session}: {rejected}",
            )
        result.append(
            SessionArtifact(
                session=session,
                decision_cutoff=datetime.fromisoformat(task["decision_cutoff"]),
                features=features,
                cards=[
                    ResearchCard.model_validate(item)
                    for item in json.loads((run_dir / "research_cards.json").read_text())
                ],
                initial_cards=[
                    ResearchCard.model_validate(item)
                    for item in json.loads(
                        (run_dir / "initial_research_cards.json").read_text()
                    )
                ],
                macro=MacroContext.model_validate(
                    json.loads((run_dir / "macro_context.json").read_text())
                ).model_copy(
                    update={
                        "target_session": datetime.fromisoformat(session).date(),
                        "decision_cutoff": datetime.fromisoformat(
                            task["decision_cutoff"]
                        ),
                    }
                ),
                exposures={
                    item.symbol: item for item in exposure_records
                },
                company_evidence=[
                    EvidenceRecord.model_validate(json.loads(line))
                    for line in (
                        run_dir / "snapshot/company_evidence.jsonl"
                    ).read_text().splitlines()
                    if line
                ],
                macro_evidence=[
                    MacroEvidenceRecord.model_validate(json.loads(line))
                    for line in (
                        run_dir / "snapshot/macro_evidence.jsonl"
                    ).read_text().splitlines()
                    if line
                ],
            )
        )
    return result


def _load_and_validate_labels(
    path: Path,
    sessions: list[str],
    artifacts: list[SessionArtifact],
    market_bars: Path,
) -> dict[str, dict[str, float]]:
    """Load one split and require one finite intraday-range label per symbol."""

    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("session").cast(pl.String).is_in(sessions))
        .collect()
    )
    required_columns = {"session", "symbol"}
    if not required_columns <= set(frame.columns):
        raise ConfigError("INVALID_LABELS", "labels are missing required columns")
    rows = frame.select("session", "symbol").to_dicts()
    expected = {
        (artifact.session, symbol)
        for artifact in artifacts
        for symbol in artifact.features
    }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        session = str(row["session"])
        key = (session, str(row["symbol"]))
        if key in seen:
            raise ConfigError("DUPLICATE_LABEL", f"{session}/{row['symbol']}")
        seen.add(key)
    bars = pl.read_parquet(market_bars).filter(
        pl.col("session").cast(pl.String).is_in(sessions)
    )
    if not {"session", "symbol", "high", "low"} <= set(bars.columns):
        raise ConfigError("INVALID_LABELS", "market bars are missing high/low")
    prices: dict[tuple[str, str], tuple[float, float]] = {}
    for row in bars.select("session", "symbol", "high", "low").iter_rows(named=True):
        key = (str(row["session"]), str(row["symbol"]))
        if key in prices:
            raise ConfigError("DUPLICATE_LABEL", f"{key[0]}/{key[1]}")
        try:
            prices[key] = (float(row["high"]), float(row["low"]))
        except (TypeError, ValueError) as exc:
            raise ConfigError("INVALID_LABEL_VALUE", f"{key[0]}/{key[1]}") from exc
    labels: dict[str, dict[str, float]] = {}
    for session, symbol in seen:
        price = prices.get((session, symbol))
        ratio = None if price is None else range_ratio(*price)
        if ratio is None:
            raise ConfigError("INVALID_LABEL_VALUE", f"{session}/{symbol}")
        labels.setdefault(session, {})[symbol] = ratio
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ConfigError(
            "LABEL_COVERAGE_MISMATCH",
            f"missing={missing[:3]}, extra={extra[:3]}",
        )
    return labels


def _begin_validation_consumption(
    config: TuningConfig,
    splits: dict,
    index: dict,
    label_hash: str,
    candidate_path: Path,
) -> Path:
    """Reserve the chronological holdout with a durable, split-keyed record."""

    # Keep the holdout ledger beside the frozen split, not under configurable
    # output paths. Changing report or run destinations must not unlock validation.
    ledger = config.splits.parent / ".validation_consumed" / f"{splits['content_hash']}.json"
    fingerprint = content_hash(
        {
            "split_hash": splits["content_hash"],
            "artifact_index_hash": content_hash(index),
            "labels_sha256": label_hash,
            "search": config.search.model_dump(mode="json"),
            "large_move_threshold": config.large_move_threshold,
            "candidate_sha256": file_sha256(candidate_path),
        }
    )
    if ledger.exists():
        current = json.loads(ledger.read_text())
        if current.get("status") == "in_progress" and current.get("fingerprint") == fingerprint:
            return ledger
        raise ConfigError("VALIDATION_ALREADY_CONSUMED", str(ledger))
    _write_json_atomic(
        ledger,
        {
            "version": 1,
            "status": "in_progress",
            "fingerprint": fingerprint,
        },
    )
    return ledger


def _complete_validation_consumption(
    ledger: Path,
    report_path: Path,
    accepted: bool,
) -> None:
    record = json.loads(ledger.read_text())
    record.update(
        {
            "status": "completed",
            "report_path": str(report_path),
            "report_sha256": file_sha256(report_path),
            "accepted": accepted,
        }
    )
    _write_json_atomic(ledger, record)


def _parameter_candidates(
    base: MVPConfig,
    config: TuningConfig,
) -> list[CandidateParameters]:
    volatility_weights = base.ranking.volatility_weights
    anomaly_weights = base.ranking.anomaly_weights
    if (
        volatility_weights is None
        or anomaly_weights is None
        or base.ranking.market_feature_weights is not None
    ):
        raise ConfigError(
            "TUNING_REQUIRES_COMPONENT_RANKER",
            "tune starts from volatility_weights and anomaly_weights, "
            "with no market_feature_weights",
        )
    base_parameters = CandidateParameters(
        final_weights=tuple(base.ranking.final_weights.model_dump().values()),
        volatility_weights=tuple(volatility_weights.model_dump().values()),
        anomaly_weights=tuple(anomaly_weights.model_dump().values()),
        minimum_score=base.ranking.minimum_score,
        candidate_count=max(
            base.screening.candidate_multiplier * base.task.top_k,
            base.screening.minimum_candidates,
        ),
    )
    rng = random.Random(config.search.seed)
    values = {json.dumps(base_parameters.as_dict(), sort_keys=True): base_parameters}
    market_total = (
        base.ranking.final_weights.volatility + base.ranking.final_weights.anomaly
    )
    anchor_volatility = base.ranking.final_weights.volatility / market_total
    anchor = CandidateParameters(
        final_weights=(anchor_volatility, 1.0 - anchor_volatility, 0.0, 0.0),
        volatility_weights=base_parameters.volatility_weights,
        anomaly_weights=base_parameters.anomaly_weights,
        minimum_score=base.ranking.minimum_score,
        candidate_count=max(config.search.candidate_counts),
    )
    values[json.dumps(anchor.as_dict(), sort_keys=True)] = anchor
    while len(values) < config.search.trials:
        volatility_ratio = min(
            0.80,
            max(0.35, anchor_volatility + rng.uniform(-0.15, 0.15)),
        )
        final_weights = (volatility_ratio, 1.0 - volatility_ratio, 0.0, 0.0)
        random_volatility = _simplex(rng, 2)
        volatility_weights = tuple(
            0.7 * base_value + 0.3 * random_value
            for base_value, random_value in zip(
                base_parameters.volatility_weights,
                random_volatility,
                strict=True,
            )
        )
        random_anomaly = _simplex(rng, 3)
        anomaly_weights = tuple(
            0.7 * base_value + 0.3 * random_value
            for base_value, random_value in zip(
                base_parameters.anomaly_weights,
                random_anomaly,
                strict=True,
            )
        )
        parameters = CandidateParameters(
            final_weights=final_weights,
            volatility_weights=volatility_weights,
            anomaly_weights=anomaly_weights,
            minimum_score=rng.choice(config.search.minimum_scores),
            candidate_count=rng.choice(config.search.candidate_counts),
        )
        values.setdefault(json.dumps(parameters.as_dict(), sort_keys=True), parameters)
    return list(values.values())


def _simplex(rng: random.Random, size: int) -> tuple[float, ...]:
    raw = [rng.gammavariate(1.0, 1.0) for _ in range(size)]
    total = sum(raw)
    return tuple(value / total for value in raw)


def _apply_parameters(base: MVPConfig, parameters: CandidateParameters) -> MVPConfig:
    ranking = RankingConfig(
        minimum_score=parameters.minimum_score,
        market_feature_weights=base.ranking.market_feature_weights,
        final_weights=WeightConfig(
            volatility=parameters.final_weights[0],
            anomaly=parameters.final_weights[1],
            company_event=parameters.final_weights[2],
            macro=parameters.final_weights[3],
        ),
        volatility_weights=PairWeightConfig(
            median_abs_oc_return_20d=parameters.volatility_weights[0],
            realized_volatility_20d=parameters.volatility_weights[1],
        ),
        anomaly_weights=AnomalyWeightConfig(
            abs_return_5d=parameters.anomaly_weights[0],
            previous_session_relative_volume=parameters.anomaly_weights[1],
            abs_distance_to_ma20=parameters.anomaly_weights[2],
        ),
        percentile=base.ranking.percentile,
        company_event=base.ranking.company_event,
        macro=base.ranking.macro,
    )
    screening = base.screening.model_copy(
        update={
            "candidate_multiplier": 1,
            "minimum_candidates": parameters.candidate_count,
        }
    )
    return base.model_copy(update={"ranking": ranking, "screening": screening})


def _evaluate_parameters(
    artifacts: list[SessionArtifact],
    labels: dict[str, dict[str, float]],
    base: MVPConfig,
    parameters: CandidateParameters,
    large_move_threshold: float,
) -> dict[str, dict]:
    tuned = _apply_parameters(base, parameters)
    results: dict[str, dict] = {}
    for artifact in artifacts:
        volatility, anomaly = market_components(artifact.features, tuned.ranking)
        candidates = _coarse_candidates(
            volatility,
            anomaly,
            tuned,
            parameters.candidate_count,
        )
        cards = [card for card in artifact.cards if card.symbol in candidates]
        entries = rank_cards(
            cards,
            volatility,
            anomaly,
            artifact.macro,
            artifact.exposures,
            tuned.ranking,
            tuned.task.top_k,
        )
        session_labels = labels.get(artifact.session, {})
        results[artifact.session] = evaluate_ranking(
            [entry.symbol for entry in entries],
            session_labels,
            k=tuned.task.top_k,
            eligible_count=len(session_labels),
            large_move_threshold=large_move_threshold,
        )
    return results


def _evaluate_no_agent_baseline(
    artifacts: list[SessionArtifact],
    labels: dict[str, dict[str, float]],
    base: MVPConfig,
    large_move_threshold: float,
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for artifact in artifacts:
        volatility, anomaly = market_components(artifact.features, base.ranking)
        symbols = baseline_rankings(
            artifact.features,
            volatility,
            anomaly,
            artifact.company_evidence,
            base,
        )["no_agent_ranker"]
        session_labels = labels[artifact.session]
        results[artifact.session] = evaluate_ranking(
            symbols,
            session_labels,
            k=base.task.top_k,
            eligible_count=len(session_labels),
            large_move_threshold=large_move_threshold,
        )
    return results


def _select_with_nested_cv(
    candidates: dict[str, tuple[CandidateParameters, dict[str, dict]]],
    baseline: dict[str, dict],
    ordered_sessions: list[str],
) -> tuple[CandidateParameters, dict[str, dict], dict]:
    """Select on expanding inner folds and report untouched outer-fold deltas."""

    if len(ordered_sessions) < 100:
        folds = _expanding_test_folds(ordered_sessions, 40, 20, 1)
        selected_key = _select_candidate_key(candidates, baseline, folds)
        parameters, metrics = candidates[selected_key]
        return parameters, metrics, {"outer_folds": [], "final_inner_folds": folds}

    outer_reports: list[dict] = []
    outer_starts = list(range(80, len(ordered_sessions), 20))
    for outer_start in outer_starts:
        outer_test = ordered_sessions[outer_start : outer_start + 20]
        if not outer_test:
            continue
        inner_sessions = ordered_sessions[: max(0, outer_start - 1)]
        inner_folds = _expanding_test_folds(inner_sessions, 40, 20, 1)
        selected_key = _select_candidate_key(candidates, baseline, inner_folds)
        parameters, metrics = candidates[selected_key]
        deltas = [
            metrics[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
            for session in outer_test
            if metrics[session]["ndcg_at_k"] is not None
            and baseline[session]["ndcg_at_k"] is not None
        ]
        outer_reports.append(
            {
                "train_end": inner_sessions[-1],
                "test_sessions": outer_test,
                "selected_parameters": parameters.as_dict(),
                "mean_paired_ndcg_delta": sum(deltas) / len(deltas),
                "won_sessions": sum(delta > 0 for delta in deltas),
                "lost_sessions": sum(delta < 0 for delta in deltas),
            }
        )

    final_folds = _expanding_test_folds(ordered_sessions, 40, 20, 1)
    selected_key = _select_candidate_key(candidates, baseline, final_folds)
    parameters, metrics = candidates[selected_key]
    return parameters, metrics, {
        "embargo_sessions": 1,
        "minimum_train_sessions": 40,
        "test_block_sessions": 20,
        "outer_folds": outer_reports,
        "final_inner_folds": final_folds,
    }


def _expanding_test_folds(
    sessions: list[str],
    minimum_train: int,
    test_size: int,
    embargo: int,
) -> list[list[str]]:
    folds: list[list[str]] = []
    start = minimum_train + embargo
    while start + test_size <= len(sessions):
        folds.append(sessions[start : start + test_size])
        start += test_size
    if not folds:
        folds.append(sessions[-min(test_size, len(sessions)) :])
    return folds


def _select_candidate_key(
    candidates: dict[str, tuple[CandidateParameters, dict[str, dict]]],
    baseline: dict[str, dict],
    folds: list[list[str]],
) -> str:
    scored: list[tuple[tuple[float, ...], str]] = []
    for key, (_parameters, metrics) in candidates.items():
        fold_deltas: list[float] = []
        wins = 0
        for fold in folds:
            deltas = [
                metrics[session]["ndcg_at_k"] - baseline[session]["ndcg_at_k"]
                for session in fold
                if metrics[session]["ndcg_at_k"] is not None
                and baseline[session]["ndcg_at_k"] is not None
            ]
            delta = sum(deltas) / len(deltas)
            fold_deltas.append(delta)
            wins += delta > 0
        ordered_deltas = sorted(fold_deltas)
        median = ordered_deltas[len(ordered_deltas) // 2]
        all_sessions = [session for fold in folds for session in fold]
        coverage = sum(metrics[s]["selection_coverage"] for s in all_sessions) / len(
            all_sessions
        )
        mean_return = sum(
            metrics[s]["mean_absolute_return_at_k"] for s in all_sessions
        ) / len(all_sessions)
        objective = (
            float(wins),
            median,
            min(fold_deltas),
            sum(fold_deltas) / len(fold_deltas),
            coverage,
            mean_return,
        )
        scored.append((objective, key))
    scored.sort(key=lambda item: (*(-value for value in item[0]), item[1]))
    return scored[0][1]


def _coarse_candidates(
    volatility: dict[str, float],
    anomaly: dict[str, float],
    config: MVPConfig,
    candidate_count: int,
) -> set[str]:
    scores = {
        symbol: config.screening.volatility_weight * volatility[symbol]
        + config.screening.anomaly_weight * anomaly[symbol]
        for symbol in volatility.keys() & anomaly.keys()
    }
    ordered = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    return set(ordered[:candidate_count])


def _evaluate_validation_variants(
    artifacts: list[SessionArtifact],
    labels: dict[str, dict[str, float]],
    base: MVPConfig,
    tuned: MVPConfig,
    candidate_count: int,
    large_move_threshold: float,
) -> dict[str, dict[str, dict]]:
    variant_names = [
        "original_full_system",
        "tuned_full_system",
        "historical_volatility",
        "historical_abs_oc",
        "fixed_earnings_rule",
        "no_agent_ranker",
        "deterministic_event_baseline",
    ]
    if tuned.ranking.residual_model.enabled:
        variant_names.extend(["no_company_event", "no_macro", "no_semantic_verifier"])
    variants: dict[str, dict[str, dict]] = {name: {} for name in variant_names}
    for artifact in artifacts:
        base_v, base_a = market_components(artifact.features, base.ranking)
        tuned_v, tuned_a = market_components(artifact.features, tuned.ranking)
        original_candidate_count = max(
            base.screening.candidate_multiplier * base.task.top_k,
            base.screening.minimum_candidates,
        )
        original_candidates = _coarse_candidates(
            base_v,
            base_a,
            base,
            original_candidate_count,
        )
        tuned_candidates = _coarse_candidates(
            tuned_v, tuned_a, tuned, candidate_count
        )
        original_entries = rank_cards(
            [
                card
                for card in artifact.cards
                if card.symbol in original_candidates
            ],
            base_v,
            base_a,
            artifact.macro,
            artifact.exposures,
            base.ranking,
            base.task.top_k,
        )
        cards = [card for card in artifact.cards if card.symbol in tuned_candidates]
        initial = [
            card for card in artifact.initial_cards if card.symbol in tuned_candidates
        ]
        tuned_entries = rank_cards(
            cards,
            tuned_v,
            tuned_a,
            artifact.macro,
            artifact.exposures,
            tuned.ranking,
            tuned.task.top_k,
        )
        symbol_lists = {
            "original_full_system": [item.symbol for item in original_entries],
            "tuned_full_system": [item.symbol for item in tuned_entries],
        }
        symbol_lists.update(
            baseline_rankings(
                artifact.features,
                base_v,
                base_a,
                artifact.company_evidence,
                base,
            )
        )
        ablations: list[tuple[str, list]] = []
        if tuned.ranking.residual_model.enabled:
            ablations = [
                ("no_company_event", cards),
                ("no_macro", cards),
                ("no_semantic_verifier", initial),
            ]
        for variant, variant_cards in ablations:
            symbol_lists[variant] = [
                item.symbol
                for item in rank_cards(
                    variant_cards,
                    tuned_v,
                    tuned_a,
                    artifact.macro,
                    artifact.exposures,
                    tuned.ranking,
                    tuned.task.top_k,
                    variant,
                )
            ]
        session_labels = labels.get(artifact.session, {})
        true_large_movers = {
            symbol
            for symbol, value in session_labels.items()
            if value >= large_move_threshold
        }
        for name, symbols in symbol_lists.items():
            metrics = evaluate_ranking(
                symbols,
                session_labels,
                k=base.task.top_k,
                eligible_count=len(session_labels),
                large_move_threshold=large_move_threshold,
            )
            if name == "original_full_system":
                metrics["coarse_screen_recall"] = (
                    len(true_large_movers & original_candidates)
                    / len(true_large_movers)
                    if true_large_movers
                    else None
                )
            elif name == "tuned_full_system":
                metrics["coarse_screen_recall"] = (
                    len(true_large_movers & tuned_candidates) / len(true_large_movers)
                    if true_large_movers
                    else None
                )
            variants[name][artifact.session] = metrics
    return variants


def _artifact_reliability(artifacts: list[SessionArtifact]) -> dict[str, float]:
    evidence_total = 0
    time_violations = 0
    fact_total = 0
    unsupported = 0
    for artifact in artifacts:
        evidence = [*artifact.company_evidence, *artifact.macro_evidence]
        evidence_by_id = {item.evidence_id: item for item in evidence}
        evidence_total += len(evidence)
        time_violations += sum(
            item.available_at > artifact.decision_cutoff for item in evidence
        )
        for card in artifact.cards:
            # Unsupported claims on an abstained card demonstrate that verification
            # worked: the card cannot enter the full-system ranking. Reliability must
            # measure unsupported facts that escaped both verification gates, rather
            # than treating correctly rejected drafts as published research.
            if (
                card.deterministic_verification is None
                or card.deterministic_verification.verdict != Verdict.PASS
                or card.semantic_verification is None
                or card.semantic_verification.verdict != Verdict.PASS
            ):
                continue
            for claim in card.claims:
                if claim.claim_type.value != "observed_fact":
                    continue
                fact_total += 1
                evidence_supported = bool(claim.evidence_ids) and all(
                    evidence_id in evidence_by_id for evidence_id in claim.evidence_ids
                )
                feature_supported = bool(claim.feature_ids) and all(
                    feature_id == card.features.feature_id
                    for feature_id in claim.feature_ids
                )
                unsupported += not (evidence_supported or feature_supported)
    return {
        "time_boundary_violation_rate": (
            time_violations / evidence_total if evidence_total else 0.0
        ),
        "unsupported_claim_rate": unsupported / fact_total if fact_total else 0.0,
    }


def _metric_mean(metrics: dict[str, dict], name: str) -> float:
    values = [item[name] for item in metrics.values() if item[name] is not None]
    return sum(values) / len(values) if values else float("-inf")


def _summarize_metrics(
    metrics: dict[str, dict],
    seed: int,
    bootstrap_samples: int,
    block_size: int,
) -> dict:
    summary: dict[str, Any] = {
        "session_count": len(metrics),
        "monthly": {},
    }
    for name in (
        "ndcg_at_k",
        "mean_absolute_return_at_k",
        "precision_at_k",
        "selection_coverage",
        "coarse_screen_recall",
    ):
        ordered = [
            metrics[session].get(name)
            for session in sorted(metrics)
            if metrics[session].get(name) is not None
        ]
        if not ordered and name == "coarse_screen_recall":
            continue
        summary[f"average_{name}"] = (
            sum(ordered) / len(ordered) if ordered else None
        )
        summary[f"{name}_95ci"] = _block_bootstrap_ci(
            ordered,
            seed,
            bootstrap_samples,
            block_size,
        )
        summary[f"{name}_95ci_block_sensitivity"] = {
            str(sensitivity): _block_bootstrap_ci(
                ordered,
                seed,
                bootstrap_samples,
                sensitivity,
            )
            for sensitivity in (5, 10, 20)
        }
    months = sorted({session[:7] for session in metrics})
    for month in months:
        monthly_metrics = {
            session: value
            for session, value in metrics.items()
            if session.startswith(month)
        }
        month_summary = {}
        for name in (
            "ndcg_at_k",
            "mean_absolute_return_at_k",
            "precision_at_k",
            "selection_coverage",
            "coarse_screen_recall",
        ):
            values = [
                item.get(name)
                for item in monthly_metrics.values()
                if item.get(name) is not None
            ]
            if values or name != "coarse_screen_recall":
                month_summary[name] = sum(values) / len(values) if values else None
        summary["monthly"][month] = month_summary
    return summary


def _paired_metric_delta(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    metric: str,
    seed: int,
    samples: int,
    block_size: int,
) -> dict[str, float | list[float] | None]:
    """Estimate a paired session-level improvement and preserve time blocks."""

    differences = [
        candidate[session][metric] - baseline[session][metric]
        for session in sorted(candidate.keys() & baseline.keys())
        if candidate[session].get(metric) is not None
        and baseline[session].get(metric) is not None
    ]
    return {
        "mean": sum(differences) / len(differences) if differences else None,
        "95ci": _block_bootstrap_ci(differences, seed, samples, block_size),
    }


def _learned_market_cv(
    artifacts: list[SessionArtifact],
    labels: dict[str, dict[str, float]],
    base: MVPConfig,
    baseline: dict[str, dict],
    large_move_threshold: float,
    seed: int,
    bootstrap_samples: int,
    bootstrap_block_sessions: int,
) -> tuple[np.ndarray, dict[str, dict], dict]:
    """Compare pre-registered rank objectives using only expanding train folds."""

    ordered_sessions = [artifact.session for artifact in artifacts]
    components: dict[str, dict[str, np.ndarray]] = {}
    for artifact in artifacts:
        percentile_maps = market_feature_percentiles(
            artifact.features,
            float(base.ranking.percentile["singleton_value"]),
        )
        components[artifact.session] = {
            symbol: np.asarray(
                [percentile_maps[field][symbol] for field in MARKET_FEATURE_FIELDS]
            )
            for symbol in artifact.features
        }
    volatility_weights = base.ranking.volatility_weights
    anomaly_weights = base.ranking.anomaly_weights
    if volatility_weights is None or anomaly_weights is None:
        raise ConfigError(
            "TUNING_REQUIRES_COMPONENT_RANKER",
            "the learned-ranker anchor is built from volatility_weights and anomaly_weights",
        )
    anchor = np.zeros(len(MARKET_FEATURE_FIELDS), dtype=float)
    market_total = base.ranking.final_weights.volatility + base.ranking.final_weights.anomaly
    anchor[MARKET_FEATURE_FIELDS.index("median_abs_oc_return_20d")] = (
        base.ranking.final_weights.volatility
        / market_total
        * volatility_weights.median_abs_oc_return_20d
    )
    anchor[MARKET_FEATURE_FIELDS.index("realized_volatility_20d")] = (
        base.ranking.final_weights.volatility
        / market_total
        * volatility_weights.realized_volatility_20d
    )
    anomaly_share = base.ranking.final_weights.anomaly / market_total
    anchor[MARKET_FEATURE_FIELDS.index("return_5d")] = (
        anomaly_share * anomaly_weights.abs_return_5d
    )
    anchor[MARKET_FEATURE_FIELDS.index("previous_session_relative_volume")] = (
        anomaly_share * anomaly_weights.previous_session_relative_volume
    )
    anchor[MARKET_FEATURE_FIELDS.index("distance_to_ma20")] = (
        anomaly_share * anomaly_weights.abs_distance_to_ma20
    )
    variants: dict[str, dict] = {}
    schemes = (
        "raw_magnitude",
        "winsorized_magnitude",
        "session_percentile",
        "large_move_then_magnitude",
    )
    for learner in ("linear", "pairwise"):
        for scheme in schemes:
            key = f"{learner}:{scheme}"
            fold_metrics: dict[str, dict] = {}
            fold_weights: list[list[float]] = []
            for test_start in range(41, len(ordered_sessions), 20):
                train_sessions = ordered_sessions[: test_start - 1]
                test_sessions = ordered_sessions[test_start : test_start + 20]
                if not test_sessions:
                    continue
                weights = _fit_market_ranker(
                    train_sessions,
                    components,
                    labels,
                    anchor,
                    learner,
                    scheme,
                    large_move_threshold,
                    0.05 if learner == "pairwise" else 0.25,
                )
                fold_weights.append(weights.tolist())
                fold_metrics.update(
                    _evaluate_market_weights(
                        test_sessions,
                        components,
                        labels,
                        weights,
                        base.task.top_k,
                        large_move_threshold,
                    )
                )
            delta = _paired_metric_delta(
                fold_metrics,
                baseline,
                "ndcg_at_k",
                seed,
                bootstrap_samples,
                bootstrap_block_sessions,
            )
            variants[key] = {
                "paired_ndcg_delta": delta,
                "fold_weights": fold_weights,
                "test_session_count": len(fold_metrics),
            }
    winner = max(
        variants,
        key=lambda key: (
            variants[key]["paired_ndcg_delta"]["mean"] or float("-inf"),
            key,
        ),
    )
    learner, scheme = winner.split(":", maxsplit=1)
    final_weights = _fit_market_ranker(
        ordered_sessions,
        components,
        labels,
        anchor,
        learner,
        scheme,
        large_move_threshold,
        0.05 if learner == "pairwise" else 0.25,
    )
    full_metrics = _evaluate_market_weights(
        ordered_sessions,
        components,
        labels,
        final_weights,
        base.task.top_k,
        large_move_threshold,
    )
    winner_ci = variants[winner]["paired_ndcg_delta"]["95ci"]
    return final_weights, full_metrics, {
        "winner": winner,
        "final_weights": final_weights.tolist(),
        "promotable": bool(winner_ci is not None and winner_ci[0] > 0),
        "variants": variants,
    }


def _fit_market_ranker(
    sessions: list[str],
    components: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    anchor: np.ndarray,
    learner: str,
    scheme: str,
    large_move_threshold: float,
    shrinkage: float,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    targets: list[float] = []
    session_ids: list[str] = []
    for session in sessions:
        symbols = sorted(components[session].keys() & labels[session].keys())
        session_targets = np.asarray([labels[session][symbol] for symbol in symbols])
        transformed = transform_rank_target(
            session_targets,
            scheme,
            large_move_threshold=large_move_threshold,
        )
        rows.extend(components[session][symbol] for symbol in symbols)
        targets.extend(transformed.tolist())
        session_ids.extend([session] * len(symbols))
    feature_array = np.asarray(rows, dtype=float)
    target_array = np.asarray(targets, dtype=float)
    if learner == "linear":
        return fit_nonnegative_linear_ranker(
            feature_array,
            target_array,
            anchor,
            shrinkage=shrinkage,
        )
    return fit_pairwise_ranker(
        feature_array,
        target_array,
        np.asarray(session_ids),
        anchor,
        shrinkage=shrinkage,
    )


def _evaluate_market_weights(
    sessions: list[str],
    components: dict[str, dict[str, np.ndarray]],
    labels: dict[str, dict[str, float]],
    weights: np.ndarray,
    top_k: int,
    large_move_threshold: float,
) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for session in sessions:
        ranked = sorted(
            components[session],
            key=lambda symbol: (
                -float(np.dot(components[session][symbol], weights)),
                symbol,
            ),
        )
        result[session] = evaluate_ranking(
            ranked[:top_k],
            labels[session],
            k=top_k,
            eligible_count=len(labels[session]),
            large_move_threshold=large_move_threshold,
        )
    return result


def _diagnostic_breakdowns(
    candidate: dict[str, dict],
    baseline: dict[str, dict],
    artifacts: list[SessionArtifact],
    labels: dict[str, dict[str, float]],
) -> dict:
    monthly: dict[str, list[float]] = {}
    event_groups: dict[str, list[float]] = {"event": [], "no_event": []}
    overlaps: list[float] = []
    sector_returns: dict[str, dict[str, list[float]]] = {}
    artifacts_by_session = {artifact.session: artifact for artifact in artifacts}
    for session in sorted(candidate.keys() & baseline.keys()):
        candidate_ndcg = candidate[session]["ndcg_at_k"]
        baseline_ndcg = baseline[session]["ndcg_at_k"]
        if candidate_ndcg is None or baseline_ndcg is None:
            continue
        delta = candidate_ndcg - baseline_ndcg
        monthly.setdefault(session[:7], []).append(delta)
        artifact = artifacts_by_session[session]
        event_group = "event" if any(card.events for card in artifact.cards) else "no_event"
        event_groups[event_group].append(delta)
        candidate_symbols = list(candidate[session].get("selected_symbols", []))
        baseline_symbols = list(baseline[session].get("selected_symbols", []))
        union = set(candidate_symbols) | set(baseline_symbols)
        overlaps.append(
            len(set(candidate_symbols) & set(baseline_symbols)) / len(union)
            if union
            else 1.0
        )
        for variant, symbols in (
            ("candidate", candidate_symbols),
            ("baseline", baseline_symbols),
        ):
            for symbol in symbols:
                exposure = artifact.exposures.get(symbol)
                if exposure is None or symbol not in labels[session]:
                    continue
                sector_returns.setdefault(
                    exposure.sector,
                    {"candidate": [], "baseline": []},
                )[variant].append(labels[session][symbol])
    return {
        "monthly_mean_paired_ndcg_delta": {
            month: sum(values) / len(values) for month, values in monthly.items()
        },
        "event_mean_paired_ndcg_delta": {
            group: (sum(values) / len(values) if values else None)
            for group, values in event_groups.items()
        },
        "mean_top_k_jaccard_overlap": (
            sum(overlaps) / len(overlaps) if overlaps else None
        ),
        "sector_selected_mean_return": {
            sector: {
                variant: (sum(values) / len(values) if values else None)
                for variant, values in variants.items()
            }
            for sector, variants in sector_returns.items()
        },
    }


def _block_bootstrap_ci(
    values: list[float],
    seed: int,
    samples: int,
    block_size: int,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    block_count = math.ceil(len(values) / block_size)
    effective_block = min(block_size, len(values))
    possible_starts = range(0, len(values) - effective_block + 1)
    estimates: list[float] = []
    for _ in range(samples):
        sampled: list[float] = []
        for _block in range(block_count):
            start = rng.choice(possible_starts)
            sampled.extend(values[start : start + effective_block])
        sampled = sampled[: len(values)]
        estimates.append(sum(sampled) / len(sampled))
    estimates.sort()
    lower = estimates[int(0.025 * (len(estimates) - 1))]
    upper = estimates[int(0.975 * (len(estimates) - 1))]
    return [lower, upper]


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_yaml_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            yaml.safe_dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)
