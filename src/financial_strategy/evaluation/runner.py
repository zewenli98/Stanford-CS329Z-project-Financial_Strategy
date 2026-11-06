"""Create fresh historical predictions, freeze them, then load labels."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, time
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from financial_strategy.config import load_evaluation_config, load_mvp_config
from financial_strategy.errors import ConfigError
from financial_strategy.evaluation.baselines import baseline_rankings
from financial_strategy.evaluation.bootstrap import paired_moving_block_bootstrap
from financial_strategy.evaluation.labels import range_ratio
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.evaluation.process_scores import score_run_process
from financial_strategy.features.engine import compute_features
from financial_strategy.ranking.ranker import market_components, rank_cards
from financial_strategy.schemas import (
    EvidenceRecord,
    FeatureRecord,
    MacroContext,
    ResearchCard,
    SymbolMacroExposure,
)
from financial_strategy.service import run_screening


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_evaluation(path: str | Path) -> tuple[Path, dict]:
    from financial_strategy.adaptive.config import is_adaptive_config
    from financial_strategy.config import _load_yaml

    raw = _load_yaml(Path(path))
    if is_adaptive_config(raw["screening_config"]):
        from financial_strategy.adaptive.evaluation import run_adaptive_evaluation

        return run_adaptive_evaluation(path)
    evaluation_config = load_evaluation_config(path)
    screening_path = evaluation_config.screening_config
    config = load_mvp_config(screening_path)
    inactive_ablations = {
        "no_company_event",
        "no_macro",
        "no_semantic_verifier",
    } & set(evaluation_config.evaluation.variants)
    if inactive_ablations and not config.ranking.residual_model.enabled:
        raise ConfigError(
            "INACTIVE_RANKING_ABLATION",
            "company, macro, and semantic ablations change the order only when "
            "residual_model.enabled is true",
        )
    labels_frame = pl.read_parquet(evaluation_config.evaluation.labels)
    market_bars = pl.read_parquet(config.data.market_bars)
    reports: list[dict] = []
    report_lock = threading.Lock()

    evaluation_id = f"evaluation_{uuid4().hex[:10]}"

    def evaluate_session(item) -> None:
        replicate_id, session = item
        cutoff = datetime.combine(
            session,
            time.fromisoformat(evaluation_config.evaluation.cutoff_time),
            ZoneInfo(evaluation_config.evaluation.timezone),
        )
        run_id = f"eval_r{replicate_id}_{session.isoformat()}_{uuid4().hex[:8]}"
        try:
            run_dir, _watchlist = run_screening(
                screening_path,
                run_id=run_id,
                task_overrides={"target_session": session, "decision_cutoff": cutoff},
                cache_namespace=f"{evaluation_id}/replicate_{replicate_id}",
            )
        except RuntimeError:
            run_dir = Path(config.output["runs_path"]) / run_id
            error_path = run_dir / "error.json"
            error = json.loads(error_path.read_text()) if error_path.exists() else {}
            with report_lock:
                reports.append(
                    {
                        "session": session.isoformat(),
                        "replicate_id": str(replicate_id),
                        "run_id": run_id,
                        "status": "failed",
                        "failure_code": error.get("code", "UNKNOWN_FAILURE"),
                        "metrics": {},
                    }
                )
            return
        run_config = load_mvp_config(run_dir / "config.yaml")
        features = {
            row["symbol"]: FeatureRecord.model_validate(row)
            for row in pl.read_parquet(run_dir / "features.parquet").to_dicts()
        }
        volatility, anomaly = market_components(features, run_config.ranking)
        company_evidence_records = [
            EvidenceRecord.model_validate(item)
            for item in _read_jsonl(run_dir / "snapshot/company_evidence.jsonl")
        ]
        variants = baseline_rankings(
            features,
            volatility,
            anomaly,
            company_evidence_records,
            run_config,
            cutoff,
        )
        final_cards = [
            ResearchCard.model_validate(item)
            for item in json.loads((run_dir / "research_cards.json").read_text())
        ]
        initial_cards = [
            ResearchCard.model_validate(item)
            for item in json.loads((run_dir / "initial_research_cards.json").read_text())
        ]
        macro = MacroContext.model_validate(
            json.loads((run_dir / "macro_context.json").read_text())
        )
        exposures = {
            item.symbol: item
            for item in (
                SymbolMacroExposure.model_validate(value)
                for value in json.loads((run_dir / "snapshot/exposures.json").read_text())
            )
        }
        for variant, cards in (
            ("full_system", final_cards),
            ("no_company_event", final_cards),
            ("no_macro", final_cards),
            ("no_semantic_verifier", initial_cards),
        ):
            entries = rank_cards(
                cards,
                volatility,
                anomaly,
                macro,
                exposures,
                run_config.ranking,
                run_config.task.top_k,
                variant,
            )
            variants[variant] = [entry.symbol for entry in entries]

        # Labels are loaded into the evaluation process only after the immutable run exists.
        panel_symbols = set(labels_frame.filter(pl.col("session") == session)["symbol"].to_list())
        labels = {}
        for row in market_bars.filter(pl.col("session") == session).to_dicts():
            symbol = row["symbol"]
            if symbol not in features or symbol not in panel_symbols:
                continue
            try:
                ratio = range_ratio(float(row["high"]), float(row["low"]))
            except (TypeError, ValueError):
                continue
            if ratio is not None:
                labels[symbol] = ratio
        metrics = {
            name: evaluate_ranking(
                symbols,
                labels,
                k=run_config.task.top_k,
                eligible_count=len(labels),
                large_move_threshold=evaluation_config.evaluation.large_move_threshold,
            )
            for name, symbols in variants.items()
            if name in evaluation_config.evaluation.variants
        }
        screening = json.loads((run_dir / "screening.json").read_text())
        true_large_movers = {
            symbol
            for symbol, value in labels.items()
            if value >= evaluation_config.evaluation.large_move_threshold
        }
        coarse_recall = (
            len(true_large_movers & set(screening["candidates"])) / len(true_large_movers)
            if true_large_movers
            else None
        )
        all_evidence = [*company_evidence_records, *macro.events]
        task_cutoff = datetime.fromisoformat(
            json.loads((run_dir / "task.json").read_text())["decision_cutoff"]
        )
        time_violations = sum(item.available_at > task_cutoff for item in all_evidence)
        fact_claims = [
            (card, claim)
            for card in final_cards
            for claim in card.claims
            if claim.claim_type.value == "observed_fact"
        ]
        evidence_ids = {item.evidence_id for item in all_evidence}
        unsupported_facts = sum(
            not (
                (
                    bool(claim.evidence_ids)
                    and all(evidence_id in evidence_ids for evidence_id in claim.evidence_ids)
                )
                or (
                    bool(claim.feature_ids)
                    and all(
                        feature_id == card.features.feature_id for feature_id in claim.feature_ids
                    )
                )
            )
            for card, claim in fact_claims
        )
        company_by_id = {item.evidence_id: item for item in company_evidence_records}
        event_status_checks = [
            event.status == company_by_id[evidence_id].event_status
            for card in final_cards
            for event in card.events
            for evidence_id in event.evidence_ids
            if evidence_id in company_by_id and company_by_id[evidence_id].event_status is not None
        ]
        revision_distribution = Counter(card.revision_round for card in final_cards)
        abstain_reasons = Counter(
            reason
            for card in final_cards
            if not card.eligibility.eligible
            for reason in card.eligibility.abstain_reasons
        )
        attempts = _read_jsonl(run_dir / "model_attempts.jsonl")

        snapshot_bars = pl.read_parquet(run_dir / "snapshot/market.parquet")
        required_sessions = int(run_config.features["required_complete_sessions"])
        expected_sessions = sorted(snapshot_bars["session"].unique().to_list())[-required_sessions:]
        recomputed, _rejected = compute_features(
            snapshot_bars,
            expected_sessions,
            required_sessions,
            {symbol: item.sector for symbol, item in exposures.items()},
        )
        consistent = sum(
            symbol in recomputed
            and recomputed[symbol].model_dump(mode="json") == feature.model_dump(mode="json")
            for symbol, feature in features.items()
        )
        reliability = {
            "time_boundary_violation_rate": (
                time_violations / len(all_evidence) if all_evidence else 0.0
            ),
            "unsupported_claim_rate": (
                unsupported_facts / len(fact_claims) if fact_claims else 0.0
            ),
            "feature_recompute_consistency_rate": (
                consistent / len(features) if features else None
            ),
            "event_status_accuracy": (
                sum(event_status_checks) / len(event_status_checks) if event_status_checks else None
            ),
            "revision_round_distribution": {
                str(key): value for key, value in sorted(revision_distribution.items())
            },
            "abstain_reason_distribution": dict(sorted(abstain_reasons.items())),
            "model_attempt_count": len(attempts),
            "model_latency_seconds": sum(
                float(item.get("latency_seconds", 0.0)) for item in attempts
            ),
        }
        process_scores = score_run_process(run_dir)
        with report_lock:
            reports.append(
                {
                    "session": session.isoformat(),
                    "replicate_id": str(replicate_id),
                    "run_id": run_dir.name,
                    "status": "completed",
                    "coarse_screen_recall": coarse_recall,
                    "reliability": reliability,
                    "process_scores": process_scores,
                    "metrics": metrics,
                }
            )

    sessions = sorted(evaluation_config.evaluation.sessions)
    work = [
        (replicate_id, session)
        for replicate_id in range(evaluation_config.evaluation.replicates)
        for session in sessions
    ]
    workers = min(evaluation_config.evaluation.session_parallelism, len(work))
    if workers == 1:
        for item in work:
            evaluate_session(item)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            tuple(pool.map(evaluate_session, work))
    reports.sort(key=lambda item: (item["replicate_id"], item["session"]))

    completed = [item for item in reports if item["status"] == "completed"]
    failed = [item for item in reports if item["status"] == "failed"]
    failure_reasons = Counter(item["failure_code"] for item in failed)
    report = {
        "screening_config": str(screening_path),
        "large_move_threshold": evaluation_config.evaluation.large_move_threshold,
        "summary": {
            "session_count": len(reports),
            "completed_session_count": len(completed),
            "failed_session_count": len(failed),
            "failed_run_rate": len(failed) / len(reports) if reports else None,
            "failure_reason_distribution": dict(sorted(failure_reasons.items())),
            "session_coverage": len(completed) / len(reports) if reports else None,
            "average_selection_count": _average_selection_count(completed),
            "process_scores": _process_score_summary(completed),
            "variants": _variant_summary(
                completed,
                evaluation_config.evaluation.variants,
                len(failed),
            ),
            "paired_comparisons": summarize_paired_comparisons(
                completed,
                evaluation_config.evaluation.bootstrap,
            ),
        },
        "sessions": reports,
    }
    output = Path(config.output["runs_path"]) / f"{evaluation_id}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name, dir=output.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
    return output, report


def _process_score_summary(completed_reports: list[dict]) -> dict:
    paths = {
        "overall_process_score": ("overall_process_score",),
        "model_execution_score": ("model_execution", "score"),
        "retrieval_and_tool_decisions_score": ("retrieval_and_tool_decisions", "score"),
        "evidence_and_verification_score": ("evidence_and_verification", "score"),
    }
    result: dict[str, float | int | None | str] = {
        "score_scale": "0_to_1",
        "session_count": len(completed_reports),
    }
    for name, path in paths.items():
        values = []
        for report in completed_reports:
            value = report.get("process_scores", {})
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None:
                values.append(float(value))
        result[f"average_{name}"] = sum(values) / len(values) if values else None
    return result


def _average_selection_count(completed_reports: list[dict]) -> float | None:
    counts = [
        metric["selection_count"]
        for report in completed_reports
        for metric in report["metrics"].values()
    ]
    return sum(counts) / len(counts) if counts else None


def _variant_summary(
    completed_reports: list[dict],
    variants: list[str],
    failed_days: int,
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    metric_names = (
        "precision_at_k",
        "mean_absolute_return_at_k",
        "ndcg_at_k",
    )
    for variant in variants:
        variant_metrics = [
            report["metrics"][variant]
            for report in completed_reports
            if variant in report["metrics"]
        ]
        item: dict = {
            "valid_days": len(variant_metrics),
            "failed_days": failed_days,
        }
        for metric_name in metric_names:
            values = [
                metrics[metric_name]
                for metrics in variant_metrics
                if metrics[metric_name] is not None
            ]
            item[f"average_{metric_name}"] = sum(values) / len(values) if values else None
            item[f"{metric_name}_null_days"] = len(variant_metrics) - len(values)
        summary[variant] = item
    return summary


def summarize_paired_comparisons(records: list[dict], bootstrap) -> dict:
    """Summarize paired dates within each replicate; never pool replicate-days."""

    comparisons = {
        "full_system_vs_no_agent_ranker": ("full_system", "no_agent_ranker"),
        "deterministic_event_baseline_vs_no_agent_ranker": (
            "deterministic_event_baseline",
            "no_agent_ranker",
        ),
    }
    result = {}
    for name, (challenger_name, baseline_name) in comparisons.items():
        result[name] = {}
        for metric in ("ndcg_at_k", "mean_absolute_return_at_k", "precision_at_k"):
            replicates = []
            for replicate_id in sorted({row["replicate_id"] for row in records}):
                rows = [row for row in records if row["replicate_id"] == replicate_id]
                challenger = {
                    row["session"]: row["metrics"].get(challenger_name, {}).get(metric)
                    for row in rows
                }
                baseline = {
                    row["session"]: row["metrics"].get(baseline_name, {}).get(metric)
                    for row in rows
                }
                paired = paired_moving_block_bootstrap(
                    challenger,
                    baseline,
                    seed=bootstrap.seed,
                    samples=bootstrap.samples,
                    block_sizes=bootstrap.block_sizes,
                    confidence=bootstrap.confidence,
                )
                replicates.append({"replicate_id": replicate_id, **paired})
            deltas = [item["mean_delta"] for item in replicates if item["mean_delta"] is not None]
            result[name][metric] = {
                "direction": "positive=challenger_better",
                "replicates": replicates,
                "aggregate_replicate_deltas": _delta_distribution(deltas),
            }
    return result


def _delta_distribution(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5,
        "min": min(values),
        "max": max(values),
    }
