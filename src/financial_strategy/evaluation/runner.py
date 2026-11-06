"""Create fresh historical predictions, freeze them, then load labels."""

from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from datetime import datetime, time
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl

from financial_strategy.config import load_evaluation_config, load_mvp_config
from financial_strategy.evaluation.baselines import baseline_rankings
from financial_strategy.evaluation.metrics import evaluate_ranking
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
    evaluation_config = load_evaluation_config(path)
    screening_path = evaluation_config.screening_config
    config = load_mvp_config(screening_path)
    labels_frame = pl.read_parquet(evaluation_config.evaluation.labels)
    reports: list[dict] = []

    for session in sorted(evaluation_config.evaluation.sessions):
        cutoff = datetime.combine(
            session,
            time.fromisoformat(evaluation_config.evaluation.cutoff_time),
            ZoneInfo(evaluation_config.evaluation.timezone),
        )
        run_id = f"eval_{session.isoformat()}_{uuid4().hex[:8]}"
        try:
            run_dir, _watchlist = run_screening(
                screening_path,
                run_id=run_id,
                task_overrides={"target_session": session, "decision_cutoff": cutoff},
            )
        except RuntimeError:
            run_dir = Path(config.output["runs_path"]) / run_id
            error_path = run_dir / "error.json"
            error = json.loads(error_path.read_text()) if error_path.exists() else {}
            reports.append(
                {
                    "session": session.isoformat(),
                    "run_id": run_id,
                    "status": "failed",
                    "failure_code": error.get("code", "UNKNOWN_FAILURE"),
                    "metrics": {},
                }
            )
            continue
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
        session_labels = labels_frame.filter(pl.col("session") == session)
        labels = {
            row["symbol"]: abs(row["close"] / row["open"] - 1.0)
            for row in session_labels.to_dicts()
            if row["symbol"] in features
        }
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
        time_violations = sum(
            item.available_at > task_cutoff for item in all_evidence
        )
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
                    and all(
                        evidence_id in evidence_ids
                        for evidence_id in claim.evidence_ids
                    )
                )
                or (
                    bool(claim.feature_ids)
                    and all(
                        feature_id == card.features.feature_id
                        for feature_id in claim.feature_ids
                    )
                )
            )
            for card, claim in fact_claims
        )
        company_by_id = {item.evidence_id: item for item in company_evidence_records}
        event_status_checks = [
            event.status
            == company_by_id[evidence_id].event_status
            for card in final_cards
            for event in card.events
            for evidence_id in event.evidence_ids
            if evidence_id in company_by_id
            and company_by_id[evidence_id].event_status is not None
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
        expected_sessions = sorted(snapshot_bars["session"].unique().to_list())[
            -required_sessions:
        ]
        recomputed, _rejected = compute_features(
            snapshot_bars,
            expected_sessions,
            required_sessions,
            {symbol: item.sector for symbol, item in exposures.items()},
        )
        consistent = sum(
            symbol in recomputed
            and recomputed[symbol].model_dump(mode="json")
            == feature.model_dump(mode="json")
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
                sum(event_status_checks) / len(event_status_checks)
                if event_status_checks
                else None
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
        reports.append(
            {
                "session": session.isoformat(),
                "run_id": run_dir.name,
                "status": "completed",
                "coarse_screen_recall": coarse_recall,
                "reliability": reliability,
                "metrics": metrics,
            }
        )

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
            "variants": _variant_summary(
                completed,
                evaluation_config.evaluation.variants,
                len(failed),
            ),
        },
        "sessions": reports,
    }
    output = Path(config.output["runs_path"]) / f"evaluation_{uuid4().hex[:10]}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=output.name, dir=output.parent)
    with os.fdopen(fd, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, output)
    return output, report


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
            item[f"average_{metric_name}"] = (
                sum(values) / len(values) if values else None
            )
            item[f"{metric_name}_defined_days"] = len(values)
            item[f"{metric_name}_null_days"] = len(variant_metrics) - len(values)
        summary[variant] = item
    return summary
