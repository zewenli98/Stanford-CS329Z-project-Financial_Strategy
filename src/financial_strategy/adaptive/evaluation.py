"""Chronological, isolated range-ranking comparisons with post-publication labels."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from collections.abc import Callable
from datetime import date, datetime, time
from pathlib import Path
from typing import Literal
from uuid import uuid4
from zoneinfo import ZoneInfo

import yaml
from pydantic import Field, model_validator

from financial_strategy.adaptive.data import derive_range_labels
from financial_strategy.audit import RunWriter
from financial_strategy.audit.writer import verify_manifest
from financial_strategy.config import ConfigModel
from financial_strategy.errors import AuditError, ConfigError

POLICIES = (
    "baseline",
    "history_only",
    "uniform",
    "random",
    "boundary",
    "lexicographic",
    "scenario",
)


class AdaptiveEvaluationBody(ConfigModel):
    sessions: list[date] = Field(min_length=1)
    cutoff_time: str = "09:00:00"
    timezone: str = "America/New_York"
    # Accepted for migration, never read: legacy labels contain open/close only.
    labels: Path | None = None
    large_move_threshold: float = Field(default=0.02, gt=0)
    variants: list[str] | None = None
    ndcg: dict[str, str | None] | None = None
    policies: list[
        Literal[
            "baseline", "history_only", "uniform", "random", "boundary", "lexicographic", "scenario"
        ]
    ] = Field(default_factory=lambda: list(POLICIES), min_length=1)
    budgets_usd: list[float] = Field(default_factory=lambda: [0.5, 1.0, 2.0], min_length=1)
    replicates: int = Field(default=1, ge=1)
    aggregate_budget_usd: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    report_top_ks: list[int] = Field(default_factory=lambda: [3, 5, 10], min_length=1)

    @model_validator(mode="after")
    def check_grid(self) -> AdaptiveEvaluationBody:
        if len(self.sessions) != len(set(self.sessions)):
            raise ValueError("Evaluation sessions must be unique")
        if len(self.policies) != len(set(self.policies)):
            raise ValueError("Evaluation policies must be unique")
        if any(not math.isfinite(value) or value <= 0 for value in self.budgets_usd):
            raise ValueError("Daily budgets must be positive finite dollar amounts")
        if len(self.budgets_usd) != len(set(self.budgets_usd)):
            raise ValueError("Evaluation budgets must be unique")
        if len(self.report_top_ks) != len(set(self.report_top_ks)) or any(
            value <= 0 for value in self.report_top_ks
        ):
            raise ValueError("Reporting cutoffs must be unique positive integers")
        time.fromisoformat(self.cutoff_time)
        ZoneInfo(self.timezone)
        return self


class AdaptiveEvaluationConfig(ConfigModel):
    screening_config: Path
    evaluation: AdaptiveEvaluationBody


def evaluate_range_ranking(
    predictions: list[dict],
    labels: dict[str, float],
    *,
    k: int,
    large_move_threshold: float = 0.02,
) -> dict:
    """Evaluate full forecasts, preserving zero-filled unlabelled rank positions.

    Top-k regret is the oracle's mean realised range minus the selected mean range.
    Both divide by min(k, labelled eligible universe size); missing selected positions
    contribute zero. MAE uses only labelled forecasts and reports its sample count.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    by_symbol: dict[str, float] = {}
    for row in predictions:
        symbol, value = str(row["symbol"]), float(row["predicted_range"])
        if symbol in by_symbol:
            raise ValueError(f"Duplicate forecast for {symbol}")
        if not math.isfinite(value) or value < 0:
            raise ValueError("Range forecasts must be finite and non-negative")
        by_symbol[symbol] = value
    if any(not math.isfinite(value) or value < 0 for value in labels.values()):
        raise ValueError("Range labels must be finite and non-negative")
    ordered = sorted(by_symbol, key=lambda symbol: (-by_symbol[symbol], symbol))
    eligible_labels = {symbol: labels[symbol] for symbol in by_symbol if symbol in labels}
    k_effective = min(k, len(eligible_labels))
    common = sorted(eligible_labels)
    if k_effective == 0:
        return {
            "k_effective": 0,
            "selection_count": 0,
            "label_count": 0,
            "forecast_count": len(by_symbol),
            "label_coverage": 0.0 if by_symbol else None,
            "top_k_regret": None,
            "mean_selected_range": None,
            "oracle_mean_range": None,
            "top_k_overlap": None,
            "ndcg_at_k": None,
            "mae": None,
            "large_range_precision_at_k": None,
            "missing_selected_label_count": 0,
        }
    selected = ordered[:k_effective]
    values = [eligible_labels.get(symbol, 0.0) for symbol in selected]
    values += [0.0] * (k_effective - len(values))
    oracle = sorted(eligible_labels, key=lambda symbol: (-eligible_labels[symbol], symbol))[
        :k_effective
    ]
    oracle_mean = sum(eligible_labels[symbol] for symbol in oracle) / k_effective
    selected_mean = sum(values) / k_effective
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(values))
    ideal_dcg = sum(
        eligible_labels[symbol] / math.log2(index + 2) for index, symbol in enumerate(oracle)
    )
    return {
        "k_effective": k_effective,
        "selection_count": len(selected),
        "label_count": len(common),
        "forecast_count": len(by_symbol),
        "label_coverage": len(common) / len(by_symbol),
        "top_k_regret": max(0.0, oracle_mean - selected_mean),
        "mean_selected_range": selected_mean,
        "oracle_mean_range": oracle_mean,
        "top_k_overlap": len(set(selected) & set(oracle)) / k_effective,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else None,
        "mae": sum(abs(by_symbol[symbol] - labels[symbol]) for symbol in common) / len(common),
        "large_range_precision_at_k": sum(value >= large_move_threshold for value in values)
        / k_effective,
        "missing_selected_label_count": sum(symbol not in labels for symbol in selected),
    }


def evaluate_range_cutoffs(
    predictions: list[dict],
    labels: dict[str, float],
    *,
    top_ks: list[int],
    large_move_threshold: float = 0.02,
) -> dict[str, dict]:
    """Score prefixes of one forecast ordering against the full eligible universe.

    In particular, top-3 is not the hindsight best three inside the selected ten.
    MAE remains a full-universe metric at every cutoff.
    """
    return {
        str(k): evaluate_range_ranking(
            predictions, labels, k=k, large_move_threshold=large_move_threshold
        )
        for k in sorted(set(top_ks))
    }


def _write_csv(writer: RunWriter, name: str, rows: list[dict]) -> None:
    if not rows:
        return
    path = writer.temporary / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        output = csv.DictWriter(stream, fieldnames=list(rows[0]))
        output.writeheader()
        output.writerows(rows)


def load_published_predictions(run_dir: Path) -> list[dict]:
    """Require a successfully published, integrity-checked forecast before labels."""
    manifest = verify_manifest(run_dir)
    if manifest["status"] != "completed":
        raise AuditError("PREDICTIONS_NOT_FROZEN", str(run_dir))
    path = run_dir / "range_predictions.json"
    listed = {item["path"].replace("\\", "/") for item in manifest["artifacts"]}
    if "range_predictions.json" not in listed or not path.exists():
        raise AuditError("PREDICTIONS_NOT_FROZEN", "Missing manifested range_predictions.json")
    rows = json.loads(path.read_text())
    if not isinstance(rows, list):
        raise AuditError("INVALID_RANGE_PREDICTIONS", str(path))
    return rows


def _json_artifact(run_dir: Path, name: str, default):
    path = run_dir / name
    return json.loads(path.read_text()) if path.is_file() else default


def _jsonl_artifact(run_dir: Path, name: str) -> list[dict]:
    path = run_dir / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_run_integrity(run_dir: Path, forecasts: list[dict]) -> dict:
    """Report measured cost and auditable card-level proxies, not invented truth.

    Unsupported-observation rates below refer to verifier rejection codes. They are
    not independently established factual-error rates. Prior-day cards and duplicate
    rejections are separated from newly accepted evidence yield.
    """
    budget = _json_artifact(run_dir, "budget.json", {})
    attempts = _jsonl_artifact(run_dir, "model_attempts.jsonl")
    calls = _jsonl_artifact(run_dir, "model_calls.jsonl")
    cards = _json_artifact(run_dir, "research_cards.json", [])
    rejected = _json_artifact(run_dir, "rejected_cards.json", [])
    prior = _json_artifact(run_dir, "memory_initial.json", [])
    results = _jsonl_artifact(run_dir, "research_results.jsonl")
    revisions = _jsonl_artifact(run_dir, "forecast_revisions.jsonl")
    tools = _jsonl_artifact(run_dir, "tool_traces.jsonl")
    forecast_issues = _json_artifact(run_dir, "forecast_issues.json", [])
    codes: Counter[str] = Counter()
    unsupported_cards = 0
    unsupported_observation_count = 0
    for card in rejected:
        issue_codes = {
            str(issue).split(":", 1)[0].strip() for issue in card.get("verification_issues", [])
        }
        codes.update(issue_codes)
        if any(
            code.startswith("UNSUPPORTED_")
            or code
            in {
                "MISSING_EVIDENCE",
                "INVALID_SOURCE_SPAN",
                "SOURCE_SPAN_MISMATCH",
                "UNREAD_SOURCE",
                "WRONG_CITATION",
            }
            for code in issue_codes
        ):
            unsupported_cards += 1
            unsupported_observation_count += len(card.get("observations", []))
    prior_ids = {(card["card_id"], card.get("version", 1)) for card in prior}
    accepted = [card for card in cards if card.get("status") == "accepted"]
    unique_new = {
        (card["card_id"], card.get("version", 1))
        for card in accepted
        if (card["card_id"], card.get("version", 1)) not in prior_ids
    }
    observed = sum(len(card.get("observations", [])) for card in [*accepted, *rejected])
    resolved = sum(bool(row.get("accepted_card_ids")) for row in results)
    changed = 0
    for revision in revisions:
        before = {row["symbol"]: row.get("predicted_range") for row in revision.get("before", [])}
        changed += sum(
            before.get(row["symbol"]) != row.get("predicted_range")
            for row in revision.get("after", [])
        )
    normalized = budget.get("normalized_uncached_usd")
    actual = budget.get("actual_usd")
    return {
        "stop_reason": _json_artifact(run_dir, "execution.json", {}).get("stop_reason"),
        "normalized_uncached_usd": normalized,
        "actual_usd": actual,
        "budget_cap_usd": budget.get("cap_usd"),
        "budget_blocked": budget.get("blocked"),
        "model_attempt_count": len(attempts),
        "model_call_count": len(calls),
        "failed_model_attempt_count": sum(row.get("status") == "failed" for row in attempts),
        "model_latency_seconds": sum(float(row.get("latency_seconds", 0)) for row in attempts),
        "model_input_tokens": sum(int(row.get("input_tokens", 0)) for row in attempts),
        "model_output_tokens": sum(int(row.get("output_tokens", 0)) for row in attempts),
        "estimated_usage_call_count": sum(bool(row.get("usage_estimated")) for row in calls),
        "tool_call_count": len(tools),
        "accepted_card_count": len(accepted),
        "prior_memory_card_count": len(prior),
        "new_unique_accepted_card_count": len(unique_new),
        "rejected_card_count": len(rejected),
        "duplicate_rejection_count": codes.get("DUPLICATE_EVENT", 0),
        "unsupported_rejected_card_count": unsupported_cards,
        "unsupported_observation_proxy_count": unsupported_observation_count,
        "observed_card_observation_count": observed,
        "unsupported_observation_proxy_rate": (
            unsupported_observation_count / observed if observed else None
        ),
        "rejection_code_distribution": dict(sorted(codes.items())),
        "research_question_attempt_count": len(results),
        "accepted_evidence_resolution_count": resolved,
        "accepted_evidence_resolution_rate": resolved / len(results) if results else None,
        "accepted_card_yield_per_normalized_usd": (
            len(unique_new) / normalized if normalized is not None and normalized > 0 else None
        ),
        "forecast_update_attempt_count": len(revisions),
        "failed_forecast_update_count": sum(bool(row.get("error")) for row in revisions),
        "changed_forecast_count": changed,
        "final_forecast_revision_count": sum(int(row.get("revision", 0)) for row in forecasts),
        "forecast_issue_count": len(forecast_issues),
        "forecast_issue_distribution": dict(
            sorted(Counter(row.get("code", "UNKNOWN") for row in forecast_issues).items())
        ),
        "time_boundary_rejection_count": codes.get("TIME_BOUNDARY_VIOLATION", 0),
        "measurement_note": (
            "Resolution means an accepted evidence card, not realised-event truth. Unsupported "
            "observations are verifier rejection proxies; latency sums model attempts, "
            "not wall time."
        ),
    }


def _load_config(path: str | Path) -> AdaptiveEvaluationConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text())
        return AdaptiveEvaluationConfig.model_validate(raw)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConfigError("CONFIG_INVALID", str(exc)) from exc


def run_adaptive_evaluation(
    path: str | Path, *, progress: Callable[[dict], None] | None = None
) -> tuple[Path, dict]:
    """Actually execute each policy/budget trajectory with an isolated memory.

    No labels are opened until a completed run's full forecast artifact has been
    verified. Shared raw inputs are allowed; mutable research memory is never shared
    between arms. Live mode reserves the entire grid's daily caps in advance.
    """
    from financial_strategy.adaptive.config import load_adaptive_config
    from financial_strategy.adaptive.service import run_adaptive

    spec = _load_config(path)
    config = load_adaptive_config(spec.screening_config)
    body = spec.evaluation
    if "report_top_ks" in body.model_fields_set and max(body.report_top_ks) > config.task.top_k:
        raise ConfigError("INVALID_REPORT_CUTOFF", "Reporting cutoffs must fit the watchlist")
    report_top_ks = sorted(
        {config.task.top_k, *(k for k in body.report_top_ks if k <= config.task.top_k)}
    )
    research_policies = [policy for policy in body.policies if policy != "baseline"]
    maximum_spend = (
        len(body.sessions) * body.replicates * len(research_policies) * sum(body.budgets_usd)
    )
    if config.agents.mode == "live":
        if body.aggregate_budget_usd is None:
            raise ConfigError("AGGREGATE_BUDGET_REQUIRED", "Live comparisons need an aggregate cap")
        if maximum_spend > body.aggregate_budget_usd + 1e-12:
            raise ConfigError(
                "AGGREGATE_BUDGET_TOO_SMALL",
                f"Full grid reserves ${maximum_spend:.2f}; cap is ${body.aggregate_budget_usd:.2f}",
            )
    evaluation_id = f"adaptive_evaluation_{uuid4().hex[:12]}"
    # Replicate numbers are local to an experiment, including response caches.
    config = config.model_copy(
        update={
            "agents": config.agents.model_copy(
                update={
                    "cache_path": config.agents.cache_path / evaluation_id,
                }
            )
        }
    )
    writer = RunWriter(Path(config.output["runs_path"]), evaluation_id)
    records: list[dict] = []
    coverage_records: list[dict] = []
    writer.write_json("evaluation_config.json", spec)
    writer.write_json("screening_config.json", config)
    try:
        arm_index = 0
        for replicate in range(body.replicates):
            for policy in body.policies:
                for budget in sorted(body.budgets_usd):
                    arm_id = f"{policy}_budget_{budget:g}_replicate_{replicate}"
                    # The service appends its own descriptive arm name. Keep this
                    # unique parent short enough for Windows filesystem limits.
                    memory_path = writer.temporary / "m" / str(arm_index)
                    arm_index += 1
                    adaptive = config.adaptive.model_copy(
                        update={
                            "policy": policy,
                            "daily_budget_usd": budget,
                            "replicate_id": str(replicate),
                            "seed": config.adaptive.seed + replicate,
                            "memory_path": memory_path,
                        }
                    )
                    arm_config = config.model_copy(update={"adaptive": adaptive})
                    for session in sorted(body.sessions):
                        cutoff = datetime.combine(
                            session, time.fromisoformat(body.cutoff_time), ZoneInfo(body.timezone)
                        )
                        run_id = f"ar_{uuid4().hex[:12]}"
                        if progress:
                            progress(
                                {
                                    "event": "session_started",
                                    "session": str(session),
                                    "policy": policy,
                                    "run_id": run_id,
                                }
                            )
                        try:
                            run_dir, _ = run_adaptive(
                                arm_config,
                                run_id=run_id,
                                task_overrides={
                                    "target_session": session,
                                    "decision_cutoff": cutoff,
                                },
                            )
                            forecasts = load_published_predictions(run_dir)
                        except (RuntimeError, AuditError) as exc:
                            failed_dir = Path(config.output["runs_path"]) / run_id
                            failed_integrity = {}
                            if failed_dir.is_dir():
                                try:
                                    verify_manifest(failed_dir)
                                    failed_integrity = summarize_run_integrity(failed_dir, [])
                                except AuditError:
                                    failed_integrity = {"artifact_integrity_verified": False}
                            records.append(
                                {
                                    "arm": arm_id,
                                    "policy": policy,
                                    "budget_usd": budget,
                                    "replicate_id": str(replicate),
                                    "session": session.isoformat(),
                                    "run_id": run_id,
                                    "status": "failed",
                                    "error": str(exc),
                                    "metrics": {},
                                    "integrity": failed_integrity,
                                }
                            )
                            writer.write_json("progress.json", records)
                            if progress:
                                progress({"event": "session_failed", **records[-1]})
                            continue
                        # Intentionally only here: immutable predictions already exist.
                        label_frame, coverage = derive_range_labels(
                            arm_config.data.market_bars,
                            [session],
                            [row["symbol"] for row in forecasts],
                        )
                        labels = {row["symbol"]: row["range"] for row in label_frame.to_dicts()}
                        metrics = evaluate_range_ranking(
                            forecasts,
                            labels,
                            k=arm_config.task.top_k,
                            large_move_threshold=body.large_move_threshold,
                        )
                        metrics_by_k = evaluate_range_cutoffs(
                            forecasts,
                            labels,
                            top_ks=report_top_ks,
                            large_move_threshold=body.large_move_threshold,
                        )
                        ordered = sorted(
                            forecasts, key=lambda row: (-row["predicted_range"], row["symbol"])
                        )
                        status = (
                            "missing_range_labels"
                            if not labels
                            else "partial_range_labels"
                            if len(labels) < len(forecasts)
                            else "completed"
                        )
                        record = {
                            "arm": arm_id,
                            "policy": policy,
                            "budget_usd": budget,
                            "replicate_id": str(replicate),
                            "session": session.isoformat(),
                            "run_id": run_id,
                            "status": status,
                            "metrics": metrics,
                            "metrics_by_k": metrics_by_k,
                            "watchlist": [
                                {
                                    "rank": index + 1,
                                    "symbol": row["symbol"],
                                    "predicted_range": row["predicted_range"],
                                    "realized_range": labels.get(row["symbol"]),
                                }
                                for index, row in enumerate(ordered[: arm_config.task.top_k])
                            ],
                            "integrity": summarize_run_integrity(run_dir, forecasts),
                        }
                        records.append(record)
                        writer.write_json("progress.json", records)
                        if progress:
                            progress({"event": "session_completed", **record})
                        coverage_records.append({"run_id": run_id, **coverage})
        summary: dict[str, dict] = {}
        for arm in sorted({record["arm"] for record in records}):
            rows = [record for record in records if record["arm"] == arm]
            usable = [
                record
                for record in rows
                if record["status"] in {"completed", "partial_range_labels"}
            ]
            metric_names = (
                "top_k_regret",
                "mean_selected_range",
                "oracle_mean_range",
                "top_k_overlap",
                "ndcg_at_k",
                "mae",
                "large_range_precision_at_k",
            )
            summary[arm] = {
                "session_count": len(rows),
                "scored_session_count": len(usable),
                "complete_label_session_count": sum(
                    record["status"] == "completed" for record in rows
                ),
                "partial_label_session_count": sum(
                    record["status"] == "partial_range_labels" for record in rows
                ),
                "failed_session_count": sum(record["status"] == "failed" for record in rows),
                "missing_label_session_count": sum(
                    record["status"] == "missing_range_labels" for record in rows
                ),
            }
            for name in metric_names:
                values = [
                    row["metrics"][name] for row in usable if row["metrics"][name] is not None
                ]
                summary[arm][f"average_{name}"] = sum(values) / len(values) if values else None
                summary[arm][f"{name}_defined_session_count"] = len(values)
            summary[arm]["metrics_by_k"] = {}
            for k in report_top_ks:
                cutoff_summary = {}
                for name in metric_names:
                    values = [
                        row["metrics_by_k"][str(k)][name]
                        for row in usable
                        if row["metrics_by_k"][str(k)][name] is not None
                    ]
                    cutoff_summary[f"average_{name}"] = (
                        sum(values) / len(values) if values else None
                    )
                    cutoff_summary[f"{name}_defined_session_count"] = len(values)
                summary[arm]["metrics_by_k"][str(k)] = cutoff_summary
            for name in (
                "normalized_uncached_usd",
                "actual_usd",
                "model_call_count",
                "model_attempt_count",
                "model_latency_seconds",
                "tool_call_count",
                "new_unique_accepted_card_count",
                "rejected_card_count",
                "duplicate_rejection_count",
                "unsupported_rejected_card_count",
                "unsupported_observation_proxy_count",
                "observed_card_observation_count",
                "research_question_attempt_count",
                "accepted_evidence_resolution_count",
                "forecast_update_attempt_count",
                "failed_forecast_update_count",
                "changed_forecast_count",
                "final_forecast_revision_count",
                "time_boundary_rejection_count",
            ):
                values = [
                    row["integrity"][name]
                    for row in rows
                    if row.get("integrity", {}).get(name) is not None
                ]
                summary[arm][f"total_{name}"] = sum(values) if values else None
            attempted = summary[arm]["total_research_question_attempt_count"]
            resolved = summary[arm]["total_accepted_evidence_resolution_count"]
            normalized = summary[arm]["total_normalized_uncached_usd"]
            unique_cards = summary[arm]["total_new_unique_accepted_card_count"]
            summary[arm]["accepted_evidence_resolution_rate"] = (
                resolved / attempted if attempted else None
            )
            summary[arm]["accepted_card_yield_per_normalized_usd"] = (
                unique_cards / normalized if normalized and unique_cards is not None else None
            )
            observations = summary[arm]["total_observed_card_observation_count"]
            unsupported = summary[arm]["total_unsupported_observation_proxy_count"]
            summary[arm]["unsupported_observation_proxy_rate"] = (
                unsupported / observations if observations else None
            )
            summary[arm]["stop_reason_distribution"] = dict(
                sorted(
                    Counter(
                        row.get("integrity", {}).get("stop_reason") or "unavailable" for row in rows
                    ).items()
                )
            )
        report = {
            "workflow": "adaptive_range_v1",
            "evaluation_id": evaluation_id,
            "target": "(high-low)/low",
            "primary_metric": "top_k_regret",
            "report_top_ks": report_top_ks,
            "cutoff_note": (
                "Top-3/5/10 are prefixes of the same predicted ordering; "
                "each oracle uses the full eligible universe."
            ),
            "cost_note": (
                "actual_usd uses priced token usage for live responses and conservative "
                "bounds for failed or usage-missing attempts; it is not an independently "
                "reconciled provider invoice."
            ),
            "reserved_live_budget_usd": maximum_spend,
            "aggregate_budget_usd": body.aggregate_budget_usd,
            "replication_scope": (
                "Independent policy/replicate trajectories; exact requests may share cache "
                "within a replicate."
            ),
            "label_source": "existing frozen market high/low; no open/close substitution",
            "partial_label_note": (
                "Partial sessions are flagged and scored on the labelled eligible universe; "
                "missing selected rank positions contribute zero. Each mean reports "
                "its denominator."
            ),
            "summary": summary,
            "sessions": records,
        }
        writer.write_json("report.json", report)
        writer.write_json("label_coverage.json", coverage_records)
        _write_csv(
            writer,
            "daily_costs.csv",
            [
                {
                    "session": row["session"],
                    "policy": row["policy"],
                    "run_id": row["run_id"],
                    "status": row["status"],
                    "daily_cap_usd": row["budget_usd"],
                    **{
                        key: row.get("integrity", {}).get(key)
                        for key in (
                            "actual_usd",
                            "normalized_uncached_usd",
                            "model_input_tokens",
                            "model_output_tokens",
                            "failed_model_attempt_count",
                            "estimated_usage_call_count",
                            "changed_forecast_count",
                            "new_unique_accepted_card_count",
                            "stop_reason",
                        )
                    },
                }
                for row in records
            ],
        )
        _write_csv(
            writer,
            "daily_metrics.csv",
            [
                {
                    "session": row["session"],
                    "policy": row["policy"],
                    "run_id": row["run_id"],
                    "top_k": int(k),
                    **metrics,
                }
                for row in records
                for k, metrics in row.get("metrics_by_k", {}).items()
            ],
        )
        _write_csv(
            writer,
            "daily_watchlists.csv",
            [
                {
                    "session": row["session"],
                    "policy": row["policy"],
                    "run_id": row["run_id"],
                    **stock,
                }
                for row in records
                for stock in row.get("watchlist", [])
            ],
        )
        output = writer.publish("completed")
        return output / "report.json", report
    except Exception as exc:
        writer.publish(
            "failed",
            {
                "code": getattr(exc, "code", "EVALUATION_FAILED"),
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )
        raise
