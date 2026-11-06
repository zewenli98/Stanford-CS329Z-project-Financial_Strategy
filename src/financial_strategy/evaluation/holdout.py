"""Accumulating future-holdout ledger with one-shot consumption controls."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from datetime import date
from pathlib import Path

from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.errors import ConfigError


def prepare_future_holdout(
    manifest_path: Path,
    challenger_config: Path,
    *,
    start_session: date,
    as_of: date,
    minimum_sessions: int = 60,
) -> dict:
    """Create or advance an unlabeled holdout while keeping the challenger frozen."""

    challenger_hash = hashlib.sha256(challenger_config.read_bytes()).hexdigest()
    existing = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    )
    if existing is not None:
        if existing["challenger_config_sha256"] != challenger_hash:
            raise ConfigError(
                "HOLDOUT_CHALLENGER_CHANGED",
                "the challenger must remain frozen while the holdout accumulates",
            )
        if existing.get("consumed_at") is not None:
            raise ConfigError("HOLDOUT_ALREADY_CONSUMED", str(manifest_path))
        if date.fromisoformat(existing["start_session"]) != start_session:
            raise ConfigError("HOLDOUT_START_CHANGED", str(manifest_path))

    calendar = TradingCalendar()
    sessions = [
        item["session"]
        for item in calendar.schedule_snapshot(start_session, as_of)
    ]
    manifest = {
        "version": 1,
        "start_session": start_session.isoformat(),
        "as_of": as_of.isoformat(),
        "minimum_sessions": minimum_sessions,
        "completed_sessions": sessions,
        "session_count": len(sessions),
        "ready_for_one_shot_evaluation": len(sessions) >= minimum_sessions,
        "challenger_config": str(challenger_config),
        "challenger_config_sha256": challenger_hash,
        "labels_loaded": False,
        "consumed_at": None,
        "evaluation_report": None,
    }
    _write_atomic(manifest_path, manifest)
    return manifest


def mark_holdout_consumed(
    manifest_path: Path,
    *,
    evaluation_report: Path,
    consumed_at: str,
    promotion_decision: dict | None = None,
) -> dict:
    """Seal a ready holdout after the separate evaluator finishes exactly once."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("consumed_at") is not None:
        raise ConfigError("HOLDOUT_ALREADY_CONSUMED", str(manifest_path))
    if not manifest.get("ready_for_one_shot_evaluation"):
        raise ConfigError(
            "HOLDOUT_NOT_READY",
            f"{manifest['session_count']}/{manifest['minimum_sessions']} sessions",
        )
    manifest["labels_loaded"] = True
    manifest["consumed_at"] = consumed_at
    manifest["evaluation_report"] = str(evaluation_report)
    manifest["promotion_decision"] = promotion_decision
    _write_atomic(manifest_path, manifest)
    return manifest


def assess_holdout_promotion(report: dict) -> dict:
    """Apply the pre-registered one-shot full-system promotion conditions."""

    completed = [item for item in report["sessions"] if item["status"] == "completed"]
    deltas = [
        item["metrics"]["full_system"]["ndcg_at_k"]
        - item["metrics"]["no_agent_ranker"]["ndcg_at_k"]
        for item in completed
        if item["metrics"]["full_system"]["ndcg_at_k"] is not None
        and item["metrics"]["no_agent_ranker"]["ndcg_at_k"] is not None
    ]
    paired_ci = _moving_block_ci(deltas, seed=329, samples=2000, block_size=20)
    variants = report["summary"]["variants"]
    full = variants["full_system"]
    baseline = variants["no_agent_ranker"]
    mean_return_non_degrading = (
        full["average_mean_absolute_return_at_k"]
        >= baseline["average_mean_absolute_return_at_k"]
    )
    full_coverage = [
        item["metrics"]["full_system"]["selection_coverage"] for item in completed
    ]
    baseline_coverage = [
        item["metrics"]["no_agent_ranker"]["selection_coverage"] for item in completed
    ]
    coverage_non_degrading = sum(full_coverage) >= sum(baseline_coverage)
    reliability_clean = all(
        item["reliability"]["time_boundary_violation_rate"] == 0
        and item["reliability"]["unsupported_claim_rate"] == 0
        for item in completed
    )
    lower_ci_positive = paired_ci is not None and paired_ci[0] > 0
    return {
        "promote": bool(
            lower_ci_positive
            and mean_return_non_degrading
            and coverage_non_degrading
            and reliability_clean
        ),
        "paired_ndcg_delta_95ci": paired_ci,
        "mean_return_non_degrading": mean_return_non_degrading,
        "coverage_non_degrading": coverage_non_degrading,
        "reliability_clean": reliability_clean,
        "completed_sessions": len(completed),
    }


def require_holdout_ready(manifest_path: Path) -> dict:
    """Fail before labels or an evaluator are opened unless promotion is allowed."""

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("consumed_at") is not None:
        raise ConfigError("HOLDOUT_ALREADY_CONSUMED", str(manifest_path))
    if not manifest.get("ready_for_one_shot_evaluation"):
        raise ConfigError(
            "HOLDOUT_NOT_READY",
            f"{manifest['session_count']}/{manifest['minimum_sessions']} sessions",
        )
    return manifest


def _moving_block_ci(
    values: list[float],
    *,
    seed: int,
    samples: int,
    block_size: int,
) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    effective_block = min(block_size, len(values))
    starts = range(len(values) - effective_block + 1)
    block_count = math.ceil(len(values) / effective_block)
    estimates = []
    for _ in range(samples):
        sample = []
        for _block in range(block_count):
            start = rng.choice(starts)
            sample.extend(values[start : start + effective_block])
        estimates.append(sum(sample[: len(values)]) / len(values))
    estimates.sort()
    return [
        estimates[int(0.025 * (len(estimates) - 1))],
        estimates[int(0.975 * (len(estimates) - 1))],
    ]


def _write_atomic(path: Path, payload: dict) -> None:
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
