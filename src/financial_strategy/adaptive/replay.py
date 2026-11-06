"""Replay a single audited trajectory; unseen branches are not invented."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import timedelta
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

import polars as pl
from pydantic import ValidationError

from financial_strategy.adaptive.config import load_adaptive_config
from financial_strategy.adaptive.data import RangeFeature
from financial_strategy.adaptive.models import InfoCard
from financial_strategy.audit.writer import verify_manifest
from financial_strategy.config import canonical_json
from financial_strategy.data.gateway import PointInTimeSnapshot
from financial_strategy.errors import AuditError, StrategyError
from financial_strategy.llm.client import InvocationResult
from financial_strategy.schemas import (
    EvidenceRecord,
    MacroEvidenceRecord,
    ScreeningTask,
    SymbolMacroExposure,
)


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class RecordedValidationError(ValidationError):
    def __str__(self):
        return self.recorded_message


class TapeClient:
    is_replay = True

    def __init__(self, records):
        self.lock = Lock()
        self.records = defaultdict(deque)
        for record in records:
            self.records[
                self.key(
                    record["role"], record["payload"], record["schema_name"], record["attempt_type"]
                )
            ].append(record)

    @staticmethod
    def key(role, payload, schema_name, attempt_type):
        return canonical_json([role, payload, schema_name, attempt_type])

    def generate(self, role, payload, schema, *, attempt_type="initial"):
        key = self.key(role, payload, schema.__name__, attempt_type)
        with self.lock:
            if not self.records[key]:
                raise AuditError("REPLAY_UNSEEN_REQUEST", f"unrecorded {role} request")
            record = self.records[key].popleft()
        if record["status"] != "completed":
            message = record.get("error_message", "recorded failure")
            kind = record.get("error_type", "StrategyError")
            if kind == "ValidationError":
                error = RecordedValidationError.from_exception_data(schema.__name__, [])
                error.recorded_message = message
                raise error
            if kind in {
                "TimeoutError",
                "ConnectionError",
                "APITimeoutError",
                "APIConnectionError",
                "RateLimitError",
                "InternalServerError",
            }:
                raise TimeoutError(message)
            if kind == "StrategyError":
                raise StrategyError(record.get("error_code", kind), message)
            # A provider/parser error without .code must remain without .code:
            # forecasting uses that distinction to record INVALID_FORECAST.
            # Turning every error into StrategyError changes the recorded path.
            error = type(kind, (Exception,), {})(message)
            has_code = record.get("error_has_code", record.get("error_code", kind) != kind)
            if has_code:
                error.code = record.get("error_code")
            raise error
        saved = record["result"]
        return InvocationResult(
            response=schema.model_validate(saved["raw"]),
            request_hash=saved["request_hash"],
            source=saved["source"],
            latency_seconds=saved["latency_seconds"],
            raw=saved["raw"],
            attempt_type=saved["attempt_type"],
            role=saved["role"],
            model=saved["model"],
            prompt_version=saved["prompt_version"],
            schema_name=saved["schema_name"],
            input_tokens=saved.get("input_tokens", 0),
            output_tokens=saved.get("output_tokens", 0),
        )

    def assert_consumed(self):
        if any(self.records.values()):
            raise AuditError("REPLAY_UNUSED_RESPONSES", "trajectory did not consume its tape")


def replay_adaptive(run_dir: str | Path) -> list[dict]:
    from financial_strategy.adaptive.service import run_adaptive

    run_dir = Path(run_dir)
    manifest = verify_manifest(run_dir)
    if manifest["status"] != "completed":
        raise AuditError("RUN_NOT_REPLAYABLE", str(run_dir))
    config = load_adaptive_config(run_dir / "config.yaml")
    task = ScreeningTask.model_validate(json.loads((run_dir / "task.json").read_text()))
    universe = json.loads((run_dir / "snapshot/universe.json").read_text())
    metadata = json.loads((run_dir / "snapshot/metadata.json").read_text())
    # JSON preserves only the offset. New runs record the named timezone explicitly.
    timezone = metadata.get("decision_timezone")
    if not timezone and config.calendar["name"] == "XNYS":
        # Older runs have no zone field. A stored, TTL-clamped expiry identifies
        # New York wall-clock arithmetic when it differs from fixed-offset arithmetic.
        cutoff = task.decision_cutoff
        fixed_expiry = cutoff + timedelta(days=config.adaptive.memory_days)
        ny_expiry = cutoff.astimezone(ZoneInfo("America/New_York")) + timedelta(
            days=config.adaptive.memory_days
        )
        saved_cards = json.loads((run_dir / "research_cards.json").read_text())
        if ny_expiry != fixed_expiry and any(
            InfoCard.model_validate(card).expires_at == ny_expiry for card in saved_cards
        ):
            timezone = "America/New_York"
    if timezone:
        task = task.model_copy(
            update={"decision_cutoff": task.decision_cutoff.astimezone(ZoneInfo(timezone))}
        )
    snapshot = PointInTimeSnapshot(
        snapshot_id=metadata["snapshot_id"],
        bars=pl.read_parquet(run_dir / "snapshot/market.parquet"),
        company_evidence=tuple(
            EvidenceRecord.model_validate(row)
            for row in _jsonl(run_dir / "snapshot/company_evidence.jsonl")
        ),
        macro_evidence=tuple(
            MacroEvidenceRecord.model_validate(row)
            for row in _jsonl(run_dir / "snapshot/macro_evidence.jsonl")
        ),
        corporate_actions=(),
        exposures=tuple(
            SymbolMacroExposure.model_validate(row)
            for row in json.loads((run_dir / "snapshot/exposures.json").read_text())
        ),
        calendar_schedule=tuple(json.loads((run_dir / "snapshot/calendar.json").read_text())),
        calendar_version=metadata["calendar_version"],
    )
    features = {
        key: RangeFeature.model_validate(row)
        for key, row in json.loads((run_dir / "range_features.json").read_text()).items()
    }
    cards = [
        InfoCard.model_validate(row)
        for row in json.loads((run_dir / "memory_initial.json").read_text())
    ]
    tape = TapeClient(_jsonl(run_dir / "model_calls.jsonl"))
    # A new audit is written under the original run's parent; the original is never modified.
    config = config.model_copy(update={"output": {"runs_path": run_dir.parent}})
    rebuilt_dir, rebuilt = run_adaptive(
        config,
        _replay_inputs=(task, snapshot, features, universe, cards, run_dir / "snapshot"),
        _replay_base=tape,
    )
    tape.assert_consumed()
    for artifact in (
        "watchlist.json",
        "range_predictions.json",
        "allocation_decisions.jsonl",
        "research_results.jsonl",
        "forecast_revisions.jsonl",
        "tool_traces.jsonl",
    ):
        if (run_dir / artifact).read_text() != (rebuilt_dir / artifact).read_text():
            raise AuditError("REPLAY_MISMATCH", artifact)
    return rebuilt
