"""Create and load immutable per-session artifacts without reading evaluation labels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from financial_strategy.audit.writer import file_sha256, verify_manifest
from financial_strategy.config import TuningConfig, content_hash, load_mvp_config
from financial_strategy.errors import ConfigError
from financial_strategy.service import run_screening


def load_splits(path: Path) -> dict:
    payload = json.loads(path.read_text())
    stored_hash = payload.get("content_hash")
    unhashed = {key: value for key, value in payload.items() if key != "content_hash"}
    calculated = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, default=str).encode()
    ).hexdigest()
    if stored_hash != calculated:
        raise ConfigError("SPLIT_HASH_MISMATCH", str(path))
    train = payload.get("train_sessions", [])
    validation = payload.get("validation_sessions", [])
    try:
        parsed_train = [date.fromisoformat(value) for value in train]
        parsed_validation = [date.fromisoformat(value) for value in validation]
    except (TypeError, ValueError) as exc:
        raise ConfigError("INVALID_DATA_SPLIT", "sessions must be ISO dates") from exc
    if (
        payload.get("strategy") != "chronological_holdout"
        or not train
        or not validation
        or len(set(train)) != len(train)
        or len(set(validation)) != len(validation)
        or set(train) & set(validation)
        or parsed_train != sorted(parsed_train)
        or parsed_validation != sorted(parsed_validation)
    ):
        raise ConfigError("INVALID_DATA_SPLIT", "train and validation must be non-empty/disjoint")
    if max(parsed_train) >= min(parsed_validation):
        raise ConfigError("INVALID_DATA_SPLIT", "validation must follow train chronologically")
    total = len(train) + len(validation)
    if "train_fraction" in payload and not math.isclose(
        payload["train_fraction"], len(train) / total
    ):
        raise ConfigError("INVALID_DATA_SPLIT", "train_fraction is inconsistent")
    if "validation_fraction" in payload and not math.isclose(
        payload["validation_fraction"], len(validation) / total
    ):
        raise ConfigError("INVALID_DATA_SPLIT", "validation_fraction is inconsistent")
    return payload


def prepare_artifacts(
    config: TuningConfig,
    *,
    limit: int | None = None,
    split_name: str = "all",
    retry_failed: bool = False,
) -> dict:
    """Run sessions under an exclusive index lock to prevent orphaned concurrent runs."""

    lock_path = config.artifact_index.with_suffix(config.artifact_index.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ConfigError("ARTIFACT_INDEX_LOCKED", str(lock_path)) from exc
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode())
        os.fsync(descriptor)
        return _prepare_artifacts_locked(
            config,
            limit=limit,
            split_name=split_name,
            retry_failed=retry_failed,
        )
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _prepare_artifacts_locked(
    config: TuningConfig,
    *,
    limit: int | None,
    split_name: str,
    retry_failed: bool,
) -> dict:
    """Implementation executed while the artifact index lock is held."""

    splits = load_splits(config.splits)
    if split_name == "train":
        sessions = list(splits["train_sessions"])
    elif split_name == "validation":
        sessions = list(splits["validation_sessions"])
    elif split_name == "all":
        sessions = [*splits["train_sessions"], *splits["validation_sessions"]]
    else:
        raise ConfigError("INVALID_SPLIT_NAME", split_name)
    if limit is not None:
        sessions = sessions[:limit]

    screening = load_mvp_config(config.screening_config)
    if screening.output["runs_path"] != config.runs_path:
        raise ConfigError(
            "RUNS_PATH_MISMATCH",
            "tuning runs_path must match screening output.runs_path",
        )
    universe = json.loads(screening.data.universe.read_text())
    universe_size = len(universe["securities"])
    candidate_capacity = max(
        screening.screening.candidate_multiplier * screening.task.top_k,
        screening.screening.minimum_candidates,
    )
    if candidate_capacity < universe_size:
        raise ConfigError(
            "INCOMPLETE_TUNING_ARTIFACTS",
            "artifact generation must research every eligible universe symbol",
        )

    split_hash = splits["content_hash"]
    screening_hash = _screening_config_hash(screening)
    data_hashes = _screening_data_hashes(screening)
    if config.artifact_index.exists():
        index = json.loads(config.artifact_index.read_text())
        if (
            index.get("version") != 2
            or
            index["split_hash"] != split_hash
            or index["screening_config_hash"] != screening_hash
            or index.get("data_hashes") != data_hashes
        ):
            raise ConfigError("ARTIFACT_INDEX_MISMATCH", str(config.artifact_index))
    else:
        index = {
            "version": 2,
            "split_hash": split_hash,
            "screening_config_hash": screening_hash,
            "data_hashes": data_hashes,
            "model": screening.agents.model,
            "prompt_versions": {
                role: settings.prompt_version
                for role, settings in screening.agents.roles.items()
            },
            "sessions": {},
        }

    pending: list[str] = []
    for session_text in sessions:
        if session_text in index["sessions"]:
            if (
                retry_failed
                and index["sessions"][session_text]["status"] != "completed"
            ):
                pending.append(session_text)
                continue
            run_dir = config.runs_path / index["sessions"][session_text]["run_id"]
            _validate_indexed_run(
                run_dir,
                session_text,
                index["sessions"][session_text],
                screening_hash,
            )
            if "model_usage" not in index["sessions"][session_text]:
                usage = _model_usage(run_dir)
                if usage is not None:
                    index["sessions"][session_text]["model_usage"] = usage
                    _atomic_json(config.artifact_index, index)
            continue
        pending.append(session_text)

    with ThreadPoolExecutor(max_workers=config.session_parallelism) as executor:
        futures = {
            executor.submit(_run_session, config, session_text): session_text
            for session_text in pending
        }
        for future in as_completed(futures):
            session_text, record = future.result()
            index["sessions"][session_text] = record
            _atomic_json(config.artifact_index, index)
    return index


def _screening_data_hashes(screening: object) -> dict[str, str]:
    """Bind an artifact collection to every file-backed prediction input."""

    data = screening.data
    paths = {
        "universe": data.universe,
        "market_bars": data.market_bars,
        "corporate_actions": data.corporate_actions,
        "company_evidence": data.company_evidence,
        "macro_events": data.macro_events,
        "sector_exposures": data.sector_exposures,
    }
    hashes = {name: file_sha256(path) for name, path in sorted(paths.items())}
    hashes["company_documents"] = _evidence_document_hash(
        data.company_evidence
    )
    hashes["macro_documents"] = _evidence_document_hash(data.macro_events)
    return hashes


def _evidence_document_hash(evidence_path: Path) -> str:
    records = [
        json.loads(line)
        for line in evidence_path.read_text().splitlines()
        if line
    ]
    documents = []
    for record in records:
        content_path = record.get("content_path")
        if content_path is None:
            continue
        path = evidence_path.parent / content_path
        documents.append(
            {
                "evidence_id": record["evidence_id"],
                "path": content_path,
                "sha256": file_sha256(path),
            }
        )
    return content_hash(documents)


def _screening_config_hash(screening: object) -> str:
    """Hash invariant workflow settings while excluding per-session task overrides."""

    payload = screening.model_dump(mode="json")
    payload["task"].pop("target_session")
    payload["task"].pop("decision_cutoff")
    # This post-v3 advisory setting does not alter frozen Agent artifacts. Excluding
    # it keeps historical artifacts readable while ranking remains recomputable.
    payload["ranking"].pop("semantic_warning_penalty", None)
    payload["ranking"].pop("market_feature_weights", None)
    return content_hash(payload)


def _validate_indexed_run(
    run_dir: Path,
    session_text: str,
    record: dict,
    screening_hash: str,
) -> dict:
    """Reject a valid-but-substituted run or a run assigned to the wrong date."""

    manifest = verify_manifest(run_dir)
    manifest_hash = file_sha256(run_dir / "manifest.json")
    if record.get("manifest_hash") != manifest_hash:
        raise ConfigError("ARTIFACT_INDEX_MISMATCH", f"{session_text}: manifest hash")
    if manifest.get("status") != "completed" or record.get("status") != "completed":
        raise ConfigError("INCOMPLETE_TUNING_ARTIFACT", session_text)
    task = json.loads((run_dir / "task.json").read_text())
    expected_cutoff = datetime.combine(
        datetime.fromisoformat(session_text).date(),
        time(9, 0),
        ZoneInfo("America/New_York"),
    )
    if (
        task.get("target_session") != session_text
        or datetime.fromisoformat(task["decision_cutoff"]) != expected_cutoff
    ):
        raise ConfigError("ARTIFACT_SESSION_MISMATCH", session_text)
    run_config_hash = _screening_config_hash(load_mvp_config(run_dir / "config.yaml"))
    if run_config_hash != screening_hash:
        raise ConfigError("ARTIFACT_CONFIG_MISMATCH", session_text)
    return manifest


def _run_session(config: TuningConfig, session_text: str) -> tuple[str, dict]:
    session = datetime.fromisoformat(session_text).date()
    cutoff = datetime.combine(
        session,
        time(9, 0),
        ZoneInfo("America/New_York"),
    )
    run_id = f"tuning_{session_text}_{uuid4().hex[:8]}"
    try:
        run_dir, _watchlist = run_screening(
            config.screening_config,
            run_id=run_id,
            task_overrides={"target_session": session, "decision_cutoff": cutoff},
        )
    except RuntimeError:
        run_dir = config.runs_path / run_id
    manifest = verify_manifest(run_dir)
    record = {
        "run_id": run_dir.name,
        "manifest_hash": hashlib.sha256(
            (run_dir / "manifest.json").read_bytes()
        ).hexdigest(),
        "status": manifest["status"],
    }
    error_path = run_dir / "error.json"
    if error_path.exists():
        record["error"] = json.loads(error_path.read_text())
    usage = _model_usage(run_dir)
    if usage is not None:
        record["model_usage"] = usage
    return session_text, record


def _atomic_json(path: Path, payload: dict) -> None:
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


def _model_usage(run_dir: Path) -> dict | None:
    attempts_path = run_dir / "model_attempts.jsonl"
    if not attempts_path.exists():
        return None
    attempts = [
        json.loads(line)
        for line in attempts_path.read_text().splitlines()
        if line
    ]
    return {
        "attempt_count": len(attempts),
        "input_tokens": sum(int(attempt.get("input_tokens", 0)) for attempt in attempts),
        "output_tokens": sum(int(attempt.get("output_tokens", 0)) for attempt in attempts),
        "latency_seconds": sum(
            float(attempt.get("latency_seconds", 0.0)) for attempt in attempts
        ),
    }
