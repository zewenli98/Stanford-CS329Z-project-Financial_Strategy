"""Write-once run publication with content hashes and atomic directory promotion."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl
import yaml
from pydantic import BaseModel

from financial_strategy.errors import AuditError


def _jsonable(value: object) -> object:
    if isinstance(value, BaseModel):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class RunWriter:
    """Accumulate one run under .tmp and publish it exactly once."""

    def __init__(self, runs_root: Path, run_id: str) -> None:
        self.runs_root = runs_root
        self.run_id = run_id
        self.final = runs_root / run_id
        self.temporary = runs_root / ".tmp" / f"{run_id}_{uuid4().hex}"
        if self.final.exists():
            raise AuditError("RUN_ALREADY_EXISTS", str(self.final))
        self.temporary.mkdir(parents=True, exist_ok=False)

    def write_json(self, relative: str, value: object) -> Path:
        path = self._path(relative)
        value = _jsonable(value)
        path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
        self._fsync_file(path)
        return path

    def write_jsonl(self, relative: str, values: list[object]) -> Path:
        path = self._path(relative)
        with path.open("w") as handle:
            for value in values:
                value = _jsonable(value)
                handle.write(json.dumps(value, sort_keys=True, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return path

    def write_yaml(self, relative: str, value: object) -> Path:
        path = self._path(relative)
        value = _jsonable(value)
        path.write_text(yaml.safe_dump(value, sort_keys=True))
        self._fsync_file(path)
        return path

    def write_parquet(self, relative: str, frame: pl.DataFrame) -> Path:
        path = self._path(relative)
        frame.write_parquet(path)
        self._fsync_file(path)
        return path

    def publish(self, status: str, error: dict | None = None) -> Path:
        if status not in {"completed", "failed"}:
            raise AuditError("INVALID_RUN_STATUS", status)
        self.write_json(
            "status.json",
            {
                "run_id": self.run_id,
                "status": status,
                "published_at": datetime.now(UTC).isoformat(),
            },
        )
        if error is not None:
            self.write_json("error.json", error)
        artifacts = []
        for path in sorted(self.temporary.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "manifest.json.sha256"}:
                artifacts.append(
                    {
                        "path": str(path.relative_to(self.temporary)),
                        "sha256": file_sha256(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
        manifest = {
            "run_id": self.run_id,
            "status": status,
            "artifacts": artifacts,
        }
        manifest_path = self.write_json("manifest.json", manifest)
        self._path("manifest.json.sha256").write_text(file_sha256(manifest_path) + "\n")
        self._fsync_file(self.temporary / "manifest.json.sha256")
        self._fsync_directory(self.temporary)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(self.temporary, self.final)
        except FileExistsError as exc:
            raise AuditError("RUN_ALREADY_EXISTS", str(self.final)) from exc
        self._fsync_directory(self.runs_root)
        return self.final

    def abandon(self) -> None:
        shutil.rmtree(self.temporary, ignore_errors=True)

    def _path(self, relative: str) -> Path:
        path = self.temporary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _fsync_file(path: Path) -> None:
        # Windows FlushFileBuffers requires a descriptor opened for writing.
        with path.open("r+b") as handle:
            os.fsync(handle.fileno())

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        # Windows cannot open directories through os.open. Files have already been
        # flushed and directory promotion is still atomic; POSIX also fsyncs metadata.
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def verify_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    digest_path = run_dir / "manifest.json.sha256"
    if not manifest_path.exists() or not digest_path.exists():
        raise AuditError("ARTIFACT_HASH_MISMATCH", "manifest is missing")
    if file_sha256(manifest_path) != digest_path.read_text().strip():
        raise AuditError("ARTIFACT_HASH_MISMATCH", "manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    expected_paths = {artifact["path"] for artifact in manifest["artifacts"]}
    actual_paths = {
        str(path.relative_to(run_dir))
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.name not in {"manifest.json", "manifest.json.sha256"}
    }
    if actual_paths != expected_paths:
        raise AuditError(
            "ARTIFACT_HASH_MISMATCH",
            "run contains missing or unlisted files",
        )
    for artifact in manifest["artifacts"]:
        path = run_dir / artifact["path"]
        if (
            not path.exists()
            or path.stat().st_size != artifact["size_bytes"]
            or file_sha256(path) != artifact["sha256"]
        ):
            raise AuditError("ARTIFACT_HASH_MISMATCH", artifact["path"])
    return manifest
