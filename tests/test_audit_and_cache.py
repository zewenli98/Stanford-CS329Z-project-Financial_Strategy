import json
from pathlib import Path

import pytest

from financial_strategy.audit.writer import RunWriter, verify_manifest
from financial_strategy.errors import AuditError
from financial_strategy.llm.cache import AtomicJsonCache


def test_run_writer_publishes_manifest_and_replaces_existing_run(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path, "run_1")
    writer.write_json("result.json", {"value": 1})
    run_dir = writer.publish("completed")
    manifest = verify_manifest(run_dir)
    assert manifest["status"] == "completed"

    replacement = RunWriter(tmp_path, "run_1")
    replacement.write_json("result.json", {"value": 2})
    replacement_dir = replacement.publish("completed")

    assert replacement_dir == run_dir
    assert json.loads((run_dir / "result.json").read_text()) == {"value": 2}
    verify_manifest(run_dir)


def test_manifest_detects_tampering(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path, "run_1")
    writer.write_json("result.json", {"value": 1})
    run_dir = writer.publish("completed")
    (run_dir / "result.json").write_text('{"value": 2}\n')
    with pytest.raises(AuditError) as error:
        verify_manifest(run_dir)
    assert error.value.code == "ARTIFACT_HASH_MISMATCH"


def test_failed_run_is_published_without_watchlist(tmp_path: Path) -> None:
    writer = RunWriter(tmp_path, "run_failed")
    writer.write_json("config.json", {"mode": "scripted"})
    run_dir = writer.publish("failed", {"code": "TEST_FAILURE"})
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert not (run_dir / "watchlist.json").exists()


def test_atomic_cache_keeps_valid_json(tmp_path: Path) -> None:
    cache = AtomicJsonCache(tmp_path / "cache")
    cache.write("key", {"response": {"verdict": "PASS"}})
    cache.write("key", {"response": {"verdict": "ABSTAIN"}})
    assert cache.read("key") == {"response": {"verdict": "PASS"}}
