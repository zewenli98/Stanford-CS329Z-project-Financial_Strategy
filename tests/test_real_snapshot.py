from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

from financial_strategy.data.provenance import verify_snapshot_provenance

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "tests/fixtures/integration"
CURRENT_SNAPSHOT = ROOT / "data/2023-11-01_2026-10-30"
CUTOFF = datetime.fromisoformat("2026-10-14T09:00:00-04:00")
EVALUATION_SESSIONS = [
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
    date(2026, 10, 8),
    date(2026, 10, 9),
    date(2026, 10, 12),
    date(2026, 10, 13),
    date(2026, 10, 14),
]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _matches_acquisition_hash(path: Path, expected: str) -> bool:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() == expected:
        return True
    if path.suffix.lower() not in {".txt", ".json", ".jsonl"}:
        return False
    payload.decode("utf-8", errors="strict")
    return hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest() == expected


def test_real_snapshot_shape_and_adjustment_policy() -> None:
    bars = pl.read_parquet(SNAPSHOT / "market/daily_bars.parquet")
    labels = pl.read_parquet(SNAPSHOT / "evaluation/labels.parquet")

    assert bars.height == 1_380
    assert bars["symbol"].n_unique() == 20
    assert bars["session"].n_unique() == 69
    assert bars["adjustment_policy"].unique().to_list() == ["split_adjusted"]
    assert labels.height == 200
    assert labels["symbol"].n_unique() == 20
    assert labels["session"].unique().sort().to_list() == EVALUATION_SESSIONS


def test_real_snapshot_evidence_is_point_in_time_and_real() -> None:
    evidence = read_jsonl(SNAPSHOT / "company/evidence.jsonl")

    assert len(evidence) >= 8
    assert any(item["source_type"] == "company_release" for item in evidence)
    assert (
        len({item["symbol"] for item in evidence if item["source_type"] == "company_release"}) >= 4
    )
    assert all("example.com" not in item["source_url"] for item in evidence)
    assert all(datetime.fromisoformat(item["available_at"]) <= CUTOFF for item in evidence)
    assert all(item["content_hash"] for item in evidence)


def test_real_snapshot_consensus_has_pre_release_provenance() -> None:
    events = read_jsonl(SNAPSHOT / "macro/events.jsonl")

    assert {event["event_type"] for event in events} == {
        "cpi_yoy",
        "nonfarm_payrolls",
    }
    assert {event["actual"] for event in events} == {3.7, 29.0}
    assert {event["consensus"] for event in events} == {3.6, 84.0}
    for event in events:
        assert event["consensus_source_name"] == "Dow Jones survey via CNBC"
        assert event["consensus_source_url"].startswith("https://www.cnbc.com/")
        assert event["consensus_content_hash"]
        assert datetime.fromisoformat(event["consensus_published_at"]) < datetime.fromisoformat(
            event["event_time"]
        )


def test_real_snapshot_file_hashes_match_provenance() -> None:
    provenance = json.loads((SNAPSHOT / "provenance.json").read_text())
    checked = verify_snapshot_provenance(SNAPSHOT)
    assert len(checked["files"]) == len(provenance["files"])
    verification = {row["path"]: row for row in checked["files"]}
    for relative_path, metadata in provenance["files"].items():
        payload = (SNAPSHOT / relative_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == verification[relative_path]["on_disk_sha256"]
        if verification[relative_path]["verification"] == "git_crlf_only":
            payload = payload.replace(b"\r\n", b"\n")
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        assert len(payload) == metadata["size_bytes"]


def test_expanded_snapshot_has_chronological_train_validation_split() -> None:
    bars = pl.read_parquet(CURRENT_SNAPSHOT / "market/daily_bars.parquet")
    labels = pl.read_parquet(CURRENT_SNAPSHOT / "evaluation/labels.parquet")
    splits = json.loads((CURRENT_SNAPSHOT / "splits.json").read_text())

    assert bars.height == 167_400
    assert bars["session"].n_unique() == 837
    assert labels.height == 150_400
    assert labels["session"].n_unique() == 752
    assert len(splits["train_sessions"]) == 676
    assert len(splits["validation_sessions"]) == 76
    assert max(splits["train_sessions"]) < min(splits["validation_sessions"])
    assert not set(splits["train_sessions"]) & set(splits["validation_sessions"])


def test_expanded_snapshot_has_complete_macro_backfill() -> None:
    company = read_jsonl(CURRENT_SNAPSHOT / "company/evidence.jsonl")
    macro = read_jsonl(CURRENT_SNAPSHOT / "macro/events.jsonl")

    assert len(company) == 14_383
    assert len({item["evidence_id"] for item in company}) == len(company)
    assert all(item["schema_version"] == 1 for item in company)
    taxonomy = {
        "earnings",
        "guidance",
        "merger_acquisition",
        "buyback",
        "dividend",
        "financing",
        "litigation_regulatory",
        "management_change",
        "product",
        "cybersecurity",
    }
    assert all(set(item["event_types"]) <= taxonomy for item in company)
    assert all("event_type" not in item for item in company)
    classification_audit = json.loads(
        (CURRENT_SNAPSHOT / "company/event_classification_audit.json").read_text()
    )
    assert classification_audit["record_count"] == len(company)
    assert classification_audit["network_sources_added"] == 0
    assert classification_audit["documents_modified"] == 0
    document_paths = {item["content_path"] for item in company}
    assert len(document_paths) == len(company)
    assert all(
        (CURRENT_SNAPSHOT / "company" / item["content_path"]).is_file()
        for item in company
    )
    assert all(
        _matches_acquisition_hash(
            CURRENT_SNAPSHOT / "company" / item["content_path"], item["document_content_hash"]
        )
        for item in company
    )
    assert document_paths == {
        f"documents/{path.name}"
        for path in (CURRENT_SNAPSHOT / "company/documents").glob("*.txt")
    }
    release_types = {"cpi_yoy", "nonfarm_payrolls", "unemployment", "fed_rate"}
    daily_types = {"dgs2", "dgs10", "vixcls"}
    releases = [item for item in macro if item["event_type"] in release_types]
    daily = [item for item in macro if item["event_type"] in daily_types]
    assert len(releases) == 171
    assert daily
    assert {item["snapshot_id"] for item in macro} == {
        "2023-11-01_2026-10-30"
    }
    assert {
        event_type: sum(
            item["event_type"] == event_type
            and item["status"] == "released_before_cutoff"
            for item in releases
        )
        for event_type in release_types
    } == {
        "cpi_yoy": 39,
        "nonfarm_payrolls": 39,
        "unemployment": 39,
        "fed_rate": 27,
    }
    assert sum(
        item["status"] == "scheduled_during_session" for item in releases
    ) == 27
    assert all(item["timestamp_confidence"] == "low" for item in macro)
    assert all(
        datetime.fromisoformat(item["consensus_published_at"])
        == datetime.fromisoformat(item["event_time"]) - timedelta(seconds=1)
        for item in releases
        if item["consensus"] is not None
    )
    assert any(
        datetime.fromisoformat(item["event_time"]).date()
        < date(2023, 11, 1)
        for item in macro
    )
    assert all(
        date(2023, 7, 1)
        <= datetime.fromisoformat(item["event_time"]).date()
        <= date(2026, 10, 30)
        for item in macro
    )
    assert all(
        (CURRENT_SNAPSHOT / "macro" / item["content_path"]).is_file()
        for item in releases
        if item["content_path"]
    )
    audit = json.loads(
        (CURRENT_SNAPSHOT / "macro/audit.json").read_text()
    )
    assert audit["distinct_release_sessions"] > 104
    released_counts = audit["released_event_counts"]
    assert released_counts["cpi_yoy"] == 39
    assert released_counts["fed_rate"] == 27
    assert released_counts["nonfarm_payrolls"] == 39
    assert released_counts["unemployment"] == 39
    assert released_counts["dgs2"] > 0
    assert released_counts["dgs10"] > 0
    assert released_counts["vixcls"] > 0
    assert audit["secondary_assumed_pit_count"] > 171
    assert audit["fallback_calendar_row_count"] == 4
    assert audit["fallback_actual_override_count"] == 2
    assert audit["missing_calendar_rows"] == []
    fallback = {
        (item["event_type"], item["event_time"][:10]): (
            item["actual"],
            item["consensus"],
        )
        for item in macro
        if item["source_type"]
        == "official_date_with_fallback_calendar_consensus"
    }
    assert fallback == {
        ("nonfarm_payrolls", "2023-09-01"): (187.0, 170.0),
        ("unemployment", "2023-09-01"): (3.8, 3.5),
        ("cpi_yoy", "2023-10-12"): (3.7, 3.6),
        ("cpi_yoy", "2025-05-13"): (2.3, 2.4),
    }

    manifest = json.loads((CURRENT_SNAPSHOT / "manifest.json").read_text())
    for relative_path, expected_hash in manifest["file_sha256"].items():
        assert _matches_acquisition_hash(CURRENT_SNAPSHOT / relative_path, expected_hash)
