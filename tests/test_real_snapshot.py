from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import polars as pl

from financial_strategy.data.provenance import verify_snapshot_provenance

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/2026-10-01_2026-10-14"
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


def test_sec_evidence_keeps_its_per_filing_acceptance_timestamp() -> None:
    records = [
        row
        for row in read_jsonl(SNAPSHOT / "company/evidence.jsonl")
        if row["evidence_id"].startswith("sec_")
    ]
    assert records
    for record in records:
        raw = json.loads(
            (SNAPSHOT / "raw/sec" / f"{record['symbol'].lower()}_submissions.json").read_text()
        )
        recent = raw["filings"]["recent"]
        accession = record["evidence_id"].rsplit("_", 1)[-1]
        index = next(
            i
            for i, value in enumerate(recent["accessionNumber"])
            if value.replace("-", "") == accession
        )
        accepted = datetime.fromisoformat(
            recent["acceptanceDateTime"][index].replace("Z", "+00:00")
        )
        assert accepted.tzinfo is not None
        assert accepted == datetime.fromisoformat(record["available_at"])
        assert accepted == datetime.fromisoformat(record["first_seen_at"])
