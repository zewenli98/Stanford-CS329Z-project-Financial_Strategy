"""Add FRED daily closes to the historical macro event file.

DGS2, DGS10, and VIXCLS are session closes, not survey releases. Each printed
session in the collection window becomes one macro event. ``actual`` is that
close. ``consensus`` is the previous printed close, so the existing surprise
score is the daily change. The clock is the cash close in New York, which is
knowable before the next morning's decision and not during the same morning.
The CSV has no publication timestamp, so confidence stays low.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime, time
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from financial_strategy.schemas import MacroEvidenceRecord

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data/2023-11-01_2026-10-30"
MACRO = SNAPSHOT / "macro"
WINDOW_START = date(2023, 7, 1)
WINDOW_END = date(2026, 10, 30)
NEW_YORK = ZoneInfo("America/New_York")
RETRIEVED_AT = "2026-11-06T07:20:00Z"
SNAPSHOT_ID = "2023-11-01_2026-10-30"
ORIGINAL_TYPES = ("cpi_yoy", "nonfarm_payrolls", "unemployment", "fed_rate")

SERIES = (
    {
        "event_type": "dgs2",
        "column": "DGS2",
        "close": time(16, 0),
        "unit": "percent",
        "noun": "2-year Treasury yield",
        "factors": ["interest_rates"],
        "url": "https://fred.stlouisfed.org/series/DGS2",
    },
    {
        "event_type": "dgs10",
        "column": "DGS10",
        "close": time(16, 0),
        "unit": "percent",
        "noun": "10-year Treasury yield",
        "factors": ["interest_rates"],
        "url": "https://fred.stlouisfed.org/series/DGS10",
    },
    {
        "event_type": "vixcls",
        "column": "VIXCLS",
        "close": time(16, 15),
        "unit": "index_points",
        "noun": "VIX close",
        "factors": [],
        "url": "https://fred.stlouisfed.org/series/VIXCLS",
    },
)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def as_utc(day: date, clock: time) -> str:
    stamp = datetime.combine(day, clock, tzinfo=NEW_YORK).astimezone(UTC)
    return stamp.strftime("%Y-%m-%dT%H:%M:%SZ")


def signed_change(current: str, prior: str) -> str:
    delta = Decimal(current) - Decimal(prior)
    text = format(delta, "f").rstrip("0").rstrip(".")
    if delta > 0:
        return f"+{text}"
    return text or "0"


def read_observations(path: Path, column: str) -> list[tuple[date, str]]:
    rows: list[tuple[date, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            raw = (row.get(column) or "").strip()
            if raw in {"", "."}:
                continue
            rows.append((date.fromisoformat(row["observation_date"]), raw))
    return rows


def build_records(spec: dict) -> list[dict]:
    observations = read_observations(MACRO / f"{spec['column']}.csv", spec["column"])
    prior: tuple[date, str, str] | None = None
    records = []
    for day, value in observations:
        body = f"{spec['column']},{day.isoformat()},{value}"
        digest = sha256_text(body)
        if WINDOW_START <= day <= WINDOW_END:
            available_at = as_utc(day, spec["close"])
            if prior is None:
                consensus = None
                excerpt = (
                    f"FRED {spec['column']} {spec['noun']} closed at {value}"
                    f"{'' if spec['unit'] != 'percent' else '%'}."
                )
            else:
                prior_day, prior_value, prior_digest = prior
                change = signed_change(value, prior_value)
                unit_suffix = "%" if spec["unit"] == "percent" else ""
                change_unit = " percentage points" if spec["unit"] == "percent" else ""
                consensus = {
                    "consensus": json.loads(prior_value),
                    "consensus_content_hash": prior_digest,
                    "consensus_published_at": as_utc(prior_day, spec["close"]),
                    "consensus_source_name": f"FRED {spec['column']} prior close",
                    "consensus_source_url": spec["url"],
                }
                excerpt = (
                    f"FRED {spec['column']} {spec['noun']} closed at {value}{unit_suffix} "
                    f"versus prior close {prior_value}{unit_suffix} ({change}{change_unit})."
                )
            record = {
                "accession_number": None,
                "actual": json.loads(value),
                "affected_risk_factors": list(spec["factors"]),
                "available_at": available_at,
                "chunks": [],
                "consensus": None,
                "consensus_content_hash": None,
                "consensus_published_at": None,
                "consensus_source_name": None,
                "consensus_source_url": None,
                "content_hash": digest,
                "content_path": None,
                "document_content_hash": None,
                "event_status": "released_before_cutoff",
                "event_time": available_at,
                "event_type": spec["event_type"],
                "event_types": [],
                "evidence_id": f"macro_fred_{spec['event_type']}_{day.isoformat()}",
                "excerpt": excerpt,
                "exhibit_type": None,
                "filing_form": None,
                "first_seen_at": available_at,
                "published_at": available_at,
                "retrieved_at": RETRIEVED_AT,
                "schema_version": 1,
                "scope": "macro",
                "snapshot_id": SNAPSHOT_ID,
                "source_name": "Federal Reserve Bank of St. Louis (FRED)",
                "source_type": (
                    "fred_daily_close_versus_prior_close" if consensus else "fred_daily_close"
                ),
                "source_url": spec["url"],
                "status": "released_before_cutoff",
                "symbol": None,
                "timestamp_confidence": "low",
                "title": f"{spec['event_type']} close {day.isoformat()}",
                "unit": spec["unit"],
            }
            if consensus:
                record.update(consensus)
            MacroEvidenceRecord.model_validate(record)
            records.append(record)
        prior = (day, value, digest)
    return records


def main() -> None:
    new_types = {spec["event_type"] for spec in SERIES}
    events_path = MACRO / "events.jsonl"
    kept = []
    for line in events_path.read_text().splitlines():
        if not line.strip():
            continue
        if json.loads(line)["event_type"] in new_types:
            continue
        kept.append(line)
    kept_counts = Counter(json.loads(line)["event_type"] for line in kept)
    if sum(kept_counts[name] for name in ORIGINAL_TYPES) != 171:
        raise RuntimeError(f"original macro events changed: {kept_counts}")

    fresh = [record for spec in SERIES for record in build_records(spec)]
    fresh.sort(key=lambda record: (record["available_at"], record["event_type"]))
    ids = [record["evidence_id"] for record in fresh]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate FRED evidence ids")
    lines = kept + [json.dumps(record, sort_keys=True) for record in fresh]
    events_path.write_text("\n".join(lines) + "\n")

    released = Counter(
        json.loads(line)["event_type"]
        for line in lines
        if json.loads(line)["status"] == "released_before_cutoff"
    )
    sessions = {json.loads(line)["event_time"][:10] for line in lines}
    audit_path = MACRO / "audit.json"
    audit = json.loads(audit_path.read_text())
    audit["config"]["event_types"] = [*ORIGINAL_TYPES, *[spec["event_type"] for spec in SERIES]]
    audit["distinct_release_sessions"] = len(sessions)
    audit["released_event_counts"] = {name: released[name] for name in sorted(released)}
    audit["secondary_assumed_pit_count"] = len(lines)
    audit["limitation"] = (
        "Official release dates are verified, but historical consensus availability "
        "is assumed and therefore secondary-analysis only. DGS2, DGS10, and VIXCLS "
        "are FRED daily closes: consensus is the prior close, not a survey, and the "
        "clock is the New York cash close rather than a recorded publication time."
    )
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    manifest_path = SNAPSHOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["file_sha256"]["macro/audit.json"] = file_sha256(audit_path)
    manifest["file_sha256"]["macro/events.jsonl"] = file_sha256(events_path)
    manifest["macro_backfill"]["record_count"] = len(lines)
    manifest["macro_backfill"]["distinct_release_sessions"] = len(sessions)
    manifest["macro_backfill"]["daily_close_record_count"] = len(fresh)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    summary = Counter(record["event_type"] for record in fresh)
    print(json.dumps({"added": summary, "total": len(lines), "sessions": len(sessions)}, indent=2))


if __name__ == "__main__":
    main()
