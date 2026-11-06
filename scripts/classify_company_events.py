"""Classify frozen company evidence into the canonical multi-label taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from financial_strategy.data.company_events import classify_company_event
from financial_strategy.schemas import COMPANY_EVENT_TYPES, EvidenceRecord

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def classify_snapshot(dataset: Path) -> dict:
    company = dataset / "company"
    evidence_path = company / "evidence.jsonl"
    manifest_path = dataset / "manifest.json"
    records = []
    for line in evidence_path.read_text().splitlines():
        if not line:
            continue
        payload = json.loads(line)
        payload.pop("event_type", None)
        records.append(EvidenceRecord.model_validate(payload))
    identifiers = [item.evidence_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("company evidence IDs are not unique")

    type_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    by_symbol: dict[str, Counter[str]] = defaultdict(Counter)
    updated: list[EvidenceRecord] = []
    multi_label_count = 0
    unclassified_count = 0

    for item in records:
        if item.content_path is None or item.document_content_hash is None:
            raise ValueError(f"company evidence lacks a document: {item.evidence_id}")
        document = company / item.content_path
        if not document.is_file():
            raise FileNotFoundError(document)
        if _sha256(document) != item.document_content_hash:
            raise ValueError(f"document hash mismatch: {item.evidence_id}")
        classification = classify_company_event(
            title=item.title,
            text=document.read_text(),
            filing_form=item.filing_form,
        )
        event_types = list(classification.event_types)
        if len(event_types) > 1:
            multi_label_count += 1
        if not event_types:
            unclassified_count += 1
            by_symbol[item.symbol]["unclassified"] += 1
        else:
            by_symbol[item.symbol]["classified"] += 1
        type_counts.update(event_types)
        rule_counts.update(classification.matched_rules)
        updated.append(
            EvidenceRecord.model_validate(
                {
                    **item.model_dump(),
                    "event_types": event_types,
                }
            )
        )

    audit = {
        "version": 1,
        "classifier": "deterministic_sec_company_events_v1",
        "taxonomy": list(COMPANY_EVENT_TYPES),
        "record_count": len(updated),
        "classified_record_count": len(updated) - unclassified_count,
        "unclassified_record_count": unclassified_count,
        "multi_label_record_count": multi_label_count,
        "event_types_counts": dict(sorted(type_counts.items())),
        "rule_match_counts": dict(sorted(rule_counts.items())),
        "symbol_coverage": {
            symbol: dict(sorted(counts.items()))
            for symbol, counts in sorted(by_symbol.items())
        },
        "network_sources_added": 0,
        "documents_modified": 0,
    }
    evidence_payload = "".join(
        item.model_dump_json() + "\n" for item in updated
    )
    audit_payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"

    manifest = json.loads(manifest_path.read_text())
    manifest["company_event_taxonomy"] = {
        "version": 1,
        "classifier": audit["classifier"],
        "taxonomy": audit["taxonomy"],
        "classified_record_count": audit["classified_record_count"],
        "unclassified_record_count": audit["unclassified_record_count"],
        "multi_label_record_count": audit["multi_label_record_count"],
    }

    temporary_files: list[tuple[Path, Path]] = []
    for target, payload in (
        (evidence_path, evidence_payload),
        (company / "event_classification_audit.json", audit_payload),
    ):
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
        with os.fdopen(descriptor, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_files.append((Path(temporary_name), target))

    evidence_temporary = temporary_files[0][0]
    audit_temporary = temporary_files[1][0]
    manifest["file_sha256"]["company/evidence.jsonl"] = _sha256(
        evidence_temporary
    )
    manifest["file_sha256"]["company/event_classification_audit.json"] = _sha256(
        audit_temporary
    )
    manifest_payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    descriptor, manifest_temporary_name = tempfile.mkstemp(
        prefix=f".{manifest_path.name}.", dir=manifest_path.parent
    )
    with os.fdopen(descriptor, "w") as handle:
        handle.write(manifest_payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary_files.append((Path(manifest_temporary_name), manifest_path))

    for temporary, target in temporary_files:
        os.replace(temporary, target)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/2023-11-01_2026-10-30",
    )
    arguments = parser.parse_args()
    print(json.dumps(classify_snapshot(arguments.dataset.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
