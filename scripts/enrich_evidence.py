"""Create or replace a document-aware snapshot from the expanded market snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl

from financial_strategy.data.company_events import classify_company_event
from financial_strategy.schemas import EvidenceRecord, MacroEvidenceRecord
from scripts.build_expanded_data import (
    DEFAULT_SEC_USER_AGENT,
    _content_hash,
    _document_chunks,
    _normalize_document,
)

ROOT = Path(__file__).resolve().parents[1]


def enrich_snapshot(
    *,
    source: Path,
    output: Path,
    candidate_cards: Path,
    macro_source: Path | None,
    maximum_documents: int | None,
    resume_documents: Path | None = None,
    all_documents: bool = False,
) -> Path:
    if not source.is_dir():
        raise FileNotFoundError(f"source snapshot does not exist: {source}")
    all_evidence = [
        EvidenceRecord.model_validate_json(line)
        for line in (source / "company/evidence.jsonl").read_text().splitlines()
        if line
    ]
    if all_documents:
        selected = [
            item
            for item in all_evidence
            if item.content_path is None or item.document_content_hash is None
        ]
        candidate_source = None
    else:
        candidate_ids = {
            evidence_id
            for card in json.loads(candidate_cards.read_text())
            for evidence_id in card["evidence_ids"]
        }
        selected = [
            item for item in all_evidence if item.evidence_id in candidate_ids
        ]
        candidate_source = str(candidate_cards)
    selected.sort(key=lambda item: (item.available_at, item.evidence_id))
    if maximum_documents is not None:
        selected = selected[:maximum_documents]
    selected_ids = {item.evidence_id for item in selected}
    if resume_documents is not None:
        resume_documents.mkdir(parents=True, exist_ok=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        shutil.rmtree(temporary)
        # Build the complete replacement before touching the published snapshot.
        # This also makes source == output safe: the source remains available
        # until copytree has created the temporary working copy.
        shutil.copytree(source, temporary)
        (temporary / "company/documents").mkdir(parents=True, exist_ok=True)
        retrieved_at = datetime.now(UTC)
        headers = {
            "User-Agent": os.environ.get("SEC_USER_AGENT", DEFAULT_SEC_USER_AGENT),
            "Accept-Encoding": "gzip, deflate",
        }
        replacements: dict[str, EvidenceRecord] = {}
        exhibits: list[EvidenceRecord] = []
        with httpx.Client(headers=headers, timeout=60, follow_redirects=True) as client:
            for position, item in enumerate(selected, start=1):
                resume_path = (
                    resume_documents / f"{item.evidence_id}.txt"
                    if resume_documents is not None
                    else None
                )
                if resume_path is not None and resume_path.exists():
                    text = resume_path.read_text()
                else:
                    primary = _get_with_retries(client, str(item.source_url))
                    primary.raise_for_status()
                    text = _normalize_document(primary.text)
                    if resume_path is not None:
                        resume_path.write_text(text)
                replacements[item.evidence_id] = _document_record(
                    item,
                    text,
                    f"documents/{item.evidence_id}.txt",
                    retrieved_at,
                )
                (temporary / "company/documents" / f"{item.evidence_id}.txt").write_text(
                    text
                )
                if item.filing_form == "8-K" or "8-K" in item.title:
                    exhibits.extend(
                        _fetch_exhibits(
                            client,
                            item,
                            temporary / "company/documents",
                            retrieved_at,
                        )
                    )
                if position % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "downloaded_documents": position,
                                "requested_documents": len(selected),
                            }
                        ),
                        flush=True,
                    )
                time.sleep(0.13)

        merged = [
            replacements.get(item.evidence_id, item).model_copy(
                update={"snapshot_id": output.name}
            )
            for item in all_evidence
        ]
        merged.extend(
            item.model_copy(update={"snapshot_id": output.name})
            for item in exhibits
        )
        merged.sort(key=lambda item: (item.available_at, item.evidence_id))
        if all_documents and maximum_documents is None:
            _validate_document_complete_evidence(temporary, merged)
        _write_jsonl(
            temporary / "company/evidence.jsonl",
            [item.model_dump(mode="json") for item in merged],
        )
        macro_count = _import_macro_fixture(
            temporary,
            macro_source,
            snapshot_id=output.name,
        )
        bars_path = temporary / "market/daily_bars.parquet"
        pl.read_parquet(bars_path).with_columns(
            pl.lit(output.name).alias("snapshot_id")
        ).write_parquet(bars_path)
        universe_path = temporary / "universe/symbols.json"
        universe = json.loads(universe_path.read_text())
        universe["snapshot_id"] = output.name
        universe_path.write_text(json.dumps(universe, indent=2, sort_keys=True) + "\n")
        split_path = temporary / "splits.json"
        split = json.loads(split_path.read_text())
        split["snapshot_id"] = output.name
        split_without_hash = {
            key: value for key, value in split.items() if key != "content_hash"
        }
        split["content_hash"] = _content_hash(split_without_hash)
        split_path.write_text(json.dumps(split, indent=2, sort_keys=True) + "\n")
        manifest_path = temporary / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.update(
            {
                "version": 2,
                "snapshot_id": output.name,
                "created_at": retrieved_at.isoformat(),
                "document_enrichment": {
                    "candidate_evidence_requested": len(selected_ids),
                    "document_evidence_written": len(replacements) + len(exhibits),
                    "macro_records_imported": macro_count,
                    "candidate_source": candidate_source,
                    "all_company_evidence_document_complete": (
                        all_documents and maximum_documents is None
                    ),
                },
            }
        )
        limitations = [
            item
            for item in manifest.get("source_limitations", [])
            if item
            != "Full SEC documents are limited to evidence reached by the frozen Top-50 screen."
        ]
        macro_limitation = (
            "Official macro records without verifiable historical availability are excluded."
        )
        if macro_limitation not in limitations:
            limitations.append(macro_limitation)
        manifest["source_limitations"] = limitations
        manifest["file_sha256"] = {
            str(path.relative_to(temporary)): _file_hash(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file() and path != manifest_path
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        # Publishing is the only point at which an existing output is removed.
        # A retrieval or validation failure therefore leaves the prior snapshot
        # intact instead of deleting it at the beginning of the run.
        if output.exists():
            if output.is_dir():
                shutil.rmtree(output)
            else:
                output.unlink()
        os.replace(temporary, output)
        return output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _document_record(
    item: EvidenceRecord,
    text: str,
    content_path: str,
    retrieved_at: datetime,
    *,
    evidence_id: str | None = None,
    exhibit_type: str | None = None,
    source_url: str | None = None,
) -> EvidenceRecord:
    identifier = evidence_id or item.evidence_id
    digest = hashlib.sha256(text.encode()).hexdigest()
    filing_form = item.filing_form or _form_from_title(item.title)
    classification = classify_company_event(
        title=item.title,
        text=text,
        filing_form=filing_form,
    )
    # ``model_copy(update=...)`` deliberately skips Pydantic validation.  Build
    # a fresh model so dynamically discovered exhibit URLs remain typed HttpUrl
    # values and serialization never silently emits an unvalidated string.
    return EvidenceRecord.model_validate(
        {
            **item.model_dump(),
            "evidence_id": identifier,
            "source_url": source_url or item.source_url,
            "retrieved_at": retrieved_at,
            "content_hash": digest,
            "content_path": content_path,
            "schema_version": 1,
            "document_content_hash": digest,
            "accession_number": item.accession_number or _accession_from_id(item.evidence_id),
            "filing_form": filing_form,
            "exhibit_type": exhibit_type,
            "event_types": list(classification.event_types),
            "chunks": _document_chunks(identifier, text),
        }
    )


def _fetch_exhibits(
    client: httpx.Client,
    item: EvidenceRecord,
    document_directory: Path,
    retrieved_at: datetime,
) -> list[EvidenceRecord]:
    primary_url = str(item.source_url)
    directory = primary_url.rsplit("/", 1)[0]
    try:
        response = _get_with_retries(client, f"{directory}/index.json")
        response.raise_for_status()
        names = [
            entry.get("name", "")
            for entry in response.json().get("directory", {}).get("item", [])
        ]
    except (httpx.HTTPError, ValueError):
        return []
    result = []
    for index, name in enumerate(
        name
        for name in names
        if "99" in name.lower()
        and name.lower().endswith((".htm", ".html", ".txt"))
    ):
        response = _get_with_retries(client, f"{directory}/{name}")
        response.raise_for_status()
        text = _normalize_document(response.text)
        identifier = f"{item.evidence_id}_ex99_{index:02}"
        filename = f"{identifier}.txt"
        document_directory.joinpath(filename).write_text(text)
        result.append(
            _document_record(
                item,
                text,
                f"documents/{filename}",
                retrieved_at,
                evidence_id=identifier,
                exhibit_type="EX-99",
                source_url=f"{directory}/{name}",
            )
        )
    return result


def _import_macro_fixture(
    temporary: Path,
    source: Path | None,
    *,
    snapshot_id: str,
) -> int:
    if source is None:
        existing = temporary / "macro/events.jsonl"
        return (
            sum(bool(line) for line in existing.read_text().splitlines())
            if existing.exists()
            else 0
        )
    if not source.exists():
        _write_jsonl(temporary / "macro/events.jsonl", [])
        return 0
    records = [
        MacroEvidenceRecord.model_validate_json(line)
        for line in (source / "macro/events.jsonl").read_text().splitlines()
        if line
    ]
    records = [
        item
        for item in records
        if item.source_type in {"official_release", "government_release"}
    ]
    output = []
    for item in records:
        if item.content_path is not None:
            source_document = source / "macro" / item.content_path
            target_document = temporary / "macro" / item.content_path
            target_document.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_document, target_document)
        output.append(item.model_copy(update={"snapshot_id": snapshot_id}))
    _write_jsonl(
        temporary / "macro/events.jsonl",
        [item.model_dump(mode="json") for item in output],
    )
    return len(output)


def _get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    attempts: int = 3,
) -> httpx.Response:
    for attempt in range(attempts):
        try:
            response = client.get(url)
            response.raise_for_status()
            time.sleep(0.13)
            return response
        except (httpx.TimeoutException, httpx.NetworkError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2**attempt)
    raise AssertionError("retry loop must return or raise")


def _accession_from_id(identifier: str) -> str | None:
    compact = identifier.rsplit("_", 1)[-1]
    return compact if compact.isdigit() else None


def _form_from_title(title: str) -> str | None:
    return next((form for form in ("8-K", "10-Q", "10-K") if form in title), None)


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(value, sort_keys=True, default=str) + "\n" for value in values)
    )


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_document_complete_evidence(
    snapshot: Path,
    records: list[EvidenceRecord],
) -> None:
    identifiers = [item.evidence_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        duplicates = sorted(
            identifier
            for identifier in set(identifiers)
            if identifiers.count(identifier) > 1
        )
        raise ValueError(f"duplicate company evidence ids: {duplicates[:10]}")
    incomplete = [
        item.evidence_id
        for item in records
        if item.content_path is None or item.document_content_hash is None
    ]
    if incomplete:
        raise ValueError(f"company evidence remains metadata-only: {incomplete[:10]}")
    for item in records:
        if item.content_path is None or item.document_content_hash is None:
            raise ValueError(f"document metadata missing for {item.evidence_id}")
        document = snapshot / "company" / item.content_path
        if not document.is_file():
            raise FileNotFoundError(
                f"company evidence document missing for {item.evidence_id}: {document}"
            )
        if _file_hash(document) != item.document_content_hash:
            raise ValueError(f"company evidence hash mismatch: {item.evidence_id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "data/2023-11-01_2026-10-30",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/2023-11-01_2026-10-30",
    )
    parser.add_argument(
        "--candidate-cards",
        type=Path,
        default=ROOT / "runs/expanded_agent/published_research_cards.json",
    )
    parser.add_argument(
        "--macro-source",
        type=Path,
        default=ROOT / "data/2023-11-01_2026-10-30",
    )
    parser.add_argument(
        "--preserve-macro",
        action="store_true",
        help="Keep the source snapshot's macro records unchanged.",
    )
    parser.add_argument("--maximum-documents", type=int)
    parser.add_argument("--resume-documents", type=Path)
    parser.add_argument(
        "--all-documents",
        action="store_true",
        help="Download every metadata-only SEC filing and require complete documents.",
    )
    arguments = parser.parse_args()
    output = enrich_snapshot(
        source=arguments.source,
        output=arguments.output,
        candidate_cards=arguments.candidate_cards,
        macro_source=None if arguments.preserve_macro else arguments.macro_source,
        maximum_documents=arguments.maximum_documents,
        resume_documents=arguments.resume_documents,
        all_documents=arguments.all_documents,
    )
    print(json.dumps({"snapshot": str(output)}))


if __name__ == "__main__":
    main()
