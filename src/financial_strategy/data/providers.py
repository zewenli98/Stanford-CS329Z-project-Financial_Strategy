"""Provider protocols and local file implementations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

import polars as pl

from financial_strategy.schemas import (
    CorporateActionRecord,
    EvidenceChunk,
    EvidenceRecord,
    MacroEvidenceRecord,
)


class MarketDataProvider(Protocol):
    def bars(self) -> pl.DataFrame: ...

    def corporate_actions(self) -> list[CorporateActionRecord]: ...


class EvidenceProvider(Protocol):
    def company_evidence(self) -> list[EvidenceRecord]: ...

    def macro_evidence(self) -> list[MacroEvidenceRecord]: ...


class UniverseProvider(Protocol):
    def universe(self) -> dict: ...

    def sector_exposures(self) -> dict: ...


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LocalMarketDataProvider:
    def __init__(self, bars_path: Path, actions_path: Path) -> None:
        self.bars_path = bars_path
        self.actions_path = actions_path

    def bars(self) -> pl.DataFrame:
        return pl.read_parquet(self.bars_path)

    def corporate_actions(self) -> list[CorporateActionRecord]:
        return [
            CorporateActionRecord.model_validate(item) for item in _read_jsonl(self.actions_path)
        ]


class LocalEvidenceProvider:
    def __init__(self, company_path: Path, macro_path: Path) -> None:
        self.company_path = company_path
        self.macro_path = macro_path

    def company_evidence(self) -> list[EvidenceRecord]:
        records = [
            EvidenceRecord.model_validate(item) for item in _read_jsonl(self.company_path)
        ]
        return _load_documents(records, self.company_path.parent)

    def company_evidence_by_ids(
        self,
        evidence_ids: set[str],
    ) -> list[EvidenceRecord]:
        records = [
            EvidenceRecord.model_validate(item)
            for item in _read_jsonl(self.company_path)
            if item["evidence_id"] in evidence_ids
        ]
        return _load_documents(records, self.company_path.parent)

    def macro_evidence(self) -> list[MacroEvidenceRecord]:
        records = [
            MacroEvidenceRecord.model_validate(item) for item in _read_jsonl(self.macro_path)
        ]
        return _load_documents(records, self.macro_path.parent)


def _load_documents(
    records: list[EvidenceRecord],
    evidence_directory: Path,
) -> list[EvidenceRecord]:
    """Resolve only fixture-local documents and verify declared content hashes."""

    loaded = []
    root = evidence_directory.resolve()
    for record in records:
        if record.content_path is None:
            loaded.append(record)
            continue
        path = (evidence_directory / record.content_path).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"evidence content_path escapes fixture root: {record.content_path}")
        text = path.read_text()
        digest = hashlib.sha256(text.encode()).hexdigest()
        if (
            record.document_content_hash is not None
            and digest != record.document_content_hash
        ):
            raise ValueError(f"document hash mismatch: {record.evidence_id}")
        chunks = record.chunks or _chunk_text(record.evidence_id, text)
        for chunk in chunks:
            content = text[chunk.start_char : chunk.end_char]
            if hashlib.sha256(content.encode()).hexdigest() != chunk.content_hash:
                raise ValueError(
                    f"evidence chunk hash mismatch: {record.evidence_id}/{chunk.chunk_id}"
                )
        loaded.append(
            record.model_copy(update={"document_text": text, "chunks": chunks})
        )
    return loaded


def _chunk_text(
    evidence_id: str,
    text: str,
    *,
    maximum_chars: int = 4_000,
    overlap_chars: int = 400,
) -> list[EvidenceChunk]:
    """Create deterministic character chunks without splitting persisted content."""

    if not text:
        return []
    chunks = []
    start = 0
    index = 0
    while start < len(text):
        end = min(len(text), start + maximum_chars)
        content = text[start:end]
        chunks.append(
            EvidenceChunk(
                chunk_id=f"{evidence_id}_chunk_{index:04}",
                start_char=start,
                end_char=end,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
            )
        )
        if end == len(text):
            break
        start = end - overlap_chars
        index += 1
    return chunks


class LocalUniverseProvider:
    def __init__(self, universe_path: Path, exposures_path: Path) -> None:
        self.universe_path = universe_path
        self.exposures_path = exposures_path

    def universe(self) -> dict:
        return json.loads(self.universe_path.read_text())

    def sector_exposures(self) -> dict:
        return json.loads(self.exposures_path.read_text())
