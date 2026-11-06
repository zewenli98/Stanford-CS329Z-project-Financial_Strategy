"""Point-in-time BM25 retrieval with stable ordering."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date

from rank_bm25 import BM25Okapi

from financial_strategy.config import RetrievalConfig
from financial_strategy.schemas import EvidenceRecord

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return TOKEN_RE.findall(normalized)


@dataclass(frozen=True)
class RetrievalHit:
    evidence: EvidenceRecord
    score: float
    chunk_id: str | None = None
    text: str | None = None


def search(
    evidence: list[EvidenceRecord],
    query: str,
    config: RetrievalConfig,
    *,
    scope: str,
    symbol: str | None = None,
    target_session: date | None = None,
) -> list[RetrievalHit]:
    """Filter before IDF calculation so future or cross-entity documents cannot leak."""

    eligible = [
        item
        for item in evidence
        if item.scope == scope and (scope != "company" or item.symbol == symbol)
    ]
    eligible.sort(
        key=lambda item: (
            -_source_priority(item.source_type),
            item.evidence_id,
        )
    )
    deduplicated: dict[tuple, EvidenceRecord] = {}
    for item in eligible:
        event_key = (
            ("content", item.document_content_hash)
            if item.document_content_hash is not None
            else
            ("event", item.symbol, tuple(item.event_types), item.event_time)
            if item.event_types and item.event_time is not None
            else ("content", item.content_hash)
        )
        deduplicated.setdefault(event_key, item)
    eligible = list(deduplicated.values())
    if not eligible:
        return []
    rows: list[tuple[EvidenceRecord, str | None, str]] = []
    for item in eligible:
        if item.document_text is not None and item.chunks:
            rows.extend(
                (
                    item,
                    chunk.chunk_id,
                    item.document_text[chunk.start_char : chunk.end_char],
                )
                for chunk in item.chunks
            )
        else:
            rows.append((item, None, item.excerpt))
    corpus = [tokenize(f"{item.title}\n{text}") for item, _chunk, text in rows]
    engine = BM25Okapi(corpus, k1=config.k1, b=config.b)
    scores = engine.get_scores(tokenize(query))
    hits = [
        RetrievalHit(
            evidence=item,
            score=float(score)
            + 0.10 * _source_priority(item.source_type)
            + _recency_bonus(item, target_session),
            chunk_id=chunk_id,
            text=text,
        )
        for (item, chunk_id, text), score in zip(rows, scores, strict=True)
    ]
    hits.sort(
        key=lambda item: (
            -item.score,
            item.evidence.evidence_id,
            item.chunk_id or "",
        )
    )
    return hits[: config.top_n]


def _source_priority(source_type: str) -> int:
    normalized = source_type.lower()
    if normalized in {"sec_filing", "company_release", "regulatory_filing"}:
        return 2
    if normalized in {"exchange_notice", "government_release"}:
        return 1
    return 0


def _recency_bonus(item: EvidenceRecord, target_session: date | None) -> float:
    if target_session is None:
        return 0.0
    reference = item.event_time.date() if item.event_time else item.available_at.date()
    distance = abs((target_session - reference).days)
    return 0.25 / (1.0 + distance)
