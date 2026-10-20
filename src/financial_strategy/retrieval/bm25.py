"""Point-in-time BM25 retrieval with stable ordering."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

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


def search(
    evidence: list[EvidenceRecord],
    query: str,
    config: RetrievalConfig,
    *,
    scope: str,
    symbol: str | None = None,
) -> list[RetrievalHit]:
    """Filter before IDF calculation so future or cross-entity documents cannot leak."""

    eligible = [
        item
        for item in evidence
        if item.scope == scope and (scope != "company" or item.symbol == symbol)
    ]
    eligible.sort(key=lambda item: item.evidence_id)
    if not eligible:
        return []
    corpus = [tokenize(f"{item.title}\n{item.excerpt}") for item in eligible]
    engine = BM25Okapi(corpus, k1=config.k1, b=config.b)
    scores = engine.get_scores(tokenize(query))
    hits = [
        RetrievalHit(evidence=item, score=float(score))
        for item, score in zip(eligible, scores, strict=True)
    ]
    hits.sort(key=lambda item: (-item.score, item.evidence.evidence_id))
    return hits[: config.top_n]
