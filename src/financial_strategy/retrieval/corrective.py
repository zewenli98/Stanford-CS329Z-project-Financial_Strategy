"""CRAG-style company retrieval: question, BM25, then accept, one rewrite, or a gap.

Judgments are recorded and replayed. The web-search branch of corrective RAG is
not used. Ranking does not read the judgment.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, model_validator

from financial_strategy.config import RetrievalConfig
from financial_strategy.errors import StrategyError
from financial_strategy.retrieval.bm25 import search
from financial_strategy.schemas import EvidenceRecord
from financial_strategy.schemas.models import StrictModel

DEFAULT_QUESTION = (
    "What current filing could change the next regular-session range: earnings, guidance, "
    "a merger, buyback, financing, litigation, management change, cybersecurity, or product news?"
)
INITIAL_QUERY = (
    "earnings guidance merger buyback financing litigation management cybersecurity product 8-K"
)
REWRITE_QUERY = "current 8-K earnings guidance merger buyback management product litigation"
CURRENT_RELEASE_NOT_IN_SNAPSHOT = "CURRENT_RELEASE_NOT_IN_SNAPSHOT"
MATERIAL_EVENT_TYPES = frozenset(
    {
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
)
WRONG_COMPANY = "WRONG_COMPANY"
RETRIEVAL_BUDGET_EXHAUSTED = "RETRIEVAL_BUDGET_EXHAUSTED"


class CompanyQuestion(StrictModel):
    question: str = Field(min_length=1)


class RetrievalJudgment(StrictModel):
    company_match: bool
    event_match: bool
    sufficient: bool
    action: Literal["accept", "rewrite", "record_gap"]
    gap: str | None = None
    rewritten_query: str | None = None
    accepted_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def action_matches_fields(self) -> RetrievalJudgment:
        if self.action == "accept":
            if not (self.company_match and self.event_match and self.sufficient):
                raise ValueError("accept requires company, event, and sufficiency")
            if not self.accepted_evidence_ids:
                raise ValueError("accept requires accepted_evidence_ids")
            if self.gap is not None or self.rewritten_query is not None:
                raise ValueError("accept cannot carry a gap or rewritten query")
        elif self.action == "rewrite":
            if self.company_match and self.event_match and self.sufficient:
                raise ValueError("rewrite cannot be a sufficient match")
            if self.rewritten_query is None or not self.rewritten_query.strip():
                raise ValueError("rewrite requires rewritten_query")
        elif self.gap is None or not self.gap.strip():
            raise ValueError("record_gap requires gap")
        elif self.rewritten_query is not None:
            raise ValueError("record_gap cannot carry a rewritten query")
        return self


class CorrectiveResult(StrictModel):
    question: str
    accepted_evidence_ids: list[str]
    gap: str | None = None
    traces: list[dict]


def dedupe_evidence(records: list[EvidenceRecord]) -> list[EvidenceRecord]:
    """Collapse a filing and its reprints before they compete for the top-n slots."""

    grouped: dict[str, EvidenceRecord] = {}
    for record in records:
        key = record.accession_number or record.content_hash
        current = grouped.get(key)
        if current is None or _more_specific(record, current):
            grouped[key] = record
    return sorted(grouped.values(), key=lambda item: item.evidence_id)


def _more_specific(candidate: EvidenceRecord, incumbent: EvidenceRecord) -> bool:
    candidate_rank = (candidate.event_time is not None, bool(candidate.accession_number))
    incumbent_rank = (incumbent.event_time is not None, bool(incumbent.accession_number))
    if candidate_rank != incumbent_rank:
        return candidate_rank > incumbent_rank
    return candidate.evidence_id < incumbent.evidence_id


def coerce_judgment(
    raw: RetrievalJudgment,
    hit_ids: set[str],
    *,
    allow_rewrite: bool,
) -> RetrievalJudgment:
    """Keep one rewrite, and drop evidence ids the search did not return."""

    accepted = [evidence_id for evidence_id in raw.accepted_evidence_ids if evidence_id in hit_ids]
    if (
        raw.action == "accept"
        and raw.company_match
        and raw.event_match
        and raw.sufficient
        and accepted
    ):
        return RetrievalJudgment(
            company_match=True,
            event_match=True,
            sufficient=True,
            action="accept",
            accepted_evidence_ids=accepted,
        )
    if (
        allow_rewrite
        and raw.action == "rewrite"
        and raw.rewritten_query
        and raw.rewritten_query.strip()
        and not (raw.company_match and raw.event_match and raw.sufficient)
    ):
        return raw.model_copy(
            update={"rewritten_query": raw.rewritten_query.strip(), "accepted_evidence_ids": []}
        )
    return RetrievalJudgment(
        company_match=raw.company_match,
        event_match=False,
        sufficient=False,
        action="record_gap",
        gap=(raw.gap or CURRENT_RELEASE_NOT_IN_SNAPSHOT).strip(),
    )


def scripted_retrieval_judgment(payload: dict) -> dict:
    """Offline judge for the current-earnings question. Replay does not call a provider."""

    evidence = list(payload.get("evidence") or [])
    attempt = int(payload.get("attempt", 0))
    symbol = payload["symbol"]
    cutoff = _as_datetime(payload["decision_cutoff"])
    window = timedelta(days=int(payload.get("news_lookback_days", 7)))
    company_match = bool(evidence) and all(
        item.get("symbol") == symbol and item.get("scope", "company") == "company"
        for item in evidence
    )
    if evidence and not company_match:
        return _gap(company_match=False, gap=WRONG_COMPANY)
    matched = [
        item["evidence_id"]
        for item in evidence
        if _is_current(item, cutoff, window) and _is_material_event(item)
    ]
    event_match = bool(matched)
    if company_match and event_match:
        return {
            "company_match": True,
            "event_match": True,
            "sufficient": True,
            "action": "accept",
            "gap": None,
            "rewritten_query": None,
            "accepted_evidence_ids": matched,
        }
    if attempt == 0:
        return {
            "company_match": company_match,
            "event_match": False,
            "sufficient": False,
            "action": "rewrite",
            "gap": CURRENT_RELEASE_NOT_IN_SNAPSHOT,
            "rewritten_query": REWRITE_QUERY,
            "accepted_evidence_ids": [],
        }
    return _gap(company_match=company_match, gap=CURRENT_RELEASE_NOT_IN_SNAPSHOT)


def run_corrective_retrieval(
    *,
    evidence: list[EvidenceRecord],
    symbol: str,
    snapshot_id: str,
    retrieval: RetrievalConfig,
    question: str,
    reserve_search: Callable[[], None],
    judge: Callable[[str, str, list[EvidenceRecord], int], RetrievalJudgment],
) -> tuple[CorrectiveResult, list[EvidenceRecord]]:
    """Search, judge, rewrite at most once, then keep only the accepted passages."""

    company_records = [
        item for item in evidence if item.scope == "company" and item.symbol == symbol
    ]
    if not company_records:
        gap_judgment = _gap_judgment(CURRENT_RELEASE_NOT_IN_SNAPSHOT)
        return (
            CorrectiveResult(
                question=question,
                accepted_evidence_ids=[],
                gap=CURRENT_RELEASE_NOT_IN_SNAPSHOT,
                traces=[
                    _trace(
                        symbol=symbol,
                        snapshot_id=snapshot_id,
                        question=question,
                        attempt=0,
                        query=INITIAL_QUERY,
                        retrieval=retrieval,
                        candidate_count=0,
                        deduped_count=0,
                        hits=[],
                        judgment=gap_judgment,
                    )
                ],
            ),
            [],
        )
    corpus = dedupe_evidence(company_records)
    query = INITIAL_QUERY
    traces: list[dict] = []
    accepted: list[EvidenceRecord] = []
    gap: str | None = None
    for attempt in range(2):
        try:
            reserve_search()
        except StrategyError as exc:
            if "BUDGET" not in exc.code:
                raise
            gap = gap or RETRIEVAL_BUDGET_EXHAUSTED
            traces.append(
                _trace(
                    symbol=symbol,
                    snapshot_id=snapshot_id,
                    question=question,
                    attempt=attempt,
                    query=query,
                    retrieval=retrieval,
                    candidate_count=len(company_records),
                    deduped_count=len(corpus),
                    hits=[],
                    judgment=_gap_judgment(gap),
                    budget_error=exc.code,
                )
            )
            break
        hits = search(corpus, query, retrieval, scope="company", symbol=symbol)
        hit_records = [hit.evidence for hit in hits]
        judgment = coerce_judgment(
            judge(question, query, hit_records, attempt),
            {item.evidence_id for item in hit_records},
            allow_rewrite=attempt == 0,
        )
        traces.append(
            _trace(
                symbol=symbol,
                snapshot_id=snapshot_id,
                question=question,
                attempt=attempt,
                query=query,
                retrieval=retrieval,
                candidate_count=len(company_records),
                deduped_count=len(corpus),
                hits=hits,
                judgment=judgment,
            )
        )
        if judgment.action == "accept":
            by_id = {item.evidence_id: item for item in hit_records}
            accepted = [
                by_id[evidence_id]
                for evidence_id in judgment.accepted_evidence_ids
                if by_id[evidence_id].symbol == symbol
            ]
            gap = None
            if accepted:
                break
            gap = CURRENT_RELEASE_NOT_IN_SNAPSHOT
            accepted = []
            break
        if judgment.action == "rewrite":
            query = judgment.rewritten_query or REWRITE_QUERY
            gap = judgment.gap
            continue
        gap = judgment.gap or CURRENT_RELEASE_NOT_IN_SNAPSHOT
        accepted = []
        break
    result = CorrectiveResult(
        question=question,
        accepted_evidence_ids=[item.evidence_id for item in accepted],
        gap=gap,
        traces=traces,
    )
    return result, accepted


def _gap(*, company_match: bool, gap: str) -> dict:
    return {
        "company_match": company_match,
        "event_match": False,
        "sufficient": False,
        "action": "record_gap",
        "gap": gap,
        "rewritten_query": None,
        "accepted_evidence_ids": [],
    }


def _gap_judgment(gap: str) -> RetrievalJudgment:
    return RetrievalJudgment(
        company_match=False,
        event_match=False,
        sufficient=False,
        action="record_gap",
        gap=gap,
    )


def _trace(
    *,
    symbol: str,
    snapshot_id: str,
    question: str,
    attempt: int,
    query: str,
    retrieval: RetrievalConfig,
    candidate_count: int,
    deduped_count: int,
    hits: list,
    judgment: RetrievalJudgment,
    budget_error: str | None = None,
) -> dict:
    trace = {
        "scope": "company",
        "protocol": "corrective_retrieval",
        "symbol": symbol,
        "snapshot_id": snapshot_id,
        "question": question,
        "attempt": attempt,
        "query": query,
        "algorithm": retrieval.algorithm,
        "tokenizer_version": retrieval.tokenizer_version,
        "k1": retrieval.k1,
        "b": retrieval.b,
        "top_n": retrieval.top_n,
        "candidate_count": candidate_count,
        "deduped_count": deduped_count,
        "returned_evidence_ids": [hit.evidence.evidence_id for hit in hits],
        "scores": [hit.score for hit in hits],
        "judgment": judgment.model_dump(mode="json"),
    }
    if budget_error is not None:
        trace["budget_error"] = budget_error
    return trace


def _is_material_event(item: dict) -> bool:
    types = {item.get("event_type")} | set(item.get("event_types") or [])
    text = f"{item.get('title', '')}\n{item.get('excerpt', '')}".lower()
    keywords = (
        "earnings",
        "guidance",
        "merger",
        "acquisition",
        "buyback",
        "repurchase",
        "dividend",
        "financing",
        "litigation",
        "regulatory",
        "management",
        "cyber",
        "product",
        "8-k",
    )
    return bool(types & MATERIAL_EVENT_TYPES) or any(word in text for word in keywords)


def _is_current(item: dict, cutoff: datetime, window: timedelta) -> bool:
    published = item.get("published_at")
    if published is None:
        return False
    age = cutoff - _as_datetime(published)
    return timedelta(0) <= age <= window


def _as_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retrieval timestamps must include a timezone")
    return parsed
