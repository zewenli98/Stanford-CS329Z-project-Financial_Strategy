"""Structured agent roles. Retrieval remains deterministic and point-in-time bounded."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock

from pydantic import Field

from financial_strategy.config import MVPConfig
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.errors import StrategyError
from financial_strategy.llm import InvocationResult, LLMClient
from financial_strategy.retrieval.bm25 import search
from financial_strategy.retrieval.corrective import (
    CURRENT_RELEASE_NOT_IN_SNAPSHOT,
    DEFAULT_QUESTION,
    CompanyQuestion,
    RetrievalJudgment,
    run_corrective_retrieval,
    scripted_retrieval_judgment,
)
from financial_strategy.schemas import (
    CatalystAssessment,
    Claim,
    ClaimType,
    EventRecord,
    EvidenceChunk,
    EvidenceRecord,
    ExtractedMetric,
    FeatureRecord,
    MacroAssessment,
    MacroContext,
    MacroEvidenceRecord,
    ResearchCard,
    ScreeningTask,
    VerificationReport,
)
from financial_strategy.schemas.models import StrictModel


class MacroAgentOutput(StrictModel):
    evidence_ids: list[str]
    assessments: list[MacroAssessment] = Field(default_factory=list)


class CompanyAgentOutput(StrictModel):
    events: list[EventRecord]
    claims: list[Claim]
    metrics: list[ExtractedMetric] = Field(default_factory=list)


class SynthesisOutput(StrictModel):
    catalyst_assessment: CatalystAssessment = Field(
        default_factory=CatalystAssessment
    )
    rationale: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


def _hit_payload(hit: object) -> dict:
    payload = hit.evidence.model_dump(mode="json")
    payload["retrieved_chunk"] = {
        "chunk_id": hit.chunk_id,
        "text": hit.text,
    }
    return payload


def _accepted_payload(item: EvidenceRecord) -> dict:
    """Expose one accepted document using the same chunk contract as BM25 hits."""

    payload = item.model_dump(mode="json")
    chunk = item.chunks[0] if item.chunks and item.document_text else None
    payload["retrieved_chunk"] = {
        "chunk_id": None if chunk is None else chunk.chunk_id,
        "text": (
            item.excerpt
            if chunk is None or item.document_text is None
            else item.document_text[chunk.start_char : chunk.end_char]
        ),
    }
    return payload


def scripted_responder(role: str, payload: dict) -> dict:
    """Deterministic offline behavior used by CI."""

    if role == "macro_evidence":
        assessments = []
        cutoff = datetime.fromisoformat(str(payload["decision_cutoff"]))
        for item in payload["evidence"]:
            actual = item.get("actual")
            consensus = item.get("consensus")
            difference = (
                actual - consensus
                if actual is not None and consensus is not None
                else None
            )
            assessments.append(
                {
                    "event_type": item["event_type"],
                    "evidence_ids": [item["evidence_id"]],
                    "target_session_relevance": "high",
                    "surprise_direction": (
                        "unknown"
                        if difference is None
                        else "positive"
                        if difference > 0
                        else "negative"
                        if difference < 0
                        else "neutral"
                    ),
                    "standardized_surprise": None,
                    "affected_risk_factors": item.get(
                        "affected_risk_factors", []
                    ),
                    "target_session_timing": (
                        "before_open"
                        if datetime.fromisoformat(str(item["event_time"])) <= cutoff
                        else "during_session"
                    ),
                    "confidence": item["timestamp_confidence"],
                }
            )
        return {
            "evidence_ids": [item["evidence_id"] for item in payload["evidence"]],
            "assessments": assessments,
        }
    if role == "company_evidence":
        if payload.get("op") == "propose_question":
            return {"question": DEFAULT_QUESTION}
        if payload.get("op") == "judge_retrieval":
            return scripted_retrieval_judgment(payload)
        evidence = payload["evidence"]
        events = []
        claims = []
        for index, item in enumerate(evidence):
            event_types = item.get("event_types") or []
            status = item.get("event_status") or "timing_unknown"
            if event_types:
                events.append(
                    {
                        "event_id": f"event_{item['evidence_id']}",
                        "event_types": event_types,
                        "status": status,
                        "scheduled_at": (
                            item.get("event_time")
                            if status.startswith("scheduled_")
                            else None
                        ),
                        "released_at": (
                            item["published_at"]
                            if status == "released_before_cutoff"
                            else None
                        ),
                        "confidence": item["timestamp_confidence"],
                        "evidence_ids": [item["evidence_id"]],
                        "sessions_ago": 1,
                    }
                )
            claims.append(
                {
                    "claim_id": f"claim_company_{index}",
                    "text": item["excerpt"],
                    "claim_type": "observed_fact",
                    "evidence_ids": [item["evidence_id"]],
                }
            )
        return {"events": events, "claims": claims, "metrics": []}
    if role == "synthesis":
        claims = list(payload["company_claims"])
        evidence_ids = sorted(
            {
                evidence_id
                for claim in claims
                for evidence_id in claim.get("evidence_ids", [])
            }
        )
        return {
            "catalyst_assessment": {
                "target_session_relevance": "low" if claims else "none",
                # Scripted mode is a deterministic contract test, not an alpha
                # estimate.  Emit the most conservative encodable assessment so
                # CI exercises the same feature/training path as live mode.
                "novelty": "low" if claims else "unknown",
                "materiality": "low" if claims else "unknown",
                "expectedness": "expected" if claims else "unknown",
                "expected_abs_impact_bucket": (
                    "under_1pct" if claims else "unknown"
                ),
                "evidence_ids": evidence_ids,
                "counterevidence_ids": [],
                "signed_residual_bucket": "neutral",
                "target_session_timing": "before_open",
            },
            "rationale": [
                "Deterministic market features and verified events may support a wider "
                "regular-session range, (high - low) / low."
            ],
            "counterarguments": [
                "Historical volatility and event presence do not guarantee a large move."
            ],
            "uncertainty": ["The score is a heuristic rank, not a calibrated probability."],
        }
    if role == "verifier":
        return {"verdict": "PASS", "issues": [], "attempt": payload["attempt"]}
    raise KeyError(f"Unknown scripted role: {role}")


@dataclass
class AgentSuite:
    config: MVPConfig
    client: LLMClient
    _budget_lock: Lock = field(default_factory=Lock)
    _total_tool_calls: int = 0
    _macro_tool_calls: int = 0
    _company_tool_calls: dict[str, int] = field(default_factory=dict)
    _company_retrieval: dict[str, tuple[CompanyAgentOutput, list[dict]]] = field(
        default_factory=dict
    )

    def _reserve_tool_call(self, role: str, symbol: str | None = None) -> None:
        with self._budget_lock:
            if self._total_tool_calls >= self.config.agents.max_total_tool_calls:
                raise StrategyError(
                    "TASK_TOOL_BUDGET_EXHAUSTED",
                    "task retrieval budget exhausted",
                )
            if role == "macro":
                if self._macro_tool_calls >= self.config.agents.macro_max_tool_calls:
                    raise StrategyError(
                        "MACRO_BUDGET_EXHAUSTED",
                        "macro retrieval budget exhausted",
                    )
                self._macro_tool_calls += 1
            elif symbol is not None:
                used = self._company_tool_calls.get(symbol, 0)
                if used >= self.config.agents.company_max_tool_calls_per_symbol:
                    raise StrategyError(
                        "SYMBOL_BUDGET_EXHAUSTED",
                        f"{symbol} retrieval budget exhausted",
                    )
                self._company_tool_calls[symbol] = used + 1
            self._total_tool_calls += 1

    def macro(
        self,
        task: ScreeningTask,
        evidence: list[MacroEvidenceRecord],
        snapshot_id: str,
    ) -> tuple[MacroContext, InvocationResult, list[dict]]:
        self._reserve_tool_call("macro")
        calendar = TradingCalendar(self.config.calendar["name"])
        previous_close = calendar.session_close(
            calendar.previous_session(task.target_session)
        )
        relevant_evidence = [
            item
            for item in evidence
            if item.event_time is not None
            and (
                item.event_time.date() == task.target_session
                or item.available_at >= previous_close
            )
        ]
        hits = search(
            relevant_evidence,
            "inflation interest rates oil prices dollar economic release",
            self.config.retrieval,
            scope="macro",
            target_session=task.target_session,
        )
        payload = {
            "task_id": task.task_id,
            "decision_cutoff": task.decision_cutoff,
            "evidence": [_hit_payload(hit) for hit in hits],
        }
        invocation = self.client.generate("macro_evidence", payload, MacroAgentOutput)
        output = MacroAgentOutput.model_validate(invocation.response)
        retrieved_by_id = {hit.evidence.evidence_id: hit.evidence for hit in hits}
        selected = [
            retrieved_by_id[evidence_id]
            for evidence_id in output.evidence_ids
            if evidence_id in retrieved_by_id
        ]
        assessments = normalize_macro_assessments(
            output.assessments,
            selected,
            task,
            self.config.ranking.macro.surprise_scales,
        )
        context = MacroContext(
            task_id=task.task_id,
            events=selected,
            evidence_ids=[item.evidence_id for item in selected],
            assessments=assessments,
            target_session=task.target_session,
            decision_cutoff=task.decision_cutoff,
        )
        trace = [
            {
                "scope": "macro",
                "snapshot_id": snapshot_id,
                "query": "inflation interest rates oil prices dollar economic release",
                "algorithm": self.config.retrieval.algorithm,
                "tokenizer_version": self.config.retrieval.tokenizer_version,
                "k1": self.config.retrieval.k1,
                "b": self.config.retrieval.b,
                "top_n": self.config.retrieval.top_n,
                "candidate_count": len(evidence),
                "returned_evidence_ids": [
                    hit.evidence.evidence_id for hit in hits
                ],
                "returned_chunk_ids": [hit.chunk_id for hit in hits],
                "scores": [hit.score for hit in hits],
            }
        ]
        return context, invocation, trace

    def macro_semantic_verify(
        self,
        context: MacroContext,
    ) -> tuple[VerificationReport, InvocationResult]:
        """Independently verify the meaning of the task-level macro assessment."""

        payload = {
            "symbol": None,
            "semantic_scope": {
                "target_session": context.target_session,
                "decision_cutoff": context.decision_cutoff,
                "macro_assessments": [
                    item.model_dump(mode="json") for item in context.assessments
                ],
            },
            "trusted_computed_context": {
                "standardized_surprises_are_code_owned": True,
            },
            "evidence": [
                item.model_dump(mode="json") for item in context.events
            ],
            "attempt": 0,
        }
        invocation = self.client.generate("verifier", payload, VerificationReport)
        return normalize_semantic_verification(
            VerificationReport.model_validate(invocation.response)
        ), invocation

    def company(
        self,
        task: ScreeningTask,
        symbol: str,
        evidence: list,
        snapshot_id: str,
        revision_feedback: list[dict] | None = None,
    ) -> tuple[CompanyAgentOutput, list[InvocationResult], list[dict]]:
        cached = self._company_retrieval.get(symbol)
        if cached is not None and not revision_feedback:
            output, _traces = cached
            return output.model_copy(deep=True), [], []
        if not any(item.symbol == symbol for item in evidence):
            output = CompanyAgentOutput(
                events=[],
                claims=[
                    Claim(
                        claim_id="claim_retrieval_gap",
                        text=(
                            f"{CURRENT_RELEASE_NOT_IN_SNAPSHOT}: no company filing is "
                            "available before the decision cutoff."
                        ),
                        claim_type=ClaimType.INFERENCE,
                        evidence_ids=[],
                    )
                ],
            )
            self._company_retrieval[symbol] = (output.model_copy(deep=True), [])
            return output, [], []
        invocations: list[InvocationResult] = []
        question_payload = {
            "op": "propose_question",
            "symbol": symbol,
            "task_id": task.task_id,
            "decision_cutoff": task.decision_cutoff.isoformat(),
            "instructions": (
                "Ask one factual question about a current company filing that could change "
                "the regular-session range. Earnings, guidance, mergers, buybacks, financing, "
                "litigation, management changes, cybersecurity, and product announcements all "
                "qualify. Do not answer it, and do not ask about a prior quarter. If this "
                "company has no filing in the supplied evidence, do not invent one."
            ),
        }
        question_call = self.client.generate(
            "company_evidence",
            question_payload,
            CompanyQuestion,
            attempt_type="company_question",
        )
        invocations.append(question_call)
        question = CompanyQuestion.model_validate(question_call.response).question

        def judge(active_question: str, query: str, hits: list, attempt: int) -> RetrievalJudgment:
            payload = {
                "op": "judge_retrieval",
                "symbol": symbol,
                "task_id": task.task_id,
                "decision_cutoff": task.decision_cutoff.isoformat(),
                "news_lookback_days": task.news_lookback_days,
                "question": active_question,
                "query": query,
                "attempt": attempt,
                "evidence": [item.model_dump(mode="json") for item in hits],
                "instructions": (
                    "Judge the retrieved passages against the question. company_match is true "
                    "only when every passage is about this company. event_match is true when a "
                    "passage is a current earnings, guidance, merger, buyback, financing, "
                    "litigation, management, cybersecurity, or product filing, not a prior "
                    "quarter. sufficient is true only when those passages can answer the "
                    "question. action is accept, rewrite, or record_gap. Rewrite at most once. "
                    "If no current material filing is present, do not rewrite; record gap "
                    f"{CURRENT_RELEASE_NOT_IN_SNAPSHOT}. On accept, list only the evidence_ids "
                    "for that current filing."
                ),
            }
            invocation = self.client.generate(
                "company_evidence",
                payload,
                RetrievalJudgment,
                attempt_type="retrieval_judgment",
            )
            invocations.append(invocation)
            return RetrievalJudgment.model_validate(invocation.response)

        result, accepted = run_corrective_retrieval(
            evidence=list(evidence),
            symbol=symbol,
            snapshot_id=snapshot_id,
            retrieval=self.config.retrieval,
            question=question,
            reserve_search=lambda: self._reserve_tool_call("company", symbol),
            judge=judge,
        )
        if result.gap or not accepted:
            output = CompanyAgentOutput(
                events=[],
                claims=[
                    Claim(
                        claim_id="claim_retrieval_gap",
                        text=(
                            f"{result.gap}: the current release is not supported by "
                            "the retrieved evidence."
                        ),
                        claim_type=ClaimType.INFERENCE,
                        evidence_ids=[],
                    )
                ],
            )
        else:
            payload = {
                "symbol": symbol,
                "task_id": task.task_id,
                "question": question,
                "evidence": [_accepted_payload(item) for item in accepted],
                "revision_feedback": revision_feedback or [],
                "instructions": (
                    "Use only the supplied evidence for this question. Extract normalized "
                    "events, observed factual claims, and metrics whose numbers quote an exact "
                    "chunk. Do not describe a prior release as the current one."
                ),
            }
            invocation = self.client.generate("company_evidence", payload, CompanyAgentOutput)
            invocations.append(invocation)
            output = normalize_company_output(
                CompanyAgentOutput.model_validate(invocation.response),
                accepted,
                task.decision_cutoff,
            )
            output = _with_sessions_ago(
                output,
                accepted,
                task,
                self.config.calendar["name"],
            )
            output = _restrict_to_accepted(
                output,
                {item.evidence_id for item in accepted},
            )
        self._company_retrieval[symbol] = (output.model_copy(deep=True), result.traces)
        return output, invocations, result.traces

    def synthesize(
        self,
        task: ScreeningTask,
        feature: FeatureRecord,
        company: CompanyAgentOutput,
        macro: MacroContext,
        revision_feedback: list[dict] | None = None,
    ) -> tuple[SynthesisOutput, InvocationResult]:
        payload = {
            "symbol": feature.symbol,
            "task_id": task.task_id,
            "feature": feature.model_dump(mode="json"),
            "company_claims": [item.model_dump(mode="json") for item in company.claims],
            "events": [item.model_dump(mode="json") for item in company.events],
            "company_metrics": [
                item.model_dump(mode="json") for item in company.metrics
            ],
            "revision_feedback": revision_feedback or [],
        }
        invocation = self.client.generate("synthesis", payload, SynthesisOutput)
        output = align_synthesis_output(
            SynthesisOutput.model_validate(invocation.response),
            company,
            macro,
        )
        return output, invocation

    def semantic_verify(
        self,
        card: ResearchCard,
        attempt: int,
        evidence: list[EvidenceRecord],
    ) -> tuple[VerificationReport, InvocationResult]:
        referenced_ids = {
            evidence_id
            for claim in card.claims
            for evidence_id in claim.evidence_ids
        } | {
            evidence_id
            for event in card.events
            for evidence_id in event.evidence_ids
        }
        if card.catalyst_assessment is not None:
            referenced_ids.update(card.catalyst_assessment.evidence_ids)
            referenced_ids.update(card.catalyst_assessment.counterevidence_ids)
            referenced_ids.update(
                citation.evidence_id
                for metric in card.catalyst_assessment.metrics
                for citation in metric.citations
            )
        semantic_query = " ".join(
            [
                *(claim.text for claim in card.claims),
                *(
                    event_type
                    for event in card.events
                    for event_type in event.event_types
                ),
            ]
        )
        payload = {
            "symbol": card.symbol,
            "semantic_scope": semantic_scope(card),
            "trusted_computed_context": {
                "feature_id": card.features.feature_id,
                "market_score": card.prediction.market_score,
            },
            "evidence": semantic_evidence_payloads(
                card,
                [item for item in evidence if item.evidence_id in referenced_ids],
                self.config.retrieval.top_n,
                semantic_query,
            ),
            "attempt": attempt,
        }
        invocation = self.client.generate("verifier", payload, VerificationReport)
        return normalize_semantic_verification(
            VerificationReport.model_validate(invocation.response)
        ), invocation


def normalize_macro_assessments(
    assessments: list[MacroAssessment],
    evidence: list[MacroEvidenceRecord],
    task: ScreeningTask,
    surprise_scales: dict[str, float],
) -> list[MacroAssessment]:
    """Bind macro interpretation to retrieved records and code-owned arithmetic."""

    proposed_by_id = {
        evidence_id: assessment
        for assessment in assessments
        for evidence_id in assessment.evidence_ids
    }
    result = []
    for item in evidence:
        if item.available_at > task.decision_cutoff:
            continue
        proposed = proposed_by_id.get(item.evidence_id)
        difference = (
            item.actual - item.consensus
            if item.actual is not None and item.consensus is not None
            else None
        )
        scale = surprise_scales.get(item.event_type)
        standardized = (
            difference / scale
            if difference is not None and scale is not None and scale > 0
            else None
        )
        direction = (
            "unknown"
            if difference is None
            else "positive"
            if difference > 0
            else "negative"
            if difference < 0
            else "neutral"
        )
        event_time = item.event_time
        same_session = (
            event_time is not None and event_time.date() == task.target_session
        )
        result.append(
            MacroAssessment(
                event_type=item.event_type,
                evidence_ids=[item.evidence_id],
                target_session_relevance=(
                    proposed.target_session_relevance
                    if proposed is not None and same_session
                    else "high"
                    if same_session
                    else "low"
                ),
                surprise_direction=direction,
                standardized_surprise=standardized,
                affected_risk_factors=sorted(set(item.affected_risk_factors)),
                target_session_timing=(
                    "unknown"
                    if event_time is None
                    else "before_open"
                    if event_time <= task.decision_cutoff
                    else "during_session"
                    if same_session
                    else "prior_session"
                ),
                confidence=item.timestamp_confidence,
            )
        )
    return result


def normalize_company_output(
    output: CompanyAgentOutput,
    evidence: list[EvidenceRecord],
    decision_cutoff: datetime,
) -> CompanyAgentOutput:
    """Fail closed on unsupported extraction fields before synthesis sees them."""

    by_id = {
        item.evidence_id: item
        for item in evidence
        if item.available_at <= decision_cutoff
    }
    claims = []
    for claim in output.claims:
        evidence_ids = sorted(
            evidence_id for evidence_id in set(claim.evidence_ids) if evidence_id in by_id
        )
        # The evidence role owns atomic facts only. Inferences belong to synthesis
        # rationale and must never be promoted into a fact by this normalization.
        if claim.claim_type != ClaimType.OBSERVED_FACT or not evidence_ids:
            continue
        claims.append(
            claim.model_copy(
                update={"evidence_ids": evidence_ids, "feature_ids": []}
            )
        )

    metrics = []
    seen_metrics: set[tuple] = set()
    for metric in output.metrics:
        if not metric.citations or not all(
            _citation_supported(citation, metric.value, by_id)
            for citation in metric.citations
        ):
            continue
        key = (
            metric.name.casefold(),
            metric.period.casefold(),
            metric.kind,
            metric.value,
            metric.unit.casefold(),
        )
        if key not in seen_metrics:
            seen_metrics.add(key)
            metrics.append(metric)

    events = []
    seen_events: dict[tuple, EventRecord] = {}
    for event in output.events:
        candidate_ids = sorted(
            evidence_id for evidence_id in set(event.evidence_ids) if evidence_id in by_id
        )
        evidence_ids = [
            evidence_id
            for evidence_id in candidate_ids
            if not by_id[evidence_id].event_types
            or bool(
                set(event.event_types)
                & set(by_id[evidence_id].event_types)
            )
        ]
        if not evidence_ids:
            continue
        authoritative_types = {
            event_type
            for evidence_id in evidence_ids
            for event_type in by_id[evidence_id].event_types
        }
        event_types = [
            event_type
            for event_type in event.event_types
            if not authoritative_types or event_type in authoritative_types
        ]
        if not event_types:
            continue
        matching_metadata = [
            by_id[evidence_id]
            for evidence_id in evidence_ids
            if set(event_types) & set(by_id[evidence_id].event_types)
            and by_id[evidence_id].event_status is not None
        ]
        status = event.status
        if matching_metadata and len(
            {item.event_status for item in matching_metadata}
        ) == 1:
            status = matching_metadata[0].event_status
        released_at = event.released_at
        scheduled_at = event.scheduled_at
        if status == "released_before_cutoff":
            scheduled_at = None
            if released_at is None:
                released_at = min(by_id[evidence_id].published_at for evidence_id in evidence_ids)
        elif status.startswith("scheduled_"):
            released_at = None
        normalized = event.model_copy(
            update={
                "status": status,
                "scheduled_at": scheduled_at,
                "released_at": released_at,
                "evidence_ids": evidence_ids,
                "event_types": event_types,
            }
        )
        event_time = normalized.scheduled_at or normalized.released_at
        key = (
            tuple(
                _normalize_text(value)
                for value in normalized.event_types
            ),
            event_time.date() if event_time is not None else None,
        )
        existing = seen_events.get(key)
        if existing is None:
            seen_events[key] = normalized
        else:
            seen_events[key] = existing.model_copy(
                update={
                    "evidence_ids": sorted(
                        set(existing.evidence_ids) | set(normalized.evidence_ids)
                    )
                }
            )
    events.extend(seen_events.values())
    ordered_events = sorted(
        events,
        key=lambda item: (
            tuple(
                _normalize_text(value)
                for value in item.event_types
            ),
            item.scheduled_at
            or item.released_at
            or datetime.min.replace(tzinfo=decision_cutoff.tzinfo),
            item.event_id,
        ),
    )
    return CompanyAgentOutput(
        events=[
            event.model_copy(update={"event_id": f"event_{index:03d}"})
            for index, event in enumerate(ordered_events)
        ],
        claims=claims,
        metrics=metrics,
    )


def align_synthesis_output(
    output: SynthesisOutput,
    company: CompanyAgentOutput,
    macro: MacroContext,
) -> SynthesisOutput:
    """Keep synthesis interpretive; extracted facts remain code-owned inputs."""

    allowed_ids = {
        evidence_id
        for claim in company.claims
        for evidence_id in claim.evidence_ids
    } | {
        evidence_id
        for event in company.events
        for evidence_id in event.evidence_ids
    } | {
        citation.evidence_id
        for metric in company.metrics
        for citation in metric.citations
    }
    assessment = output.catalyst_assessment.model_copy(
        update={
            "evidence_ids": sorted(
                set(output.catalyst_assessment.evidence_ids) & allowed_ids
            ),
            "counterevidence_ids": sorted(
                set(output.catalyst_assessment.counterevidence_ids) & allowed_ids
            ),
            "metrics": list(company.metrics),
        }
    )
    if not allowed_ids:
        assessment = assessment.model_copy(
            update={
                "target_session_relevance": "none",
                "novelty": "unknown",
                "materiality": "unknown",
                "expectedness": "unknown",
                "expected_abs_impact_bucket": "unknown",
                "surprise_direction": "unknown",
                "target_session_timing": "unknown",
                "signed_residual_bucket": "neutral",
                "field_confidence": {},
            }
        )
    return output.model_copy(update={"catalyst_assessment": assessment})


def normalize_semantic_verification(
    report: VerificationReport,
) -> VerificationReport:
    """Keep factual failures blocking while treating bounded forecasts as advisory."""

    inference_fields = (
        "expected_abs_impact_bucket",
        "expectedness",
        "novelty",
        "materiality",
        "signed_residual_bucket",
        "surprise_direction",
        "rationale",
    )
    issues = []
    for issue in report.issues:
        is_inference = "catalyst_assessment" in issue.field and any(
            name in issue.field for name in inference_fields
        )
        if is_inference and issue.code in {
            "BAD_INFERENCE",
            "INVALID_NUMERIC_CLAIM",
            "SEMANTIC_WARNING",
        }:
            issue = issue.model_copy(
                update={
                    "severity": "warning",
                    "required_action": "review_advisory",
                }
            )
        issues.append(issue)
    return VerificationReport(
        verdict=report.verdict,
        issues=issues,
        attempt=report.attempt,
    )


def semantic_scope(card: ResearchCard) -> dict:
    """Return only fields whose meaning—not arithmetic—belongs to the LLM verifier."""

    return {
        "symbol": card.symbol,
        "decision_cutoff": card.decision_cutoff,
        "target_session": card.target_session,
        "events": [item.model_dump(mode="json") for item in card.events],
        "claims": [item.model_dump(mode="json") for item in card.claims],
        "catalyst_assessment": (
            card.catalyst_assessment.model_dump(mode="json")
            if card.catalyst_assessment is not None
            else None
        ),
        "rationale": card.prediction.rationale,
        "counterarguments": card.prediction.counterarguments,
        "uncertainty": card.prediction.uncertainty,
    }


def semantic_evidence_payloads(
    card: ResearchCard,
    evidence: list[EvidenceRecord],
    limit: int,
    query: str | None = None,
) -> list[dict]:
    """Attach a small deterministic support set for semantic verification."""

    semantic_query = query or " ".join(
        [
            *(claim.text for claim in card.claims),
            *(
                event_type
                for event in card.events
                for event_type in event.event_types
            ),
        ]
    )
    return [
        {
            **item.model_dump(mode="json"),
            "supporting_chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "text": item.document_text[chunk.start_char : chunk.end_char],
                }
                for chunk in _supporting_chunks_for_card(
                    item,
                    card,
                    semantic_query,
                    limit,
                )
                if item.document_text is not None
            ],
        }
        for item in evidence
    ]


def _supporting_chunks_for_card(
    evidence: EvidenceRecord,
    card: ResearchCard,
    fallback_query: str,
    limit: int,
) -> list:
    """Select support per atomic field so one long claim cannot hide another."""

    selected: dict[str, EvidenceChunk] = {}
    for metric in (
        card.catalyst_assessment.metrics
        if card.catalyst_assessment is not None
        else []
    ):
        for citation in metric.citations:
            if citation.evidence_id != evidence.evidence_id:
                continue
            chunk = next(
                (
                    candidate
                    for candidate in evidence.chunks
                    if candidate.chunk_id == citation.chunk_id
                ),
                None,
            )
            if chunk is not None:
                selected[chunk.chunk_id] = chunk
    field_queries = [
        claim.text
        for claim in card.claims
        if evidence.evidence_id in claim.evidence_ids
    ] + [
        " ".join(
            value
            for value in (
                *event.event_types,
                event.status,
                str(event.scheduled_at or ""),
                str(event.released_at or ""),
            )
            if value
        )
        for event in card.events
        if evidence.evidence_id in event.evidence_ids
    ]
    for field_query in field_queries:
        for chunk in _supporting_chunks(evidence, field_query, 2):
            selected[chunk.chunk_id] = chunk
    if not selected:
        for chunk in _supporting_chunks(evidence, fallback_query, limit):
            selected[chunk.chunk_id] = chunk
    return [selected[key] for key in sorted(selected)[: max(limit * 2, limit)]]


def _supporting_chunks(
    evidence: EvidenceRecord,
    query: str,
    limit: int,
) -> list:
    if evidence.document_text is None:
        return []
    terms = set(re.findall(r"[a-z0-9]+", query.casefold()))
    scored = []
    for chunk in evidence.chunks:
        text = evidence.document_text[chunk.start_char : chunk.end_char]
        overlap = len(terms & set(re.findall(r"[a-z0-9]+", text.casefold())))
        scored.append((overlap, chunk.chunk_id, chunk))
    return [
        chunk
        for _, _, chunk in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]
    ]


def _citation_supported(
    citation: object,
    value: float,
    evidence: dict[str, EvidenceRecord],
) -> bool:
    item = evidence.get(citation.evidence_id)
    if item is None or item.document_text is None:
        return False
    chunk = next(
        (candidate for candidate in item.chunks if candidate.chunk_id == citation.chunk_id),
        None,
    )
    if chunk is None:
        return False
    text = item.document_text[chunk.start_char : chunk.end_char]
    if _normalize_text(citation.quote) not in _normalize_text(text):
        return False
    numbers = [
        float(token.replace(",", ""))
        for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", citation.quote)
    ]
    tolerance = max(1e-9, abs(value) * 1e-6)
    return any(abs(candidate - value) <= tolerance for candidate in numbers)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _with_sessions_ago(
    output: CompanyAgentOutput,
    evidence: list[EvidenceRecord],
    task: ScreeningTask,
    calendar_name: str,
) -> CompanyAgentOutput:
    """Attach deterministic recency after evidence ownership has been checked."""

    by_id = {item.evidence_id: item for item in evidence}
    calendar = TradingCalendar(calendar_name)
    normalized = []
    for event in output.events:
        event_dates = [
            by_id[evidence_id].event_time.date()
            for evidence_id in event.evidence_ids
            if evidence_id in by_id and by_id[evidence_id].event_time is not None
        ]
        sessions_ago = (
            min(
                calendar.session_distance(event_date, task.target_session)
                for event_date in event_dates
            )
            if event_dates
            else event.sessions_ago
        )
        normalized.append(event.model_copy(update={"sessions_ago": sessions_ago}))
    return output.model_copy(update={"events": normalized})


def _restrict_to_accepted(
    output: CompanyAgentOutput,
    accepted_ids: set[str],
) -> CompanyAgentOutput:
    events = [
        event
        for event in output.events
        if event.evidence_ids and set(event.evidence_ids) <= accepted_ids
    ]
    claims = [claim for claim in output.claims if set(claim.evidence_ids) <= accepted_ids]
    if not claims:
        claims = [
            Claim(
                claim_id="claim_no_accepted_statement",
                text="No statement was retained from the accepted evidence.",
                claim_type=ClaimType.INFERENCE,
                evidence_ids=[],
            )
        ]
    return output.model_copy(update={"events": events, "claims": claims})


def empty_company_output() -> CompanyAgentOutput:
    return CompanyAgentOutput(
        events=[],
        claims=[
            Claim(
                claim_id="claim_no_event",
                text="No company event was found in the configured search scope.",
                claim_type=ClaimType.INFERENCE,
                evidence_ids=[],
            )
        ],
    )
