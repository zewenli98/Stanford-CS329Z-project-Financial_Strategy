"""Non-LLM verification of time, references, and required evidence."""

from __future__ import annotations

import re

from financial_strategy.schemas import (
    ClaimType,
    EvidenceRecord,
    MacroContext,
    MacroEvidenceRecord,
    ResearchCard,
    Verdict,
    VerificationIssue,
    VerificationReport,
)


def verify_macro_context(
    context: MacroContext,
    evidence: list[MacroEvidenceRecord],
    surprise_scales: dict[str, float],
) -> VerificationReport:
    """Verify task-level macro assessments before symbol fan-out."""

    by_id = {item.evidence_id: item for item in evidence}
    issues: list[VerificationIssue] = []
    for index, assessment in enumerate(context.assessments):
        field = f"macro_context.assessments[{index}]"
        if len(assessment.evidence_ids) != 1:
            issues.append(
                VerificationIssue(
                    code="WRONG_CITATION",
                    severity="blocking",
                    field=f"{field}.evidence_ids",
                    required_action="correct_citation",
                )
            )
            continue
        item = by_id.get(assessment.evidence_ids[0])
        if item is None:
            issues.append(
                VerificationIssue(
                    code="MISSING_EVIDENCE",
                    severity="blocking",
                    field=f"{field}.evidence_ids",
                    required_action="retrieve_primary_source",
                )
            )
            continue
        if context.decision_cutoff is not None and item.available_at > context.decision_cutoff:
            issues.append(
                VerificationIssue(
                    code="TIME_BOUNDARY_VIOLATION",
                    severity="blocking",
                    field=f"{field}.available_at",
                    required_action="abstain",
                )
            )
        if item.event_type != assessment.event_type:
            issues.append(
                VerificationIssue(
                    code="WRONG_CITATION",
                    severity="blocking",
                    field=f"{field}.event_type",
                    required_action="correct_citation",
                )
            )
        if not set(assessment.affected_risk_factors).issubset(
            item.affected_risk_factors
        ):
            issues.append(
                VerificationIssue(
                    code="BAD_INFERENCE",
                    severity="blocking",
                    field=f"{field}.affected_risk_factors",
                    required_action="revise_assessment",
                )
            )
        scale = surprise_scales.get(item.event_type)
        if (
            item.actual is not None
            and item.consensus is not None
            and scale is not None
            and assessment.standardized_surprise is not None
        ):
            expected = (item.actual - item.consensus) / scale
            if abs(expected - assessment.standardized_surprise) > 1e-9:
                issues.append(
                    VerificationIssue(
                        code="INVALID_NUMERIC_CLAIM",
                        severity="blocking",
                        field=f"{field}.standardized_surprise",
                        required_action="correct_numeric_claim",
                    )
                )
    verdict = Verdict.PASS if not issues else Verdict.REVISE
    if any(issue.code == "TIME_BOUNDARY_VIOLATION" for issue in issues):
        verdict = Verdict.ABSTAIN
    return VerificationReport(verdict=verdict, issues=issues)


def verify_card(
    card: ResearchCard,
    evidence: list[EvidenceRecord],
    attempt: int = 0,
) -> VerificationReport:
    by_id = {item.evidence_id: item for item in evidence}
    issues: list[VerificationIssue] = []

    referenced = {
        evidence_id
        for claim in card.claims
        for evidence_id in claim.evidence_ids
    } | {
        evidence_id
        for event in card.events
        for evidence_id in event.evidence_ids
    }
    if card.catalyst_assessment is not None:
        referenced.update(card.catalyst_assessment.evidence_ids)
        referenced.update(card.catalyst_assessment.counterevidence_ids)
        referenced.update(
            citation.evidence_id
            for metric in card.catalyst_assessment.metrics
            for citation in metric.citations
        )
    for evidence_id in sorted(referenced):
        item = by_id.get(evidence_id)
        if item is None:
            issues.append(
                VerificationIssue(
                    code="MISSING_EVIDENCE",
                    severity="blocking",
                    field="claims/events.evidence_ids",
                    required_action="retrieve_primary_source",
                )
            )
        elif item.available_at > card.decision_cutoff:
            issues.append(
                VerificationIssue(
                    code="TIME_BOUNDARY_VIOLATION",
                    severity="blocking",
                    field="evidence.available_at",
                    required_action="abstain",
                )
            )
        elif item.scope == "company" and item.symbol != card.symbol:
            issues.append(
                VerificationIssue(
                    code="ENTITY_MISMATCH",
                    severity="blocking",
                    field="claims/events.evidence_ids",
                    required_action="abstain",
                )
            )
    for event in card.events:
        for evidence_id in event.evidence_ids:
            item = by_id.get(evidence_id)
            if item is None:
                continue
            item_event_types = set(item.event_types)
            event_matches = bool(set(event.event_types) & item_event_types)
            # A filing can describe a different event (for example, a future
            # conference call). Its own release status is authoritative only
            # when the fixture metadata includes the same event type.
            if (
                event_matches
                and item.event_status is not None
                and item.event_status != event.status
            ):
                issues.append(
                    VerificationIssue(
                        code="UNSUPPORTED_EVENT_STATUS",
                        severity="blocking",
                        field=f"events[{event.event_id}].status",
                        required_action="retrieve_primary_source",
                    )
                )
            if item_event_types and not event_matches:
                issues.append(
                    VerificationIssue(
                        code="WRONG_CITATION",
                        severity="blocking",
                        field=f"events[{event.event_id}].event_types",
                        required_action="correct_citation",
                    )
                )
    for claim in card.claims:
        if (
            claim.claim_type == ClaimType.OBSERVED_FACT
            and not claim.evidence_ids
            and not claim.feature_ids
        ):
            issues.append(
                VerificationIssue(
                    code="MISSING_EVIDENCE",
                    severity="blocking",
                    field=f"claims[{claim.claim_id}]",
                    required_action="retrieve_primary_source",
                )
            )
        for feature_id in claim.feature_ids:
            if feature_id != card.features.feature_id:
                issues.append(
                    VerificationIssue(
                        code="WRONG_CITATION",
                        severity="blocking",
                        field=f"claims[{claim.claim_id}].feature_ids",
                        required_action="correct_citation",
                    )
                )
    if card.catalyst_assessment is not None:
        for metric_index, metric in enumerate(card.catalyst_assessment.metrics):
            for citation in metric.citations:
                item = by_id.get(citation.evidence_id)
                field = f"catalyst_assessment.metrics[{metric_index}]"
                if item is None or item.document_text is None:
                    issues.append(
                        VerificationIssue(
                            code="MISSING_EVIDENCE",
                            severity="blocking",
                            field=field,
                            required_action="retrieve_primary_source",
                        )
                    )
                    continue
                chunk = next(
                    (
                        candidate
                        for candidate in item.chunks
                        if candidate.chunk_id == citation.chunk_id
                    ),
                    None,
                )
                if chunk is None:
                    issues.append(
                        VerificationIssue(
                            code="WRONG_CITATION",
                            severity="blocking",
                            field=field,
                            required_action="correct_citation",
                        )
                    )
                    continue
                content = item.document_text[chunk.start_char : chunk.end_char]
                if _normalize_text(citation.quote) not in _normalize_text(content):
                    issues.append(
                        VerificationIssue(
                            code="WRONG_CITATION",
                            severity="blocking",
                            field=field,
                            required_action="correct_citation",
                        )
                    )
                if not _quote_contains_value(citation.quote, metric.value):
                    issues.append(
                        VerificationIssue(
                            code="INVALID_NUMERIC_CLAIM",
                            severity="blocking",
                            field=field,
                            required_action="correct_numeric_claim",
                        )
                    )
    verdict = Verdict.PASS if not issues else Verdict.REVISE
    if any(
        issue.code in {"TIME_BOUNDARY_VIOLATION", "ENTITY_MISMATCH"}
        for issue in issues
    ):
        verdict = Verdict.ABSTAIN
    return VerificationReport(verdict=verdict, issues=issues, attempt=attempt)


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _quote_contains_value(quote: str, value: float) -> bool:
    numbers = [
        float(token.replace(",", ""))
        for token in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", quote)
    ]
    tolerance = max(1e-9, abs(value) * 1e-6)
    return any(abs(candidate - value) <= tolerance for candidate in numbers)
