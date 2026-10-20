"""Non-LLM verification of time, references, and required evidence."""

from __future__ import annotations

from financial_strategy.schemas import (
    ClaimType,
    EvidenceRecord,
    ResearchCard,
    Verdict,
    VerificationIssue,
    VerificationReport,
)


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
                    code="WRONG_CITATION",
                    severity="blocking",
                    field="claims/events.evidence_ids",
                    required_action="correct_citation",
                )
            )
    for event in card.events:
        for evidence_id in event.evidence_ids:
            item = by_id.get(evidence_id)
            if item is None:
                continue
            if item.event_status is not None and item.event_status != event.status:
                issues.append(
                    VerificationIssue(
                        code="UNSUPPORTED_EVENT_STATUS",
                        severity="blocking",
                        field=f"events[{event.event_id}].status",
                        required_action="retrieve_primary_source",
                    )
                )
            if item.event_type is not None and item.event_type != event.event_type:
                issues.append(
                    VerificationIssue(
                        code="WRONG_CITATION",
                        severity="blocking",
                        field=f"events[{event.event_id}].event_type",
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
    verdict = Verdict.PASS if not issues else Verdict.REVISE
    if any(issue.code == "TIME_BOUNDARY_VIOLATION" for issue in issues):
        verdict = Verdict.ABSTAIN
    return VerificationReport(verdict=verdict, issues=issues, attempt=attempt)
