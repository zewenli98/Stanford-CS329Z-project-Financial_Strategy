from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pytest

from financial_strategy.agents.roles import (
    CompanyAgentOutput,
    SynthesisOutput,
    align_synthesis_output,
    normalize_company_output,
    normalize_semantic_verification,
    scripted_responder,
    semantic_evidence_payloads,
    semantic_scope,
)
from financial_strategy.config import load_mvp_config
from financial_strategy.data.providers import LocalEvidenceProvider
from financial_strategy.ranking.ranker import (
    calibrate_market_scores,
    direct_verified_signed_residual,
    llm_residual_features,
    rank_cards,
    signed_llm_residual,
)
from financial_strategy.schemas import (
    CatalystAssessment,
    Eligibility,
    EvidenceChunk,
    EvidenceCitation,
    EvidenceRecord,
    ExtractedMetric,
    FeatureRecord,
    MacroContext,
    ResearchCard,
    SymbolMacroExposure,
    VerificationReport,
)
from financial_strategy.tuning.learning import fit_signed_ridge_residual
from financial_strategy.verification.deterministic import verify_card
from financial_strategy.workflow.symbol_graph import revision_target
from scripts.generate_llm_artifacts import _evenly_spaced, _select_grouped_sessions

ROOT = Path(__file__).resolve().parents[1]


def _feature() -> FeatureRecord:
    return FeatureRecord(
        feature_id="feature",
        symbol="AAPL",
        snapshot_id="snapshot",
        formula_version="v2",
        median_abs_oc_return_20d=0.01,
        realized_volatility_20d=0.02,
        return_5d=0.03,
        previous_session_relative_volume=1.2,
        distance_to_ma20=0.01,
        input_columns=["close"],
        input_sessions=[date(2026, 10, 30)],
    )


def _assessment() -> CatalystAssessment:
    return CatalystAssessment(
        target_session_relevance="high",
        novelty="high",
        materiality="high",
        expectedness="surprising",
        expected_abs_impact_bucket="over_5pct",
        evidence_ids=["e1"],
        signed_residual_bucket="strong_promote",
        target_session_timing="before_open",
        metrics=[
            ExtractedMetric(
                name="revenue",
                value=123.4,
                unit="USD million",
                period="Q3 2026",
                kind="reported",
                citations=[
                    EvidenceCitation(
                        evidence_id="e1",
                        chunk_id="e1_chunk_0000",
                        quote="Revenue was 123.4 USD million in Q3 2026.",
                    )
                ],
            )
        ],
    )


def test_local_provider_verifies_v2_document_and_chunks(tmp_path: Path) -> None:
    company = tmp_path / "company"
    documents = company / "documents"
    documents.mkdir(parents=True)
    text = "Revenue was 123.4 USD million in Q3 2026."
    document_hash = hashlib.sha256(text.encode()).hexdigest()
    documents.joinpath("e1.txt").write_text(text)
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    record = EvidenceRecord(
        evidence_id="e1",
        scope="company",
        symbol="AAPL",
        source_type="sec_filing",
        source_name="SEC",
        title="10-Q",
        excerpt=text,
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash=document_hash,
        snapshot_id="snapshot",
        timestamp_confidence="high",
        content_path="documents/e1.txt",
        schema_version=1,
        document_content_hash=document_hash,
        chunks=[
            EvidenceChunk(
                chunk_id="e1_chunk_0000",
                start_char=0,
                end_char=len(text),
                content_hash=document_hash,
            )
        ],
    )
    company.joinpath("evidence.jsonl").write_text(record.model_dump_json() + "\n")
    macro = tmp_path / "macro"
    macro.mkdir()
    macro.joinpath("events.jsonl").write_text("")
    loaded = LocalEvidenceProvider(
        company / "evidence.jsonl",
        macro / "events.jsonl",
    ).company_evidence()
    assert loaded[0].document_text == text


def test_metric_quote_is_deterministically_verified() -> None:
    text = "Revenue was 123.4 USD million in Q3 2026."
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    evidence = EvidenceRecord(
        evidence_id="e1",
        scope="company",
        symbol="AAPL",
        source_type="sec_filing",
        source_name="SEC",
        title="10-Q",
        excerpt=text,
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash="hash",
        snapshot_id="snapshot",
        timestamp_confidence="high",
        chunks=[
            EvidenceChunk(
                chunk_id="e1_chunk_0000",
                start_char=0,
                end_char=len(text),
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        ],
        document_text=text,
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=_assessment(),
    )
    assert verify_card(card, [evidence]).verdict == "PASS"
    bad = card.model_copy(deep=True)
    bad.catalyst_assessment.metrics[0].value = 999.0
    assert {
        issue.code for issue in verify_card(bad, [evidence]).issues
    } == {"INVALID_NUMERIC_CLAIM"}


def test_agent_contract_filters_unsupported_fields_before_synthesis() -> None:
    text = "The conference call is scheduled before the market opens. Revenue was 12.3."
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    evidence = EvidenceRecord(
        evidence_id="e1",
        scope="company",
        symbol="AAPL",
        source_type="sec_filing",
        source_name="SEC",
        title="8-K",
        excerpt=text,
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash="hash",
        snapshot_id="snapshot",
        timestamp_confidence="high",
        event_status="released_before_cutoff",
        chunks=[
            EvidenceChunk(
                chunk_id="e1_chunk_0000",
                start_char=0,
                end_char=len(text),
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            )
        ],
        document_text=text,
    )
    raw = CompanyAgentOutput.model_validate(
        {
            "events": [
                {
                    "event_id": "call",
                    "event_types": ["earnings"],
                    "status": "scheduled_before_open",
                    "scheduled_at": "2026-11-02T13:00:00Z",
                    "confidence": "high",
                    "evidence_ids": ["e1"],
                }
            ],
            "claims": [
                {
                    "claim_id": "supported",
                    "text": "A conference call was scheduled.",
                    "claim_type": "observed_fact",
                    "evidence_ids": ["e1"],
                },
                {
                    "claim_id": "hallucinated",
                    "text": "Unsupported.",
                    "claim_type": "observed_fact",
                    "evidence_ids": ["missing"],
                },
            ],
            "metrics": [
                {
                    "name": "revenue",
                    "value": 999.0,
                    "unit": "USD",
                    "period": "Q3",
                    "kind": "reported",
                    "citations": [
                        {
                            "evidence_id": "e1",
                            "chunk_id": "e1_chunk_0000",
                            "quote": "Revenue was 12.3.",
                        }
                    ],
                }
            ],
        }
    )
    normalized = normalize_company_output(
        raw,
        [evidence],
        datetime(2026, 11, 2, tzinfo=UTC),
    )
    assert [claim.claim_id for claim in normalized.claims] == ["supported"]
    assert normalized.metrics == []
    assert normalized.events[0].status == "scheduled_before_open"
    assert normalized.events[0].event_id == "event_000"

    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 2, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        events=normalized.events,
        claims=normalized.claims,
    )
    assert verify_card(card, [evidence]).verdict == "PASS"
    payload = semantic_evidence_payloads(card, [evidence], limit=2)
    assert payload[0]["supporting_chunks"][0]["chunk_id"] == "e1_chunk_0000"


def test_company_normalization_drops_mismatched_event_citations() -> None:
    timestamp = datetime(2026, 10, 30, tzinfo=UTC)
    evidence = EvidenceRecord(
        evidence_id="e1",
        scope="company",
        symbol="AAPL",
        source_type="sec_filing",
        source_name="SEC",
        title="8-K",
        excerpt="Quarterly earnings results.",
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash="hash",
        snapshot_id="snapshot",
        timestamp_confidence="high",
        event_types=["earnings"],
        event_status="released_before_cutoff",
    )
    output = CompanyAgentOutput.model_validate(
        {
            "events": [
                {
                    "event_id": "event",
                    "event_types": ["earnings", "cybersecurity"],
                    "status": "released_before_cutoff",
                    "released_at": timestamp.isoformat(),
                    "confidence": "high",
                    "evidence_ids": ["e1"],
                }
            ],
            "claims": [],
        }
    )
    normalized = normalize_company_output(
        output, [evidence], timestamp
    )
    assert normalized.events[0].event_types == ["earnings"]


def test_synthesis_cannot_create_facts_or_unverified_metrics() -> None:
    company = CompanyAgentOutput(
        events=[],
        claims=[],
        metrics=_assessment().metrics,
    )
    synthesis = SynthesisOutput(
        catalyst_assessment=CatalystAssessment(
            evidence_ids=["e1", "hallucinated"],
            metrics=[],
            signed_residual_bucket="promote",
        )
    )
    aligned = align_synthesis_output(
        synthesis,
        company,
        MacroContext(task_id="task", events=[], evidence_ids=[]),
    )
    assert aligned.catalyst_assessment.evidence_ids == ["e1"]
    assert aligned.catalyst_assessment.metrics == company.metrics

    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 2, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=aligned.catalyst_assessment,
    )
    scope = semantic_scope(card)
    assert "features" not in scope
    assert "market_score" not in scope


def test_revision_routes_to_the_role_that_owns_the_field() -> None:
    company_issue = VerificationReport.model_validate(
        {
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "INVALID_NUMERIC_CLAIM",
                    "severity": "blocking",
                    "field": "catalyst_assessment.metrics[0]",
                    "required_action": "correct_numeric_claim",
                }
            ],
        }
    )
    synthesis_issue = VerificationReport.model_validate(
        {
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "BAD_INFERENCE",
                    "severity": "blocking",
                    "field": "catalyst_assessment.signed_residual_bucket",
                    "required_action": "revise_assessment",
                }
            ],
        }
    )
    boundary_issue = VerificationReport.model_validate(
        {
            "verdict": "ABSTAIN",
            "issues": [
                {
                    "code": "TIME_BOUNDARY_VIOLATION",
                    "severity": "blocking",
                    "field": "evidence.available_at",
                    "required_action": "abstain",
                }
            ],
        }
    )
    assert revision_target(company_issue) == "company"
    assert revision_target(synthesis_issue) == "synthesis"
    assert revision_target(boundary_issue) == "abstain"


def test_signed_residual_requires_both_verifiers_and_can_demote() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    residual = config.ranking.residual_model.model_copy(
        update={
            "enabled": True,
            "coefficients": {"signed_bucket": 0.1},
            "feature_schema_hash": "schema",
            "model_hash": "model",
        }
    )
    ranking = config.ranking.model_copy(update={"residual_model": residual})
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=_assessment(),
        deterministic_verification=VerificationReport(verdict="PASS"),
        semantic_verification=VerificationReport(verdict="PASS"),
        eligibility=Eligibility(eligible=True),
    )
    assert signed_llm_residual(card, ranking) == pytest.approx(0.1)
    card.catalyst_assessment.signed_residual_bucket = "strong_demote"
    assert signed_llm_residual(card, ranking) == pytest.approx(-0.1)
    card.semantic_verification = VerificationReport(
        verdict="REVISE",
        issues=[
            {
                "code": "BAD_INFERENCE",
                "severity": "blocking",
                "field": "assessment",
                "required_action": "revise_assessment",
            }
        ],
    )
    assert signed_llm_residual(card, ranking) == 0.0
    assert llm_residual_features(card)["signed_bucket"] == -1.0


def test_disabled_ridge_still_applies_verified_signed_bucket() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    ranking = config.ranking.model_copy(update={"minimum_score": 0.0})
    promoted = _assessment().model_copy(
        update={
            "signed_residual_bucket": "promote",
            "target_session_timing": "unknown",
            "target_session_relevance": "high",
        }
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=promoted,
        deterministic_verification=VerificationReport(verdict="PASS"),
        semantic_verification=VerificationReport(verdict="PASS"),
        eligibility=Eligibility(eligible=True),
    )
    # 0.15 cap * promote 0.5 * unknown timing 0.5 * high relevance 1.0
    assert direct_verified_signed_residual(card, ranking) == pytest.approx(0.0375)
    peer = card.model_copy(update={"symbol": "MSFT", "catalyst_assessment": None})
    entries = rank_cards(
        [card, peer],
        {"AAPL": 0.5, "MSFT": 0.52},
        {"AAPL": 0.5, "MSFT": 0.52},
        MacroContext(task_id="task", events=[], evidence_ids=[]),
        {
            symbol: SymbolMacroExposure(
                symbol=symbol,
                sector="technology",
                exposure_method="sector_heuristic_v1",
                template_version="v1",
                exposures={},
            )
            for symbol in ("AAPL", "MSFT")
        },
        ranking,
        2,
    )
    by_symbol = {entry.symbol: entry for entry in entries}
    assert by_symbol["AAPL"].score == pytest.approx(by_symbol["AAPL"].market_score + 0.0375)
    assert by_symbol["MSFT"].score == pytest.approx(by_symbol["MSFT"].market_score)
    card.semantic_verification = VerificationReport(
        verdict="REVISE",
        issues=[
            {
                "code": "BAD_INFERENCE",
                "severity": "blocking",
                "field": "catalyst_assessment",
                "required_action": "revise_assessment",
            }
        ],
    )
    assert direct_verified_signed_residual(card, ranking) == pytest.approx(0.0375)
    blocked = rank_cards(
        [card],
        {"AAPL": 0.5},
        {"AAPL": 0.5},
        MacroContext(task_id="task", events=[], evidence_ids=[]),
        {
            "AAPL": SymbolMacroExposure(
                symbol="AAPL",
                sector="technology",
                exposure_method="sector_heuristic_v1",
                template_version="v1",
                exposures={},
            )
        },
        ranking,
        1,
    )
    assert blocked[0].score == pytest.approx(blocked[0].market_score)


def test_signed_residual_does_not_apply_intercept_without_features() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    residual = config.ranking.residual_model.model_copy(
        update={
            "enabled": True,
            "intercept": 0.1,
            "coefficients": {"signed_bucket": 0.1},
            "feature_schema_hash": "schema",
            "model_hash": "model",
        }
    )
    ranking = config.ranking.model_copy(update={"residual_model": residual})
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        deterministic_verification=VerificationReport(verdict="PASS"),
        semantic_verification=VerificationReport(verdict="PASS"),
        eligibility=Eligibility(eligible=True),
    )
    assert llm_residual_features(card) == {}
    assert signed_llm_residual(card, ranking) == 0.0


def test_unknown_impact_remains_trainable_when_evidence_is_present() -> None:
    assessment = _assessment().model_copy(
        update={
            "materiality": "unknown",
            "expected_abs_impact_bucket": "unknown",
        }
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=assessment,
    )
    features = llm_residual_features(card)
    assert features["materiality"] == 0
    assert features["expected_abs_impact"] == 0


def test_residual_features_require_same_day_timing() -> None:
    assessment = _assessment().model_copy(
        update={"target_session_timing": "prior_session"}
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=assessment,
    )
    assert llm_residual_features(card) == {}


def test_numeric_surprise_uses_matched_reported_and_consensus() -> None:
    reported = _assessment().metrics[0]
    consensus = reported.model_copy(
        update={"kind": "consensus", "value": 100.0}
    )
    assessment = _assessment().model_copy(
        update={"metrics": [reported, consensus]}
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=assessment,
    )
    features = llm_residual_features(card)
    assert features["numeric_surprise_magnitude"] == pytest.approx(0.234)
    assert features["has_numeric_surprise"] == 1.0


def test_negative_surprise_preserves_signed_demote_bucket() -> None:
    assessment = _assessment().model_copy(
        update={
            "surprise_direction": "negative",
            "signed_residual_bucket": "strong_demote",
        }
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=assessment,
    )
    assert llm_residual_features(card)["signed_bucket"] == -1.0

    card.catalyst_assessment.surprise_direction = "unknown"
    assert llm_residual_features(card)["signed_bucket"] == -1.0


def test_semantic_inference_issues_are_advisory_but_facts_stay_blocking() -> None:
    inference = VerificationReport.model_validate(
        {
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "BAD_INFERENCE",
                    "severity": "blocking",
                    "field": "catalyst_assessment.expected_abs_impact_bucket",
                    "required_action": "revise_assessment",
                }
            ],
        }
    )
    assert normalize_semantic_verification(inference).verdict == "PASS"

    factual = VerificationReport.model_validate(
        {
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "MISSING_EVIDENCE",
                    "severity": "blocking",
                    "field": "claims.claim_001",
                    "required_action": "retrieve_primary_source",
                }
            ],
        }
    )
    assert normalize_semantic_verification(factual).verdict == "REVISE"


def test_scripted_synthesis_exercises_residual_feature_contract() -> None:
    response = scripted_responder(
        "synthesis",
        {
            "company_claims": [
                {
                    "claim_id": "claim",
                    "text": "A filing was released.",
                    "claim_type": "observed_fact",
                    "evidence_ids": ["e1"],
                }
            ]
        },
    )
    assessment = SynthesisOutput.model_validate(response).catalyst_assessment
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=assessment,
    )
    encoded = llm_residual_features(card)
    assert encoded["materiality"] > 0
    assert encoded["expected_abs_impact"] > 0
    assert encoded["signed_bucket"] == 0


def test_signed_ridge_residual_recovers_direction() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = np.asarray([-0.2, 0.0, 0.2, 0.4])
    intercept, coefficients = fit_signed_ridge_residual(
        features,
        targets,
        shrinkage=0.01,
    )
    assert intercept < 0
    assert coefficients[0] > 0


def test_market_scores_are_calibrated_to_cross_sectional_percentiles() -> None:
    calibrated = calibrate_market_scores(
        {"LOW": 0.2, "MID_A": 0.6, "MID_B": 0.6, "HIGH": 0.9}
    )
    assert calibrated == {
        "LOW": 0.0,
        "MID_A": pytest.approx(0.5),
        "MID_B": pytest.approx(0.5),
        "HIGH": 1.0,
    }
    assert calibrate_market_scores({"ONLY": 0.7}) == {"ONLY": 0.5}


def test_llm_pilot_sampling_spans_full_period_without_labels() -> None:
    values = [{"session": index} for index in range(100)]
    selected = _evenly_spaced(values, 5)
    assert [item["session"] for item in selected] == [0, 24, 49, 74, 99]


def test_session_sampling_keeps_complete_candidate_cross_sections() -> None:
    values = [
        {"target_session": session, "symbol": symbol}
        for session in ("2026-01-02", "2026-01-05", "2026-01-06")
        for symbol in ("A", "B")
    ]
    selected = _select_grouped_sessions(values, 2)
    assert selected == [
        {"target_session": "2026-01-02", "symbol": "A"},
        {"target_session": "2026-01-02", "symbol": "B"},
        {"target_session": "2026-01-06", "symbol": "A"},
        {"target_session": "2026-01-06", "symbol": "B"},
    ]
