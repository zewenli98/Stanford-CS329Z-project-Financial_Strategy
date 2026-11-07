"""Regressions for source-supported cards lost to masking and numeric/date parsing."""

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from financial_strategy.adaptive.agents import (
    ResearchAgentSuite,
    _numbers,
    verify_card_deterministic,
)
from financial_strategy.adaptive.models import InfoCard, RangeForecast, ResearchQuestion
from financial_strategy.adaptive.tools import OpaqueToolView
from financial_strategy.schemas import EvidenceRecord

CUTOFF = datetime(2026, 7, 24, 13, tzinfo=UTC)


def make_view(
    text="Intel Corporation reported revenue growth of 3% - 4%.", *, target=True, protocol=True
):
    stamp = CUTOFF - timedelta(days=1)
    record = EvidenceRecord(
        evidence_id="intel_release",
        scope="company",
        symbol="INTC",
        source_type="company_release",
        source_name="issuer",
        title="Intel Corporation update",
        excerpt=text,
        published_at=stamp,
        first_seen_at=stamp,
        retrieved_at=stamp,
        available_at=stamp,
        content_hash="f" * 64,
        snapshot_id="frozen",
        timestamp_confidence="high",
        event_type="guidance",
        event_status="released_before_cutoff",
        event_time=stamp,
    )
    universe = {
        "securities": [
            {"symbol": "INTC", "company_name": "Intel Corporation"},
            {"symbol": "TGT", "company_name": "Target Corporation"},
            {"symbol": "ON", "company_name": "ON Semiconductor"},
        ]
    }
    features = {s: {} for s in (["INTC", "TGT", "ON"] if target else ["INTC", "ON"])}
    return OpaqueToolView(
        SimpleNamespace(company_evidence=[record], macro_evidence=[]),
        SimpleNamespace(decision_cutoff=CUTOFF, target_session=date(2026, 7, 24)),
        universe,
        features,
        protocol_v2=protocol,
    )


def card_and_question(view, observation):
    source, metadata = next(iter(view._records.items()))
    span = view.source_span(source, 0, len(view.documents[source]))
    alias = view.aliases["INTC"]
    question = ResearchQuestion(
        question_id="q",
        text="Assess uncertainty for the target date.",
        symbols=[alias],
        evidence_ids=[source],
        discovery=True,
        resolution_probability=1,
        reserved_cost_usd=0.1,
    )
    card = InfoCard(
        card_id="c",
        question_id="q",
        symbols=[alias],
        event_id=metadata["event_id"],
        event_type="guidance",
        observations=[observation],
        range_mechanism="Uncertain repricing.",
        source_spans=[span],
        available_at=CUTOFF - timedelta(days=1),
    )
    return card, question


@pytest.mark.parametrize("protocol", [False, True])
@pytest.mark.parametrize("target", [False, True])
def test_common_target_language_survives_with_and_without_issuer_in_shortlist(protocol, target):
    view = make_view(target=target, protocol=protocol)
    for text in [
        "target date",
        "Target session",
        "TARGET PRICE",
        "Target the next session.",
        "on a low range",
    ]:
        assert view.redact_text(text) == text
    expected = view.aliases.get("TGT", "[other_company]")
    for text in [
        "Target Corporation",
        "target corporation",
        "$TGT",
        "Target shares",
        "TARGET reported",
    ]:
        assert expected in view.redact_text(text)
    assert view.aliases["INTC"] in view.redact_text("Intel raised guidance.")
    assert view.aliases["INTC"] in view.redact_text("intel raised guidance.")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3%-4%", {"3", "4"}),
        ("3% - 4%", {"3", "4"}),
        ("20.1%-20.5%", {"20.1", "20.5"}),
        ("3%\u20134%", {"3", "4"}),
        ("$19.8B-$20.0B", {"19.8", "2E+1"}),
        ("3 million-4 million", {"3", "4"}),
        ("-4%", {"-4"}),
        ("-4% to -3%", {"-4", "-3"}),
        ("1e-3", {"0.001"}),
    ],
)
def test_ranges_keep_positive_endpoints_and_preserve_actual_negative_values(text, expected):
    assert _numbers(text) == expected


def test_source_supported_positive_range_passes_but_wrong_sign_and_value_fail():
    view = make_view()
    card, question = card_and_question(view, "Revenue growth was 3%-4%.")
    assert verify_card_deterministic(card, question, view) == []
    for observation in ["Revenue growth was -4%.", "Revenue growth was 3%-9%."]:
        card.observations = [observation]
        assert "UNSUPPORTED_NUMERICAL_FACT" in verify_card_deterministic(card, question, view)


@pytest.mark.parametrize("mentioned", ["2026-07-24", "July 24, 2026", "Jul 24"])
def test_known_task_date_is_context_and_does_not_whitelist_a_financial_number(mentioned):
    view = make_view()
    card, question = card_and_question(view, "The filing precedes the " + mentioned + " session.")
    assert verify_card_deterministic(card, question, view) == []
    card.observations.append("Revenue growth was 24%.")
    assert "UNSUPPORTED_NUMERICAL_FACT" in verify_card_deterministic(card, question, view)


@pytest.mark.parametrize("mentioned", ["2026-07-25", "July 25, 2026", "2026-02-30"])
def test_unknown_or_invalid_date_stays_rejected(mentioned):
    view = make_view()
    card, question = card_and_question(view, "The filing preceded the " + mentioned + " session.")
    assert "UNSUPPORTED_DATE_FACT" in verify_card_deterministic(card, question, view)


def test_future_effective_date_in_source_remains_admissible():
    view = make_view(
        "Intel Corporation announced on July 23, 2026 an effective date of 2026-08-01."
    )
    card, question = card_and_question(view, "The announced effective date is 2026-08-01.")
    assert verify_card_deterministic(card, question, view) == []


def test_verifier_sees_unchanged_target_session_language_and_forecaster_receives_task_context():
    view = make_view()
    card, question = card_and_question(view, "Revenue growth was 3%-4%.")
    card.inferences = ["The catalyst may affect the target session."]
    calls = []

    class Client:
        def generate(self, role, payload, schema, *, attempt_type="initial"):
            calls.append(payload)
            if payload["op"] == "adaptive_verify":
                assert payload["card"]["inferences"] == card.inferences
                assert payload["question"]["text"] == question.text
                response = {"verdict": "PASS", "attempt": 0, "issues": []}
            else:
                response = {"forecasts": payload["forecasts"]}
            return SimpleNamespace(response=response)

    suite = ResearchAgentSuite(Client(), view, limits={"research_protocol": "packet"})
    assert suite._semantic_issues(card, [], question, []) == []
    card.status = "accepted"
    suite.update_forecasts(
        [RangeForecast(symbol=view.aliases["INTC"], baseline_range=0.03, predicted_range=0.03)],
        [card],
    )
    assert len(calls) == 2
    assert all(
        c["target_session"] == "2026-07-24" and c["decision_cutoff"] == CUTOFF.isoformat()
        for c in calls
    )


def test_numeric_date_fixes_do_not_admit_wrong_entity_or_altered_quote():
    view = make_view()
    card, question = card_and_question(view, "The filing precedes the 2026-07-24 session.")
    card.symbols = [view.aliases["TGT"]]
    assert "WRONG_ENTITY" in verify_card_deterministic(card, question, view)
    card.symbols = [view.aliases["INTC"]]
    card.source_spans[0].quote = "Changed quotation."
    assert "SOURCE_SPAN_MISMATCH" in verify_card_deterministic(card, question, view)
