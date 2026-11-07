"""Behavioral tests for the closed research boundary and evidence update protocol."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import pytest

from financial_strategy.adaptive.agents import (
    ResearchAgentSuite,
    deterministic_responder,
    verify_card_deterministic,
)
from financial_strategy.adaptive.memory import AcceptedCardMemory
from financial_strategy.adaptive.models import InfoCard, RangeForecast, ResearchQuestion
from financial_strategy.adaptive.tools import OpaqueToolView, ResearchToolError
from financial_strategy.schemas import EvidenceRecord

CUTOFF = datetime(2026, 10, 14, 13, tzinfo=UTC)


def evidence(symbol="AAPL", *, at=None, text=None, evidence_id="private_aapl_id"):
    text = text or "Apple Inc. reports revenue growth of 3.7% and unresolved regulatory review."
    available = at or CUTOFF - timedelta(hours=1)
    return EvidenceRecord(
        evidence_id=evidence_id,
        scope="company",
        symbol=symbol,
        source_type="company_release",
        source_name="apple.com",
        source_url="https://apple.com/news/private_aapl_id",
        title="Apple Inc. update",
        excerpt=text,
        published_at=available,
        first_seen_at=available,
        retrieved_at=max(CUTOFF, available),
        available_at=available,
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        snapshot_id="private_snapshot",
        timestamp_confidence="high",
        event_type="guidance",
        event_status="released_before_cutoff",
        event_time=at or CUTOFF - timedelta(hours=1),
    )


def view_for(records=None, *, reserve=None, features=None, memory=None):
    universe = {
        "securities": [
            {
                "symbol": "AAPL",
                "company_name": "Apple Inc.",
                "cik": "0000320193",
                "permanent_entity_id": "sec_cik_0000320193",
                "sector": "technology",
            },
            {"symbol": "MSFT", "company_name": "Microsoft Corp", "sector": "technology"},
        ]
    }
    features = features or {
        "AAPL": {"mean_range_20d": 0.025, "latest_range": 0.03},
        "MSFT": {"mean_range_20d": 0.02, "latest_range": 0.021},
    }
    snapshot = SimpleNamespace(company_evidence=records or [evidence()], macro_evidence=[])
    task = SimpleNamespace(decision_cutoff=CUTOFF, target_session=date(2026, 10, 14))
    return OpaqueToolView(snapshot, task, universe, features, reserve_tool=reserve, memory=memory)


def question_for(view):
    lead = view.lead_index()[0]
    return ResearchQuestion(
        question_id="question_test",
        text="Does the event leave unresolved session uncertainty?",
        symbols=[view.aliases["AAPL"]],
        evidence_ids=[lead["evidence_id"]],
        event_ids=[lead["event_id"]],
        discovery=True,
        resolution_probability=0.5,
        reserved_cost_usd=0.1,
    )


def card_for(view, *, card_id="card_test"):
    lead = view.lead_index()[0]
    span = view.source_span(lead["evidence_id"], 0, len(view.documents[lead["evidence_id"]]))
    return InfoCard(
        card_id=card_id,
        question_id="question_test",
        symbols=[view.aliases["AAPL"]],
        event_id=lead["event_id"],
        event_type="guidance",
        observations=[span["quote"]],
        inferences=["The unresolved review could affect the session range."],
        range_mechanism="Unresolved information may generate intraday repricing.",
        source_spans=[span],
        available_at=CUTOFF - timedelta(hours=1),
    )


def forecasts_for(view):
    return [
        RangeForecast(symbol=alias, baseline_range=0.025, predicted_range=0.025)
        for alias in view.reverse_aliases
    ]


class LocalClient:
    def __init__(self, responder=deterministic_responder):
        self.responder = responder
        self.payloads = []

    def generate(self, role, payload, schema, *, attempt_type="initial"):
        self.payloads.append((role, payload, attempt_type))
        raw = self.responder(role, payload)
        return SimpleNamespace(response=schema.model_validate(raw), raw=raw)


def test_tools_exclude_future_and_outside_universe_and_reserve_every_call():
    calls = []
    records = [
        evidence(),
        evidence(at=CUTOFF + timedelta(seconds=1), text="future secret"),
        evidence("OUTSIDE", text="outside secret", evidence_id="outside"),
    ]
    view = view_for(records, reserve=calls.append)
    output = view.dispatch({"tool": "search", "arguments": {"query": ""}})
    assert output["eligible_count"] == 1
    assert calls == ["search"]
    assert "future secret" not in json.dumps(output)
    assert "outside secret" not in json.dumps(output)
    with pytest.raises(ResearchToolError, match="allowlist"):
        view.dispatch({"tool": "web", "arguments": {"url": "https://example.com"}})
    with pytest.raises(ResearchToolError, match="unexpected tool"):
        view.dispatch({"tool": "read", "arguments": {"path": "evaluation/labels.parquet"}})


def test_identity_masking_and_stable_sources_when_eligible_corpus_changes():
    view = view_for()
    earlier = evidence(
        text="Microsoft Corp announced a separate change.",
        symbol="MSFT",
        at=CUTOFF - timedelta(days=1),
        evidence_id="msft_private",
    )
    expanded = view_for([earlier, evidence()])
    original_id = view.lead_index()[0]["evidence_id"]
    assert original_id in expanded.evidence_map
    visible = json.dumps(view.model_context()) + json.dumps(view.lead_index())
    visible += view.redact_text(
        "Apple Inc. AAPL CIK 0000320193 https://apple.com/a C:\\private\\AAPL.txt"
    )
    for identifier in ("Apple", "AAPL", "0000320193", "apple.com", "private_aapl_id"):
        assert identifier.lower() not in visible.lower()
    assert "company_0001" in visible
    # A missing feature must not renumber another company across evaluation days.
    subset = view_for(features={"MSFT": {"mean_range_20d": 0.02}})
    assert subset.aliases["MSFT"] == view.aliases["MSFT"]


def test_deterministic_verifier_rejects_mismatched_span_and_unsupported_number():
    view = view_for()
    card = card_for(view)
    question = question_for(view)
    assert verify_card_deterministic(card, question, view) == []
    card.source_spans[0].quote = "Invented quotation"
    card.observations = ["Revenue grew 9.9%."]
    issues = verify_card_deterministic(card, question, view)
    assert "SOURCE_SPAN_MISMATCH" in issues
    assert "UNSUPPORTED_NUMERICAL_FACT" in issues


def test_unread_and_wrong_entity_evidence_cannot_be_accepted():
    view = view_for()
    card = card_for(view)
    assert "UNREAD_SOURCE" in verify_card_deterministic(card, question_for(view), view, [])
    card.symbols = [view.aliases["MSFT"]]
    assert "WRONG_ENTITY" in verify_card_deterministic(card, question_for(view), view)
    card.available_at = CUTOFF - timedelta(days=2)
    assert "BACKDATED_AVAILABILITY" in verify_card_deterministic(card, question_for(view), view)


def test_targeted_repair_receives_actual_verifier_issues():
    view = view_for()
    finishes = 0

    def responder(role, payload):
        nonlocal finishes
        raw = deterministic_responder(role, payload)
        if payload["op"] == "adaptive_research" and raw["tool"] == "finish":
            finishes += 1
            if finishes == 1:
                raw["cards"][0]["source_spans"][0]["quote"] = "fabricated"
            else:
                assert "SOURCE_SPAN_MISMATCH" in payload["verification_issues"]
        return raw

    client = LocalClient(responder)
    suite = ResearchAgentSuite(client, view)
    result = suite.research(question_for(view), forecasts_for(view))
    assert result.stop_reason == "accepted"
    assert len(result.accepted_cards) == 1
    assert len(result.rejected_cards) == 1
    assert result.accepted_cards[0].version == 2
    assert sum(role == "company_evidence" for role, _, _ in client.payloads) == 3


def test_semantic_rejection_is_not_overridden_by_exact_citation():
    view = view_for()

    def responder(role, payload):
        if payload["op"] == "adaptive_verify":
            return {
                "verdict": "ABSTAIN",
                "attempt": 0,
                "issues": [
                    {
                        "code": "BAD_INFERENCE",
                        "severity": "blocking",
                        "field": "range_mechanism",
                        "required_action": "Explain why repricing remains after open.",
                    }
                ],
            }
        return deterministic_responder(role, payload)

    result = ResearchAgentSuite(LocalClient(responder), view).research(
        question_for(view), forecasts_for(view)
    )
    assert not result.accepted_cards
    assert result.stop_reason == "repair_rejected"
    assert "BAD_INFERENCE" in result.rejected_cards[-1].verification_issues[0]


def test_forecast_is_unchanged_without_new_accepted_evidence():
    view = view_for()
    client = LocalClient()
    suite = ResearchAgentSuite(client, view)
    forecasts = forecasts_for(view)
    assert suite.update_forecasts(forecasts, []) == forecasts
    pending = card_for(view)
    assert suite.update_forecasts(forecasts, [pending]) == forecasts
    assert not client.payloads
    pending.status = "accepted"
    updated = suite.update_forecasts(forecasts, [pending])
    assert updated[0].predicted_range == forecasts[0].baseline_range
    assert updated[0].supporting_card_ids == [pending.card_id]
    previous_calls = len(client.payloads)
    assert suite.update_forecasts(updated, [pending]) == updated
    assert len(client.payloads) == previous_calls


def test_forecast_rejects_change_without_cited_card():
    view = view_for()

    def responder(role, payload):
        raw = deterministic_responder(role, payload)
        if payload["op"] == "adaptive_forecast":
            raw["forecasts"][0]["predicted_range"] = 0.5
            raw["forecasts"][0]["supporting_card_ids"] = []
        return raw

    suite = ResearchAgentSuite(LocalClient(responder), view)
    forecasts = forecasts_for(view)
    card = card_for(view)
    card.status = "accepted"
    assert suite.update_forecasts(forecasts, [card]) == forecasts
    assert suite.forecast_issues[0]["code"] == "UNSUPPORTED_FORECAST_UPDATE"


def test_forecast_with_cited_card_is_clamped_to_relative_bound():
    view = view_for()

    def responder(role, payload):
        raw = deterministic_responder(role, payload)
        if payload["op"] == "adaptive_forecast":
            raw["forecasts"][0]["predicted_range"] = 0.5
        return raw

    suite = ResearchAgentSuite(
        LocalClient(responder), view, limits={"max_forecast_relative_change": 0.2}
    )
    forecasts = forecasts_for(view)
    card = card_for(view)
    card.status = "accepted"
    updated = suite.update_forecasts(forecasts, [card])
    assert updated[0].predicted_range == pytest.approx(0.03)
    assert suite.forecast_adjustments == [
        {
            "symbol": updated[0].symbol,
            "baseline_range": 0.025,
            "proposed_range": 0.5,
            "bounded_range": pytest.approx(0.03),
            "relative_change": pytest.approx(0.2),
            "was_clamped": True,
            "supporting_card_ids": [card.card_id],
        }
    ]


def test_memory_expiry_append_only_and_no_future_same_day_visibility(tmp_path):
    view = view_for()
    card = card_for(view)
    card.status = "accepted"
    memory = AcceptedCardMemory(tmp_path, "run_1", "scenario", CUTOFF)
    memory.append([card])
    frozen = memory.path.read_bytes()
    assert len(memory.active()) == 1
    assert memory.active(CUTOFF - timedelta(minutes=1)) == []
    next_day = AcceptedCardMemory(tmp_path, "run_2", "scenario", CUTOFF + timedelta(days=1))
    assert len(next_day.active()) == 1
    replacement = card_for(view, card_id="card_new")
    replacement.status = "accepted"
    next_day.supersede(replacement, [card.card_id])
    assert [item.card_id for item in next_day.active()] == ["card_new"]
    assert memory.path.read_bytes() == frozen
    assert AcceptedCardMemory(tmp_path, "other", "uniform", CUTOFF).active() == []
    expired = AcceptedCardMemory(tmp_path, "run_3", "scenario", CUTOFF + timedelta(days=9))
    assert expired.active() == []
    assert memory.path.read_bytes() == frozen


def test_offline_plan_research_and_forecast_protocol():
    view = view_for()
    suite = ResearchAgentSuite(LocalClient(), view)
    forecasts = forecasts_for(view)
    questions = suite.plan(forecasts, reserved_cost_usd=0.2)
    assert any(question.discovery for question in questions)
    assert all(question.reserved_cost_usd == 0.2 for question in questions)
    specific = next(question for question in questions if not question.discovery)
    result = suite.research(specific, forecasts)
    assert result.stop_reason == "accepted"
    updated = suite.update_forecasts(forecasts, result.accepted_cards)
    assert updated[0].baseline_range == updated[0].predicted_range
    visible = json.dumps([payload for _, payload, _ in suite.client.payloads])
    assert "AAPL" not in visible
    assert "apple.com" not in visible


def test_discovery_coverage_is_preserved_if_planner_returns_no_questions():
    view = view_for()

    def responder(role, payload):
        if payload["op"] == "adaptive_plan":
            return {"questions": []}
        return deterministic_responder(role, payload)

    suite = ResearchAgentSuite(LocalClient(responder), view)
    questions = suite.plan(
        forecasts_for(view),
        history=[{"stop_reason": "archive_insufficient"}],
        remaining_budget_usd=0.5,
    )
    assert {question.symbols[0] for question in questions} == set(view.reverse_aliases)
    assert all(question.discovery for question in questions)
    assert suite.client.payloads[0][1]["remaining_budget_usd"] == 0.5


def test_history_control_has_only_historical_market_context():
    view = view_for()
    suite = ResearchAgentSuite(LocalClient(), view)
    forecasts = forecasts_for(view)
    updated = suite.forecast_history(forecasts)
    assert all(item.predicted_range == item.baseline_range for item in updated)
    payload = suite.client.payloads[0][1]
    assert payload["op"] == "adaptive_history"
    assert "accepted_cards" not in payload
    assert "lead_index" not in payload
    assert "evidence_id" not in json.dumps(payload)


def test_compact_research_keeps_discovery_coverage_and_finishes_before_worker_limit():
    view = view_for()
    client = LocalClient()
    suite = ResearchAgentSuite(
        client, view, limits={"compact_research_prompts": 1, "max_worker_model_calls": 2}
    )
    forecasts = forecasts_for(view)
    questions = suite.plan(forecasts)
    assert {q.symbols[0] for q in questions if q.discovery} == set(view.reverse_aliases)
    assert "DO NOT generate discovery questions" in client.payloads[0][1]["instructions"]
    result = suite.research(next(q for q in questions if not q.discovery), forecasts)
    assert result.stop_reason == "accepted"
    workers = [p for _, p, _ in client.payloads if p["op"] == "adaptive_research"]
    assert len(workers) == 2
    assert not workers[0].get("final_action_required", False)
    assert workers[-1]["final_action_required"] is True
