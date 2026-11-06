from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from financial_strategy.agents.roles import AgentSuite, scripted_responder
from financial_strategy.config import load_mvp_config
from financial_strategy.evaluation.retrieval_accuracy import (
    evaluate_retrieval_labels,
    load_retrieval_labels,
    retrieval_accuracy,
)
from financial_strategy.llm import make_client
from financial_strategy.retrieval.corrective import (
    CURRENT_RELEASE_NOT_IN_SNAPSHOT,
    REWRITE_QUERY,
    RetrievalJudgment,
    coerce_judgment,
    dedupe_evidence,
)
from financial_strategy.schemas import EvidenceRecord, ScreeningTask

ROOT = Path(__file__).resolve().parents[1]
CUTOFF = datetime(2026, 10, 14, 13, tzinfo=UTC)


def _evidence(evidence_id: str, **overrides) -> EvidenceRecord:
    timestamp = datetime(2026, 10, 10, tzinfo=UTC)
    values = {
        "evidence_id": evidence_id,
        "scope": "company",
        "symbol": "AAPL",
        "source_type": "filing",
        "source_name": "sec",
        "title": "Third-quarter earnings",
        "excerpt": "The company updated guidance.",
        "published_at": timestamp,
        "first_seen_at": timestamp,
        "retrieved_at": timestamp,
        "available_at": timestamp,
        "content_hash": evidence_id,
        "snapshot_id": "snapshot",
        "timestamp_confidence": "high",
        "event_type": "earnings",
    }
    values.update(overrides)
    return EvidenceRecord(**values)


def _task() -> ScreeningTask:
    return ScreeningTask(
        task_id="task",
        universe_snapshot_id="universe",
        decision_cutoff=CUTOFF,
        target_session=date(2026, 10, 14),
        top_k=10,
        price_lookback_sessions=60,
        news_lookback_days=7,
        max_revision_rounds=2,
    )


def _suite(**agent_updates):
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    agents = config.agents.model_copy(
        update={"mode": "scripted", "model": "scripted", **agent_updates}
    )
    config = config.model_copy(update={"agents": agents})
    return AgentSuite(config, make_client(agents, scripted_responder, config.retry))


def test_current_release_is_accepted_and_prior_quarter_is_not_written() -> None:
    prior = _evidence(
        "ev_prior",
        title="Second-quarter earnings",
        excerpt="Prior quarter results and guidance.",
        published_at=CUTOFF - timedelta(days=90),
        first_seen_at=CUTOFF - timedelta(days=90),
        retrieved_at=CUTOFF - timedelta(days=89),
        available_at=CUTOFF - timedelta(days=90),
    )
    current = _evidence("ev_current")
    output, invocations, traces = _suite().company(_task(), "AAPL", [prior, current], "snapshot")

    assert [event.evidence_ids for event in output.events] == [["ev_current"]]
    assert traces[0]["judgment"]["action"] == "accept"
    assert traces[0]["judgment"]["accepted_evidence_ids"] == ["ev_current"]
    assert [item.attempt_type for item in invocations] == [
        "company_question",
        "retrieval_judgment",
        "initial",
    ]


def test_current_buyback_filing_is_accepted() -> None:
    filing = _evidence(
        "ev_buyback",
        title="Issuer repurchase 8-K",
        excerpt="The board authorized a share buyback.",
        event_type="buyback",
        event_types=["buyback"],
    )
    output, _invocations, traces = _suite().company(_task(), "AAPL", [filing], "snapshot")

    assert traces[0]["judgment"]["action"] == "accept"
    assert traces[0]["judgment"]["accepted_evidence_ids"] == ["ev_buyback"]
    assert output.events


def test_missing_filing_does_not_call_the_model() -> None:
    output, invocations, traces = _suite().company(_task(), "AAPL", [], "snapshot")

    assert invocations == []
    assert traces == []
    assert output.events == []
    assert CURRENT_RELEASE_NOT_IN_SNAPSHOT in output.claims[0].text


def test_prior_quarter_only_rewrites_once_then_records_the_gap() -> None:
    prior = _evidence(
        "ev_prior",
        title="Second-quarter earnings",
        excerpt="Prior quarter results and guidance.",
        published_at=CUTOFF - timedelta(days=90),
        first_seen_at=CUTOFF - timedelta(days=90),
        retrieved_at=CUTOFF - timedelta(days=89),
        available_at=CUTOFF - timedelta(days=90),
    )
    output, _invocations, traces = _suite().company(_task(), "AAPL", [prior], "snapshot")

    assert output.events == []
    assert output.claims[0].evidence_ids == []
    assert CURRENT_RELEASE_NOT_IN_SNAPSHOT in output.claims[0].text
    assert [trace["judgment"]["action"] for trace in traces] == ["rewrite", "record_gap"]
    assert traces[1]["query"] == REWRITE_QUERY


def test_reprints_of_one_filing_do_not_consume_a_second_hit() -> None:
    filing = _evidence("ev_filing", accession_number="0001", event_time=CUTOFF - timedelta(days=1))
    reprint = _evidence(
        "ev_reprint",
        accession_number="0001",
        content_hash="reprint",
        title="Wire reprint of earnings",
        event_time=None,
    )
    assert [item.evidence_id for item in dedupe_evidence([reprint, filing])] == ["ev_filing"]

    output, _invocations, traces = _suite().company(_task(), "AAPL", [reprint, filing], "snapshot")
    assert traces[0]["deduped_count"] == 1
    assert traces[0]["returned_evidence_ids"] == ["ev_filing"]
    assert [event.evidence_ids for event in output.events] == [["ev_filing"]]


def test_second_rewrite_is_recorded_as_a_gap() -> None:
    raw = RetrievalJudgment(
        company_match=True,
        event_match=False,
        sufficient=False,
        action="rewrite",
        gap=CURRENT_RELEASE_NOT_IN_SNAPSHOT,
        rewritten_query="current earnings guidance 8-K",
    )
    judgment = coerce_judgment(raw, {"ev_prior"}, allow_rewrite=False)
    assert judgment.action == "record_gap"
    assert judgment.gap == CURRENT_RELEASE_NOT_IN_SNAPSHOT


def test_exhausted_company_search_budget_records_a_gap() -> None:
    prior = _evidence(
        "ev_prior",
        title="Second-quarter earnings",
        published_at=CUTOFF - timedelta(days=90),
        first_seen_at=CUTOFF - timedelta(days=90),
        retrieved_at=CUTOFF - timedelta(days=89),
        available_at=CUTOFF - timedelta(days=90),
    )
    output, _invocations, traces = _suite(company_max_tool_calls_per_symbol=1).company(
        _task(), "AAPL", [prior], "snapshot"
    )
    assert output.events == []
    assert traces[0]["judgment"]["action"] == "rewrite"
    assert traces[1]["budget_error"] == "SYMBOL_BUDGET_EXHAUSTED"
    assert traces[1]["judgment"]["action"] == "record_gap"


def test_repeated_company_call_does_not_retrieve_again() -> None:
    current = _evidence("ev_current")
    suite = _suite()
    _first, first_calls, first_traces = suite.company(_task(), "AAPL", [current], "snapshot")
    _second, second_calls, second_traces = suite.company(_task(), "AAPL", [current], "snapshot")
    assert first_calls
    assert first_traces
    assert second_calls == []
    assert second_traces == []


def test_retrieval_accuracy_keeps_missing_judgments_out_of_the_rate() -> None:
    gold = {
        "company_match": True,
        "event_match": True,
        "sufficient": True,
        "action": "accept",
    }
    report = retrieval_accuracy([(gold, {**gold, "action": "record_gap"}), (gold, None)])
    assert report == {
        "count": 2,
        "matched": 1,
        "missing": 1,
        "company_match_accuracy": 1,
        "event_match_accuracy": 1,
        "sufficient_accuracy": 1,
        "action_accuracy": 0,
    }


def test_labeled_fixture_scores_the_scripted_judge_and_recorded_traces() -> None:
    labels = load_retrieval_labels(ROOT / "configs/retrieval_accuracy_labels.json")
    scripted = evaluate_retrieval_labels(labels)
    assert scripted["matched"] == len(labels) == 4
    assert scripted["missing"] == 0
    assert scripted["action_accuracy"] == 1
    assert scripted["company_match_accuracy"] == 1
    assert scripted["event_match_accuracy"] == 1
    assert scripted["sufficient_accuracy"] == 1

    traces = [
        {
            "scope": "company",
            "symbol": label["symbol"],
            "question": label["question"],
            "attempt": label["attempt"],
            "returned_evidence_ids": [item["evidence_id"] for item in label["evidence"]],
            "judgment": {
                "company_match": label["company_match"],
                "event_match": label["event_match"],
                "sufficient": label["sufficient"],
                "action": "rewrite" if label["case_id"] == "wrong_company" else label["action"],
            },
        }
        for label in labels
    ]
    recorded = evaluate_retrieval_labels(labels, traces)
    assert recorded["action_accuracy"] == 0.75
    assert recorded["company_match_accuracy"] == 1
    missing = evaluate_retrieval_labels(labels, [])
    assert missing["matched"] == 0
    assert missing["missing"] == 4
    assert missing["action_accuracy"] is None
