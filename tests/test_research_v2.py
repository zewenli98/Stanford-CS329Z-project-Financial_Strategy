from datetime import timedelta
from types import SimpleNamespace

import polars as pl
import pytest
from test_adaptive_research import CUTOFF, LocalClient, forecasts_for, question_for, view_for

from financial_strategy.adaptive.agents import ResearchAgentSuite, deterministic_responder
from financial_strategy.adaptive.models import WorkerAction
from financial_strategy.adaptive.research_v2 import history_baselines, select_candidates
from financial_strategy.adaptive.tools import OpaqueToolView
from financial_strategy.llm.structured import worker_response_schema


def test_v2_mask_preserves_common_words_and_masks_explicit_tickers():
    u = {
        "securities": [
            {"symbol": s, "company_name": n}
            for s, n in [
                ("ON", "ON Semiconductor"),
                ("LOW", "Lowe's Companies"),
                ("C", "Citigroup"),
            ]
        ]
    }
    v = OpaqueToolView(
        SimpleNamespace(company_evidence=[], macro_evidence=[]),
        SimpleNamespace(decision_cutoff=CUTOFF, target_session=CUTOFF.date()),
        u,
        {s: {} for s in ["ON", "LOW", "C"]},
        protocol_v2=True,
    )
    assert (
        v.redact_text("on a low range in Washington, D.C.") == "on a low range in Washington, D.C."
    )
    assert "$ON" not in v.redact_text("ON Semiconductor $ON $LOW $C")


def test_v2_read_skips_filing_headers_and_packet_includes_counterevidence():
    view = view_for()
    view.protocol_v2 = True
    source = view.lead_index()[0]["evidence_id"]
    view.documents[source] = (
        "XBRL HEADER " * 200 + "Item 2.02 Results of Operations. Revenue improved."
    )
    result = view.dispatch({"tool": "read", "arguments": {"evidence_id": source}})
    assert result["span"]["start"] > 0 and "Revenue improved" in result["span"]["quote"]
    view.documents[source] = (
        "EX-99.1 Revenue improved. "
        + "Other detail. " * 160
        + "Our financial outlook is unchanged."
    )
    observations = view.packet_observations(view.aliases["AAPL"])
    assert any("outlook is unchanged" in p["result"]["span"]["quote"] for p in observations)


def test_v2_reserves_last_call_for_actual_repair():
    view = view_for()
    view.protocol_v2 = True
    steps = 0

    def respond(role, payload):
        nonlocal steps
        raw = deterministic_responder(role, payload)
        if payload["op"] == "adaptive_research":
            steps += 1
            if steps == 2:
                return {
                    "tool": "read",
                    "arguments": {"evidence_id": view.lead_index()[0]["evidence_id"]},
                }
            if steps == 3:
                assert payload["final_action_required"]
                raw["cards"][0]["source_spans"][0]["quote"] = "incorrect"
            if steps == 4:
                assert "SOURCE_SPAN_MISMATCH" in payload["verification_issues"]
        return raw

    suite = ResearchAgentSuite(
        LocalClient(respond),
        view,
        limits={
            "research_protocol": "guided",
            "compact_research_prompts": 1,
            "max_worker_model_calls": 4,
        },
    )
    result = suite.research(question_for(view), forecasts_for(view))
    assert steps == 4 and result.stop_reason == "accepted"
    assert result.accepted_cards[0].version == 2


def test_hybrid_candidates_are_history_and_dated_evidence_only():
    scores = {"A": 0.1, "B": 0.08, "C": 0.06, "D": 0.01}
    records = [
        SimpleNamespace(
            symbol="D",
            available_at=CUTOFF - timedelta(hours=2),
            event_types=["earnings"],
            event_type="earnings",
        )
    ]
    assert select_candidates(scores, records, CUTOFF, 2, "volatility") == ["A", "B"]
    assert select_candidates(scores, records, CUTOFF, 2, "hybrid") == ["A", "D"]
    records[0].available_at=CUTOFF+timedelta(seconds=1)
    assert select_candidates(scores, records, CUTOFF, 2, "hybrid") == ["A", "B"]


def test_history_baseline_uses_supplied_completed_history():
    bars = pl.DataFrame(
        [{"symbol": "A", "session": i, "low": 100.0, "high": 100.0 + i} for i in range(1, 22)]
    )
    assert history_baselines({"A": None}, bars, "mean5")["A"] == pytest.approx(0.19)
    assert history_baselines({"A": None}, bars, "mean10")["A"] == pytest.approx(0.165)


def test_strict_tool_scope_cannot_be_permitted_archive():
    wire = worker_response_schema(WorkerAction, strict_tools=True)
    props = wire["schema"]["properties"]["arguments"]["properties"]
    assert props["scope"]["enum"] == ["company", "macro", None]
    assert props["top_n"]["maximum"] == 20
