from financial_strategy.adaptive.models import WorkerAction
from financial_strategy.llm.client import request_hash
from financial_strategy.llm.structured import worker_response_schema


def test_company_wire_closes_nested_events_metrics_and_quotes():
    from financial_strategy.agents.roles import CompanyAgentOutput
    from financial_strategy.llm.structured import closed_response_schema

    wire = closed_response_schema(CompanyAgentOutput)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            assert "default" not in node
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    check(wire["schema"])
    assert wire["strict"] is True
    assert set(wire["schema"]["properties"]) == {"events", "claims", "metrics"}
    assert wire["schema"]["$defs"]["EvidenceCitation"]["properties"]["quote"]["maxLength"] == 800


def test_company_wire_version_changes_request_identity(monkeypatch):
    from financial_strategy.agents.roles import CompanyAgentOutput

    before = request_hash(
        "company_evidence", {}, CompanyAgentOutput, "model", "v1", "initial", 2048
    )
    monkeypatch.setattr(
        "financial_strategy.llm.client.COMPANY_WIRE_VERSION", "next_company_protocol"
    )
    after = request_hash("company_evidence", {}, CompanyAgentOutput, "model", "v1", "initial", 2048)
    assert before != after


def test_single_action_wire_schema_is_closed_and_nullable():
    wire = worker_response_schema(WorkerAction)
    assert wire["strict"] is True
    args = wire["schema"]["properties"]["arguments"]
    assert args["additionalProperties"] is False
    assert set(args["required"]) == set(args["properties"])
    assert args["properties"]["evidence_id"]["type"] == ["string", "null"]
    assert "evidence_ids" not in args["properties"]

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node.get("properties", {}))
            assert "default" not in node
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    check(wire["schema"])


def test_wire_null_arguments_preserve_defaults_and_zero_offsets():
    action = WorkerAction(
        tool="read",
        arguments={
            "evidence_id": "source_1",
            "start": 0,
            "end": None,
            "query": None,
            "symbol": None,
            "scope": None,
            "top_n": None,
        },
    )
    assert action.arguments == {"evidence_id": "source_1", "start": 0}


def test_wire_protocol_is_part_of_adaptive_request_identity(monkeypatch):
    payload = {"op": "adaptive_research", "_workflow": "adaptive_range_v1"}
    old = request_hash("company_evidence", payload, WorkerAction, "model", "v1", "initial", 2048)
    monkeypatch.setattr("financial_strategy.llm.client.WORKER_WIRE_VERSION", "next_protocol")
    new = request_hash(
        "company_evidence",
        payload,
        WorkerAction,
        "model",
        "v1",
        "initial",
        2048,
    )
    assert old != new


def test_grounded_span_schema_only_admits_already_read_exact_citations():
    span = {"evidence_id": "source_1", "start": 0, "end": 11, "quote": "Exact quote"}
    wire = worker_response_schema(WorkerAction, [span])
    branches = wire["schema"]["$defs"]["SourceSpan"]["anyOf"]
    assert len(branches) == 1
    for key, value in span.items():
        assert branches[0]["properties"][key]["enum"] == [value]


def test_final_action_wire_schema_cannot_spend_last_attempt_on_another_read():
    normal = worker_response_schema(WorkerAction)
    final = worker_response_schema(WorkerAction, final_action=True)
    assert "read" in normal["schema"]["properties"]["tool"]["enum"]
    assert final["schema"]["properties"]["tool"]["enum"] == ["finish", "abstain"]
