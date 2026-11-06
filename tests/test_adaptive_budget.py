"""Dollar caps remain conservative across concurrency, failures, retries, and replay."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest
from pydantic import BaseModel, Field, ValidationError, create_model

from financial_strategy.adaptive.budget import AdaptiveClient, BudgetLedger, MeteredClient
from financial_strategy.adaptive.config import AdaptiveConfig
from financial_strategy.config import load_mvp_config
from financial_strategy.errors import StrategyError
from financial_strategy.llm.client import InvocationResult, RecordedClient, request_hash

ROOT = Path(__file__).resolve().parents[1]


class TinyResponse(BaseModel):
    answer: str


def adaptive_config(mode: str = "scripted") -> AdaptiveConfig:
    data = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml").model_dump(mode="json")
    data["workflow"] = "adaptive_range_v1"
    data["agents"]["mode"] = mode
    for role in data["agents"]["roles"].values():
        role["max_tokens"] = 32
    data["retry"]["backoff_seconds"] = [0, 0]
    models = {data["agents"]["model"]}
    models.update(role["model"] for role in data["agents"]["roles"].values() if role["model"])
    price = {"input_per_million": 1, "output_per_million": 4, "request_usd": 0.01}
    data["adaptive"] = {
        "max_input_bytes": 4000,
        "offline_price": price,
        "prices": dict.fromkeys(models, price),
        "pricing_as_of": "2026-11-04",
    }
    return AdaptiveConfig.model_validate(data)


def response(source: str = "live", *, input_tokens: int = 100, output_tokens: int = 20):
    parsed = TinyResponse(answer="supported")
    return InvocationResult(
        response=parsed,
        request_hash="recorded_request",
        source=source,
        latency_seconds=0.001,
        raw=parsed.model_dump(),
        role="company_evidence",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class SequenceClient:
    """A local provider double: no credentials, model requests, or network access."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def generate(self, role, payload, schema, *, attempt_type="initial"):
        self.calls.append({"role": role, "payload": payload, "attempt_type": attempt_type})
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def known_charge(config: AdaptiveConfig) -> float:
    price = config.adaptive.offline_price
    return (100 * price.input_per_million + 20 * price.output_per_million) / 1e6 + price.request_usd


def test_atomic_reservations_cannot_consume_the_finalization_hold() -> None:
    ledger = BudgetLedger(2)
    finalization = ledger.reserve(0.5, label="finalization")
    barrier = Barrier(12)

    def reserve_worker(index: int) -> bool:
        barrier.wait(timeout=10)
        try:
            ledger.reserve(0.25, label=f"worker_{index}")
        except StrategyError as exc:
            assert exc.code == "TASK_DOLLAR_BUDGET_EXHAUSTED"
            return False
        return True

    with ThreadPoolExecutor(max_workers=12) as workers:
        successes = list(workers.map(reserve_worker, range(12)))
    assert sum(successes) == 6
    assert ledger.holds[finalization] == 0.5
    assert ledger.summary()["reserved_usd"] == pytest.approx(2)
    assert ledger.remaining == 0


def test_child_settlement_refunds_parent_without_releasing_its_budget() -> None:
    ledger = BudgetLedger(2)
    finalization = ledger.reserve(0.4, label="finalization")
    assignment = ledger.reserve(0.8, label="assignment")
    child = ledger.reserve(0.3, parent=assignment, label="attempt")
    assert ledger.holds[assignment] == pytest.approx(0.5)
    assert ledger.remaining == pytest.approx(0.8)
    ledger.settle(child, 0.1, 0.07, parent=assignment)
    assert ledger.holds[assignment] == pytest.approx(0.7)
    assert ledger.remaining == pytest.approx(0.8)
    ledger.release(assignment)
    assert ledger.remaining == pytest.approx(1.5)
    ledger.settle(finalization, 0.2, 0.12)
    assert ledger.summary()["normalized_uncached_usd"] == pytest.approx(0.3)
    assert ledger.summary()["actual_usd"] == pytest.approx(0.19)
    assert ledger.remaining == pytest.approx(1.7)


@pytest.mark.parametrize("cap", [-0.1, float("nan"), float("inf")])
def test_invalid_ledger_caps_fail_closed(cap: float) -> None:
    with pytest.raises((ValueError, StrategyError)):
        BudgetLedger(cap)


@pytest.mark.parametrize("amount", [-0.1, float("nan"), float("inf")])
def test_invalid_reservations_fail_before_affecting_holds(amount: float) -> None:
    ledger = BudgetLedger(1)
    with pytest.raises((ValueError, StrategyError)):
        ledger.reserve(amount)
    assert not ledger.holds
    assert ledger.remaining == 1


@pytest.mark.parametrize("normalized", [-0.1, float("nan"), float("inf")])
def test_invalid_settlement_cannot_poison_ledger(normalized: float) -> None:
    ledger = BudgetLedger(1)
    token = ledger.reserve(0.2)
    with pytest.raises((ValueError, StrategyError)):
        ledger.settle(token, normalized)
    assert ledger.summary()["normalized_uncached_usd"] >= 0
    assert ledger.summary()["normalized_uncached_usd"] <= 1


def test_cost_bound_violation_blocks_every_subsequent_reservation() -> None:
    ledger = BudgetLedger(1)
    token = ledger.reserve(0.2)
    with pytest.raises(StrategyError, match="provider usage"):
        ledger.settle(token, 0.3, 0.3)
    assert ledger.remaining == 0
    with pytest.raises(StrategyError):
        ledger.reserve(0.01)


@pytest.mark.parametrize("actual", [-0.1, float("nan"), float("inf")])
def test_actual_cost_must_be_finite_and_nonnegative(actual: float) -> None:
    ledger = BudgetLedger(1)
    token = ledger.reserve(0.2)
    with pytest.raises((ValueError, StrategyError)):
        ledger.settle(token, 0.1, actual)
    assert 0 <= ledger.summary()["actual_usd"] <= 1


def test_live_transport_failure_charges_full_bound_and_records_estimation() -> None:
    config = adaptive_config("live")
    ledger = BudgetLedger(2)
    meter = MeteredClient(SequenceClient([TimeoutError("transport failed")]), config, ledger)
    bound = meter.bound("company_evidence")
    parent = ledger.reserve(2 * bound, label="assignment")
    with meter.using(parent), pytest.raises(TimeoutError):
        meter.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    assert ledger.spent == pytest.approx(bound)
    assert ledger.actual == pytest.approx(bound)
    assert ledger.holds[parent] == pytest.approx(bound)
    record = meter.records[0]
    assert record["status"] == "failed"
    assert record["normalized_usd"] == pytest.approx(bound)
    assert record["actual_usd"] == pytest.approx(bound)
    assert record["usage_estimated"] is True


@pytest.mark.parametrize("known_usage", [False, True])
def test_live_cache_hit_is_normalized_nonfree_but_has_no_actual_charge(known_usage: bool) -> None:
    config = adaptive_config("live")
    result = response(
        "cache", input_tokens=100 if known_usage else 0, output_tokens=20 if known_usage else 0
    )
    ledger = BudgetLedger(2)
    meter = MeteredClient(SequenceClient([result]), config, ledger)
    meter.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    expected = known_charge(config) if known_usage else meter.bound("company_evidence")
    assert ledger.spent == pytest.approx(expected)
    assert ledger.spent > 0
    assert ledger.actual == 0
    assert meter.records[0]["usage_estimated"] is not known_usage


@pytest.mark.parametrize("known_usage", [False, True])
def test_recorded_response_uses_known_cost_or_explicit_estimate(known_usage: bool) -> None:
    config = adaptive_config("recorded")
    result = response(
        "recorded", input_tokens=100 if known_usage else 0, output_tokens=20 if known_usage else 0
    )
    ledger = BudgetLedger(2)
    meter = MeteredClient(SequenceClient([result]), config, ledger)
    meter.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    if known_usage:
        assert ledger.spent == pytest.approx(known_charge(config))
    else:
        assert 0 < ledger.spent <= meter.bound("company_evidence")
    assert ledger.actual == 0
    assert meter.records[0]["usage_estimated"] is not known_usage


@pytest.mark.parametrize("known_usage", [False, True])
def test_recorded_file_retains_optional_token_usage(tmp_path: Path, known_usage: bool) -> None:
    config = adaptive_config("recorded")
    config.agents.recordings_path = tmp_path
    role = "company_evidence"
    payload = {"symbol": "company_1"}
    role_config = config.agents.roles[role]
    key = request_hash(
        role,
        payload,
        TinyResponse,
        config.agents.model_for(role),
        role_config.prompt_version,
        "initial",
        role_config.max_tokens,
    )
    recording = {"response": {"answer": "supported"}}
    if known_usage:
        recording.update(input_tokens=100, output_tokens=20)
    (tmp_path / f"{key}.json").write_text(json.dumps(recording))
    ledger = BudgetLedger(2)
    meter = MeteredClient(RecordedClient(config.agents), config, ledger)
    result = meter.generate(role, payload, TinyResponse)
    assert result.input_tokens == (100 if known_usage else 0)
    assert result.output_tokens == (20 if known_usage else 0)
    assert meter.records[0]["usage_estimated"] is not known_usage
    if known_usage:
        assert ledger.spent == pytest.approx(known_charge(config))
    assert ledger.actual == 0


def test_retry_attempts_are_each_metered_inside_controlled_client() -> None:
    config = adaptive_config("live")
    provider = SequenceClient([TimeoutError("first request failed"), response()])
    ledger = BudgetLedger(2)
    client = AdaptiveClient(config, None, ledger, base=provider)
    result = client.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    assert result.raw["answer"] == "supported"
    assert len(client.attempt_log) == 2
    assert [item["status"] for item in client.meter.records] == ["failed", "completed"]
    assert [item["attempt_type"] for item in provider.calls] == ["initial", "technical_retry_1"]
    expected = client.meter.bound("company_evidence") + known_charge(config)
    assert ledger.spent == pytest.approx(expected)
    assert ledger.actual == pytest.approx(ledger.spent)
    assert all(call["payload"]["_workflow"] == "adaptive_range_v1" for call in provider.calls)


def test_assignment_limit_counts_every_provider_attempt() -> None:
    config = adaptive_config("live")
    provider = SequenceClient([response(), response(), response()])
    ledger = BudgetLedger(2)
    client = AdaptiveClient(config, None, ledger, base=provider)
    hold = ledger.reserve(1, label="worker")
    with client.using(hold):
        for _ in range(3):
            client.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
        with pytest.raises(StrategyError, match="assignment attempt limit"):
            client.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    assert len(provider.calls) == 3
    assert len(client.meter.records) == 3
    assert ledger.spent == pytest.approx(3 * known_charge(config))


def test_structured_repair_has_a_separate_reservation_and_charge() -> None:
    config = adaptive_config("live")
    try:
        TinyResponse.model_validate({"answer": None})
    except ValidationError as exc:
        validation_error = exc
    provider = SequenceClient([validation_error, response()])
    ledger = BudgetLedger(2)
    client = AdaptiveClient(config, None, ledger, base=provider)
    client.generate("company_evidence", {"symbol": "company_1"}, TinyResponse)
    assert len(client.meter.records) == 2
    assert provider.calls[1]["attempt_type"] == "structured_repair"
    assert provider.calls[1]["payload"]["_repair"]["attempt"] == 1
    expected = client.meter.bound("company_evidence") + known_charge(config)
    assert ledger.spent == pytest.approx(expected)


def test_payload_and_schema_limits_reject_before_provider_or_budget_reservation() -> None:
    config = adaptive_config()
    provider = SequenceClient([])
    ledger = BudgetLedger(2)
    meter = MeteredClient(provider, config, ledger)
    oversized_schema = create_model(
        "OversizedResponse",
        **{f"field_{index}": (str, Field(description="x" * 1000)) for index in range(30)},
    )
    for payload, schema in (({"blob": "x" * 4100}, TinyResponse), ({}, oversized_schema)):
        with pytest.raises(StrategyError) as caught:
            meter.generate("company_evidence", payload, schema)
        assert caught.value.code == "INPUT_LIMIT_EXCEEDED"
    assert provider.calls == []
    assert ledger.events == []
    assert ledger.spent == 0


@pytest.mark.parametrize("missing", ["prices", "date", "role_override"])
def test_live_prices_must_be_dated_and_cover_every_exact_role_model(missing: str) -> None:
    data = adaptive_config("live").model_dump(mode="json")
    if missing == "prices":
        data["adaptive"]["prices"] = {}
    elif missing == "date":
        data["adaptive"]["pricing_as_of"] = None
    else:
        data["agents"]["roles"]["verifier"]["model"] = "different/exact-model"
    with pytest.raises(ValidationError, match="dated prices"):
        AdaptiveConfig.model_validate(data)
