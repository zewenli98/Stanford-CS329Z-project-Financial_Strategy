from pathlib import Path

import pytest

from financial_strategy.config import load_mvp_config
from financial_strategy.errors import StrategyError
from financial_strategy.llm.client import (
    ControlledClient,
    InvocationResult,
    LiveClient,
)
from financial_strategy.schemas import VerificationReport

ROOT = Path(__file__).resolve().parents[1]


class CountingClient:
    def __init__(self, fail_once: bool = False) -> None:
        self.calls = 0
        self.fail_once = fail_once

    def generate(self, role, payload, schema, *, attempt_type="initial"):
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise TimeoutError("transient")
        raw = {"verdict": "PASS", "issues": [], "attempt": 0}
        return InvocationResult(
            schema.model_validate(raw),
            f"hash-{self.calls}",
            "test",
            0.0,
            raw,
            attempt_type,
        )


def test_live_transient_error_retries_without_business_revision(monkeypatch) -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    live = config.agents.model_copy(update={"mode": "live"})
    base = CountingClient(fail_once=True)
    monkeypatch.setattr("financial_strategy.llm.client.time.sleep", lambda _seconds: None)
    client = ControlledClient(base, live, config.retry)
    result = client.generate(
        "verifier",
        {"symbol": "AAPL"},
        VerificationReport,
    )
    assert result.response.verdict == "PASS"
    assert base.calls == 2
    assert client.attempt_log[1]["attempt_type"] == "technical_retry_1"


def test_symbol_model_budget_is_hard_limit() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    limited = config.agents.model_copy(update={"max_model_calls_per_symbol": 1})
    client = ControlledClient(CountingClient(), limited, config.retry)
    client.generate("verifier", {"symbol": "AAPL"}, VerificationReport)
    with pytest.raises(StrategyError) as error:
        client.generate("verifier", {"symbol": "AAPL"}, VerificationReport)
    assert error.value.code == "SYMBOL_BUDGET_EXHAUSTED"


def test_openrouter_requires_key(monkeypatch, tmp_path: Path) -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    live = config.agents.model_copy(
        update={"mode": "live", "provider": "openrouter", "cache_path": tmp_path}
    )
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(StrategyError) as error:
        LiveClient(live)
    assert error.value.code == "MISSING_LLM_API_KEY"


def test_openrouter_uses_structured_output_without_leaking_key(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict = {}

    class Message:
        usage_metadata = {"input_tokens": 10, "output_tokens": 5}

    class FakeModel:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def with_structured_output(self, schema, **kwargs):
            captured["schema"] = schema
            captured["structured_kwargs"] = kwargs
            return self

        def invoke(self, _messages):
            return {
                "parsed": VerificationReport(verdict="PASS", issues=[], attempt=0),
                "raw": Message(),
                "parsing_error": None,
            }

    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    roles = dict(config.agents.roles)
    roles["verifier"] = roles["verifier"].model_copy(
        update={"model": "openai/gpt-5.6-luna"}
    )
    live = config.agents.model_copy(
        update={
            "mode": "live",
            "provider": "openrouter",
            "model": "openai/default-model",
            "cache_path": tmp_path,
            "roles": roles,
        }
    )
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-secret")
    monkeypatch.setattr("financial_strategy.llm.client.ChatOpenAI", FakeModel)
    result = LiveClient(live).generate(
        "verifier", {"symbol": "AAPL"}, VerificationReport
    )
    assert result.response.verdict == "PASS"
    assert captured["kwargs"]["base_url"] == "https://openrouter.ai/api/v1"
    assert captured["kwargs"]["model"] == "openai/gpt-5.6-luna"
    assert captured["kwargs"]["extra_body"]["provider"]["require_parameters"] is True
    assert captured["structured_kwargs"] == {
        "method": "json_schema",
        "include_raw": True,
    }
    assert "test-secret" not in result.raw
