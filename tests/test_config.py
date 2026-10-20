from datetime import date
from pathlib import Path

import pytest

from financial_strategy.config import (
    MVPConfig,
    load_evaluation_config,
    load_mvp_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_real_config_is_valid_and_weights_are_frozen() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    assert config.ranking.final_weights.macro == 0.15
    assert config.ranking.minimum_score == 0.35
    assert config.data.input_adjustment_policy == "split_adjusted"
    assert config.agents.mode == "live"


def test_evaluation_config_covers_ten_sessions_and_range_labels() -> None:
    config = load_evaluation_config(
        ROOT / "configs/evaluation_2026-10-01_2026-10-14.yaml"
    )

    assert len(config.evaluation.sessions) == 10
    assert config.evaluation.sessions[0] == date(2026, 10, 1)
    assert config.evaluation.sessions[-1] == date(2026, 10, 14)
    assert config.screening_config == Path("configs/screening_2026-10-14.yaml")
    assert config.evaluation.labels == Path(
        "data/2026-10-01_2026-10-14/evaluation/labels.parquet"
    )


def test_invalid_final_weights_are_rejected() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    raw = config.model_dump()
    raw["ranking"]["final_weights"]["macro"] = 0.25
    with pytest.raises(ValueError, match="sum to 1"):
        MVPConfig.model_validate(raw)


def test_out_of_range_event_factor_is_rejected() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    raw = config.model_dump()
    raw["ranking"]["company_event"]["confidence_factors"]["high"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        MVPConfig.model_validate(raw)


def test_real_config_uses_configured_openrouter_model() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    assert config.task.top_k == 10
    assert config.data.input_adjustment_policy == "split_adjusted"
    assert config.agents.mode == "live"
    assert config.agents.provider == "openrouter"
    assert config.agents.model == "openai/gpt-5.6-luna"


def test_openrouter_model_requires_provider_prefix() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    raw = config.model_dump()
    raw["agents"]["model"] = "gpt-5.6-luna"

    with pytest.raises(ValueError, match="provider/model"):
        MVPConfig.model_validate(raw)


def test_role_model_overrides_default_agent_model() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    roles = dict(config.agents.roles)
    roles["verifier"] = roles["verifier"].model_copy(
        update={"model": "anthropic/claude-sonnet"}
    )
    agents = config.agents.model_copy(update={"roles": roles})

    assert agents.model_for("synthesis") == "openai/gpt-5.6-luna"
    assert agents.model_for("verifier") == "anthropic/claude-sonnet"
