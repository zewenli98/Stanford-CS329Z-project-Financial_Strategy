from datetime import date
from pathlib import Path

import pytest

from financial_strategy.config import (
    EvaluationConfig,
    MVPConfig,
    load_evaluation_config,
    load_mvp_config,
    load_tuning_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_real_config_is_valid_and_weights_are_frozen() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    assert config.ranking.final_weights.company_event == 0.0
    assert config.ranking.final_weights.macro == 0.0
    assert (
        config.ranking.final_weights.volatility + config.ranking.final_weights.anomaly
    ) == pytest.approx(1.0)
    assert config.ranking.minimum_score == 0.0
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
        "tests/fixtures/integration/evaluation/labels.parquet"
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


def test_research100_uses_only_the_frozen_feature_ranker() -> None:
    config = load_mvp_config(ROOT / "configs/screening_research100_candidate.yaml")
    assert config.ranking.market_feature_weights is not None
    assert abs(sum(config.ranking.market_feature_weights.values()) - 1.0) < 1e-9
    assert config.ranking.volatility_weights is None
    assert config.ranking.anomaly_weights is None
    assert config.ranking.residual_model.enabled is False
    assert config.ranking.final_weights.company_event == 0.0
    assert config.ranking.final_weights.macro == 0.0
    assert config.agents.roles["synthesis"].prompt_version == "synthesis_v2"


def test_feature_weights_and_component_weights_cannot_both_be_set() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    raw = config.model_dump()
    raw["ranking"]["market_feature_weights"] = {"median_range_20d": 1.0}
    with pytest.raises(ValueError, match="only market score"):
        MVPConfig.model_validate(raw)


def test_evaluation_relevance_must_be_the_range_label() -> None:
    with pytest.raises(ValueError, match="intraday_range"):
        EvaluationConfig.model_validate(
            {
                "screening_config": "x.yaml",
                "evaluation": {
                    "sessions": ["2026-01-01"],
                    "cutoff_time": "09:00:00",
                    "timezone": "UTC",
                    "labels": "labels.parquet",
                    "large_move_threshold": 0.04,
                    "variants": ["full_system"],
                    "ndcg": {"relevance": "open_to_close"},
                },
            }
        )


def test_synthesis_instruction_names_the_range_target() -> None:
    from financial_strategy.agents import scripted_responder
    from financial_strategy.llm.client import SYNTHESIS_TARGET_INSTRUCTION

    assert "(high - low) / low" in SYNTHESIS_TARGET_INSTRUCTION
    assert "open-to-close" not in SYNTHESIS_TARGET_INSTRUCTION
    rationale = " ".join(
        scripted_responder("synthesis", {"company_claims": []})["rationale"]
    )
    assert "(high - low) / low" in rationale
    assert "open-to-close" not in rationale


def test_tuning_config_freezes_search_and_holdout_paths() -> None:
    config = load_tuning_config(ROOT / "configs/tuning.yaml")
    assert config.search.seed == 329
    assert config.search.trials == 750
    assert config.search.candidate_counts == [10, 15, 20]
    assert config.large_move_threshold == 0.04
