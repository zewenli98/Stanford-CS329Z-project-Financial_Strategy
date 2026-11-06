"""Acquisition mathematics, exploration probabilities, and batch safety."""

import math

import pytest
from pydantic import ValidationError

from financial_strategy.adaptive.allocation import (
    allocate_batch,
    allocate_question,
    scenario_value,
)
from financial_strategy.adaptive.models import AnswerScenario, RangeForecast, ResearchQuestion


def question(
    question_id: str = "q1",
    *,
    symbol: str = "B",
    shift: float = 1.0,
    discovery: bool = False,
    event_ids: list[str] | None = None,
) -> ResearchQuestion:
    return ResearchQuestion(
        question_id=question_id,
        text="Is the available disclosure a new regular-session catalyst?",
        symbols=[symbol],
        evidence_ids=["evidence_1"],
        event_ids=event_ids or [],
        outcomes=[]
        if discovery
        else [
            AnswerScenario(label="new", probability=0.5, log_shifts={symbol: shift}),
            AnswerScenario(label="old", probability=0.5, log_shifts={symbol: -shift}),
        ],
        resolution_probability=0.8,
        discovery=discovery,
        allowed_tools=["search", "read"],
        reserved_cost_usd=0.1,
    )


def test_scenario_value_matches_hand_calculation_and_preserves_raw_mean() -> None:
    # Clipping leaves +/-1 unchanged; shrinkage makes the applied shifts +/-0.5.
    # Z=(exp(.5)+exp(-.5))/2; B_new=.02*exp(.5)/Z; B_old=.02*exp(-.5)/Z.
    # Only the new answer changes the selection from A to B.
    value = scenario_value(question(), {"A": 0.025, "B": 0.02}, top_k=1)
    normalizer = (math.exp(0.5) + math.exp(-0.5)) / 2
    high = 0.02 * math.exp(0.5) / normalizer
    low = 0.02 * math.exp(-0.5) / normalizer
    expected_value = 0.5 * (high - 0.025)
    assert value.scenario_means["new"]["B"] == pytest.approx(high)
    assert value.scenario_means["old"]["B"] == pytest.approx(low)
    assert 0.5 * high + 0.5 * low == pytest.approx(0.02)
    assert value.conditional_value == pytest.approx(expected_value)
    assert value.priority == pytest.approx(0.8 * expected_value / 0.1)
    assert value.baseline_selection == ["A"]
    assert value.scenario_means["new"]["A"] == 0.025


def test_clipping_shrinkage_and_zero_means_are_auditable() -> None:
    value = scenario_value(question(shift=100), {"A": 0.025, "B": 0}, top_k=1)
    assert value.scenario_means["new"]["B"] == 0
    assert value.scenario_means["old"]["B"] == 0
    assert value.priority == 0
    assert value.transformations[0]["proposed_log_shift"] == 100
    assert value.transformations[0]["clipped_log_shift"] == 1.5
    assert value.transformations[0]["applied_log_shift"] == 0.75
    assert value.transformations[0]["normalizer"] > 0


def test_information_without_selection_change_has_zero_value() -> None:
    assert scenario_value(question(), {"A": 0.2, "B": 0.02}, top_k=1).priority == 0
    # Selecting the whole universe leaves no alternative set in any scenario.
    assert scenario_value(question(), {"A": 0.025, "B": 0.02}, top_k=2).priority == 0
    unresolved = question().model_copy(update={"resolution_probability": 0})
    assert scenario_value(unresolved, {"A": 0.025, "B": 0.02}, top_k=1).priority == 0


def test_unequal_probabilities_preserve_each_affected_mean() -> None:
    item = ResearchQuestion(
        question_id="joint",
        text="Which eligible event interpretation is supported?",
        symbols=["A", "B"],
        outcomes=[
            AnswerScenario(label="one", probability=0.1, log_shifts={"A": 10, "B": -2}),
            AnswerScenario(label="two", probability=0.3, log_shifts={"A": -0.2}),
            AnswerScenario(label="three", probability=0.6, log_shifts={"B": 0.7}),
        ],
        resolution_probability=1,
        reserved_cost_usd=0.1,
    )
    means = {"A": 0.025, "B": 0.02, "C": 0.03}
    value = scenario_value(item, means, top_k=2)
    for symbol, mean in means.items():
        average = sum(
            outcome.probability * value.scenario_means[outcome.label][symbol]
            for outcome in item.outcomes
        )
        assert average == pytest.approx(mean)
    assert value.conditional_value >= 0


@pytest.mark.parametrize("probabilities", [[0.4, 0.4], [-0.1, 1.1], [math.nan, 0.5]])
def test_invalid_answer_probabilities_fail_before_scoring(probabilities: list[float]) -> None:
    data = question().model_dump()
    for outcome, probability in zip(data["outcomes"], probabilities, strict=True):
        outcome["probability"] = probability
    with pytest.raises(ValidationError):
        ResearchQuestion.model_validate(data)


def test_contracts_reject_unknown_fields_nonfinite_ranges_and_uncited_symbols() -> None:
    with pytest.raises(ValidationError):
        RangeForecast(symbol="A", baseline_range=0.02, predicted_range=math.inf)
    with pytest.raises(ValidationError):
        RangeForecast(symbol="A", baseline_range=0.02, predicted_range=-0.01)
    data = question().model_dump()
    data["unexpected"] = True
    with pytest.raises(ValidationError):
        ResearchQuestion.model_validate(data)
    del data["unexpected"]
    data["outcomes"][0]["log_shifts"] = {"C": 1}
    with pytest.raises(ValidationError):
        ResearchQuestion.model_validate(data)


def test_logged_propensity_includes_exploration_and_exploitation() -> None:
    items = [question("q1", symbol="A"), question("q2", symbol="B")]
    common = dict(
        top_k=1,
        policy="lexicographic",
        epsilon=0.2,
        inspection_counts={"A": 0, "B": 3},
    )
    # Inverse-inspection weights are 1 and 1/4, normalized to .8 and .2.
    exploit = allocate_question(items, {"A": 0.025, "B": 0.02}, seed=0, **common)
    assert not exploit.explored
    assert exploit.question_id == "q1"
    assert exploit.propensity == pytest.approx(0.8 + 0.2 * 0.8)
    explore = allocate_question(items, {"A": 0.025, "B": 0.02}, seed=1, **common)
    assert explore.explored
    assert explore.question_id == "q2"
    assert explore.propensity == pytest.approx(0.2 * 0.2)
    assert sum(explore.diagnostics["exploration_distribution"].values()) == pytest.approx(1)


def test_random_policy_probability_includes_both_random_branches() -> None:
    decision = allocate_question(
        [question("q1", symbol="A"), question("q2", symbol="B")],
        {"A": 0.025, "B": 0.02},
        top_k=1,
        policy="random",
        epsilon=0.2,
        inspection_counts={"A": 0, "B": 3},
        seed=0,
    )
    expected = {"q1": 0.8 * 0.5 + 0.2 * 0.8, "q2": 0.8 * 0.5 + 0.2 * 0.2}
    assert decision.propensity == pytest.approx(expected[decision.question_id])


def test_lexicographic_uses_selection_consequences_before_question_id() -> None:
    decision = allocate_question(
        [question("alphabetical_first", shift=0), question("consequential_last", shift=1)],
        {"A": 0.025, "B": 0.02},
        top_k=1,
        policy="lexicographic",
        epsilon=0,
    )
    assert decision.question_id == "consequential_last"
    keys = decision.diagnostics["lexicographic_keys"]
    assert keys["consequential_last"]["membership_flip_count"] > 0
    assert keys["alphabetical_first"]["membership_flip_count"] == 0


def test_lexicographic_uses_lower_cost_after_other_criteria_tie() -> None:
    expensive = question("alphabetical_first")
    cheap = question("cheaper_last").model_copy(update={"reserved_cost_usd": 0.05})
    decision = allocate_question(
        [expensive, cheap],
        {"A": 0.025, "B": 0.02},
        top_k=1,
        policy="lexicographic",
        epsilon=0,
    )
    assert decision.question_id == "cheaper_last"


def test_discovery_remains_available_when_exploitation_has_no_action() -> None:
    item = question(discovery=True)
    fallback = allocate_question([item], {"B": 0.02}, top_k=1, epsilon=0.2, seed=0)
    assert fallback.question_id == item.question_id
    assert fallback.explored
    assert fallback.propensity == 1
    assert fallback.exploration_probability == 1
    assert fallback.diagnostics["configured_exploration_probability"] == 0.2
    explored = allocate_question([item], {"B": 0.02}, top_k=1, epsilon=0.2, seed=1)
    assert explored.question_id == item.question_id
    assert explored.propensity == 1
    assert explored.priority == 0


@pytest.mark.parametrize("policy", ["scenario", "uniform", "random", "boundary", "lexicographic"])
def test_input_order_invariance_and_reproducibility(policy: str) -> None:
    items = [question("q2", symbol="A"), question("q1", symbol="B")]
    left = allocate_question(
        items,
        {"A": 0.025, "B": 0.02},
        top_k=1,
        policy=policy,
        seed=7,
    )
    right = allocate_question(
        list(reversed(items)),
        {"B": 0.02, "A": 0.025},
        top_k=1,
        policy=policy,
        seed=7,
    )
    assert left == right


def test_uniform_prefers_uninspected_symbols_and_boundary_prefers_close_rank() -> None:
    items = [question("q1", symbol="A"), question("q2", symbol="B")]
    uniform = allocate_question(
        items,
        {"A": 0.025, "B": 0.02},
        top_k=1,
        policy="uniform",
        epsilon=0,
        inspection_counts={"A": 2},
    )
    assert uniform.question_id == "q2"
    boundary = allocate_question(
        [question("q1", symbol="A"), question("q2", symbol="C")],
        {"A": 0.05, "B": 0.021, "C": 0.02},
        top_k=2,
        policy="boundary",
        epsilon=0,
    )
    assert boundary.question_id == "q2"


def test_batch_prevents_duplicate_questions_events_and_overspend() -> None:
    items = [
        question("q1", event_ids=["event1"]),
        question("q2", event_ids=["event1"]),
        question("q3", event_ids=["event2"]),
    ]
    batch = allocate_batch(
        items,
        {"A": 0.025, "B": 0.02},
        batch_size=3,
        top_k=1,
        policy="lexicographic",
        epsilon=0,
        budget_usd=0.2,
        seed=9,
    )
    selected = [item for item in batch if item.question_id is not None]
    assert [item.question_id for item in selected] == ["q1", "q3"]
    assert sum(item.reserved_cost_usd for item in selected) <= 0.2
    assert all(item.diagnostics["batch_approximation"] for item in batch)
    assert batch[-1].question_id is None
    assert batch[-1].stop_reason == "no_eligible_questions"


def test_no_positive_score_stops_without_forcing_research() -> None:
    decision = allocate_question(
        [question(shift=0)],
        {"A": 0.025, "B": 0.02},
        top_k=1,
        epsilon=0,
    )
    assert decision.question_id is None
    assert decision.propensity == 1
    assert decision.candidate_scores == {"q1": 0}


@pytest.mark.parametrize("forecasts", [{"A": -1}, {"A": math.nan}, {"A": math.inf}])
def test_nonfinite_or_negative_prior_fails(forecasts: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        allocate_question([], forecasts, top_k=1)
