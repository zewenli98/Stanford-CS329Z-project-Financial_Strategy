"""Reproducible, mean-preserving acquisition heuristics and logged exploration."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from itertools import combinations

from financial_strategy.adaptive.models import (
    AllocationDecision,
    AllocationPolicy,
    ResearchQuestion,
    ScenarioValue,
)


def _validate_means(forecasts: Mapping[str, float], top_k: int) -> dict[str, float]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    means = {symbol: float(value) for symbol, value in forecasts.items()}
    if any(not math.isfinite(value) or value < 0 for value in means.values()):
        raise ValueError("current range means must be finite and nonnegative")
    return means


def _top_symbols(means: Mapping[str, float], top_k: int) -> list[str]:
    return sorted(means, key=lambda symbol: (-means[symbol], symbol))[:top_k]


def scenario_value(
    question: ResearchQuestion,
    forecasts: Mapping[str, float],
    *,
    top_k: int,
    shift_clip: float = 1.5,
    shift_shrinkage: float = 0.5,
) -> ScenarioValue:
    """Score hypothetical answer sensitivity, not calibrated expected information value.

    Clip each proposed log shift, shrink it, then normalize its multipliers across
    answers. Do not clip normalized means: that would break mean preservation.
    """
    means = _validate_means(forecasts, top_k)
    if not math.isfinite(shift_clip) or not 0 <= shift_clip <= 100:
        raise ValueError("shift_clip must be finite and between zero and 100")
    if not math.isfinite(shift_shrinkage) or not 0 <= shift_shrinkage <= 1:
        raise ValueError("shift_shrinkage must be in [0, 1]")
    unknown = set(question.symbols) - set(means)
    if unknown:
        raise ValueError(f"question references unknown symbols: {sorted(unknown)}")
    baseline = _top_symbols(means, top_k)
    if not question.outcomes:
        return ScenarioValue(
            question_id=question.question_id,
            conditional_value=0,
            priority=0,
            scenario_means={},
            baseline_selection=baseline,
        )
    scenario_means = {outcome.label: dict(means) for outcome in question.outcomes}
    transformations = []
    for symbol in sorted(question.symbols):
        shifts = {}
        for outcome in question.outcomes:
            proposed = outcome.log_shifts.get(symbol, 0.0)
            clipped = max(-shift_clip, min(shift_clip, proposed))
            applied = clipped * shift_shrinkage
            shifts[outcome.label] = applied
            transformations.append(
                {
                    "symbol": symbol,
                    "answer": outcome.label,
                    "proposed_log_shift": proposed,
                    "clipped_log_shift": clipped,
                    "applied_log_shift": applied,
                }
            )
        normalizer = math.fsum(
            outcome.probability * math.exp(shifts[outcome.label]) for outcome in question.outcomes
        )
        for outcome in question.outcomes:
            scenario_means[outcome.label][symbol] = (
                means[symbol] * math.exp(shifts[outcome.label]) / normalizer
            )
        for record in transformations:
            if record["symbol"] == symbol:
                record["normalizer"] = normalizer
    conditional_value = math.fsum(
        outcome.probability
        * max(
            0.0,
            math.fsum(
                scenario_means[outcome.label][s]
                for s in _top_symbols(scenario_means[outcome.label], top_k)
            )
            - math.fsum(scenario_means[outcome.label][s] for s in baseline),
        )
        for outcome in question.outcomes
    )
    return ScenarioValue(
        question_id=question.question_id,
        conditional_value=conditional_value,
        priority=question.resolution_probability * conditional_value / question.reserved_cost_usd,
        scenario_means=scenario_means,
        baseline_selection=baseline,
        transformations=transformations,
    )


def allocate_question(
    questions: Sequence[ResearchQuestion],
    forecasts: Mapping[str, float],
    *,
    top_k: int,
    policy: AllocationPolicy = "scenario",
    epsilon: float = 0.2,
    inspection_counts: Mapping[str, int] | None = None,
    seed: int = 0,
    budget_usd: float | None = None,
    excluded_event_ids: set[str] | None = None,
    excluded_question_ids: set[str] | None = None,
) -> AllocationDecision:
    """Select one action; propensity includes both mixture branches.

    If exploitation is unavailable and exploration is enabled, all feasible
    actions are exploratory. Inspection weights are 1/(1+sum affected counts).
    """
    means = _validate_means(forecasts, top_k)
    if policy not in {"scenario", "uniform", "random", "boundary", "lexicographic"}:
        raise ValueError(f"unknown allocation policy: {policy}")
    if not math.isfinite(epsilon) or not 0 <= epsilon <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if budget_usd is not None and (not math.isfinite(budget_usd) or budget_usd < 0):
        raise ValueError("budget_usd must be finite and nonnegative")
    counts = inspection_counts or {}
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in counts.values()):
        raise ValueError("inspection counts must be nonnegative integers")
    ids = [q.question_id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("question IDs must be unique")
    blocked_events = excluded_event_ids or set()
    blocked_questions = excluded_question_ids or set()
    eligible = sorted(
        [
            q
            for q in questions
            if q.question_id not in blocked_questions
            and not blocked_events.intersection(q.event_ids)
            and (budget_usd is None or q.reserved_cost_usd <= budget_usd)
        ],
        key=lambda q: q.question_id,
    )
    candidate_ids = [q.question_id for q in eligible]
    values = {q.question_id: scenario_value(q, means, top_k=top_k) for q in eligible}
    inspections = {q.question_id: sum(counts.get(s, 0) for s in q.symbols) for q in eligible}
    weights = {q.question_id: 1 / (1 + inspections[q.question_id]) for q in eligible}
    weight_total = math.fsum(weights.values())
    exploration = {key: value / weight_total for key, value in weights.items()}
    scores = {key: value.priority for key, value in values.items()}
    exploitation: dict[str, float] = {}
    actionable = [q for q in eligible if not q.discovery]
    lexicographic_keys: dict[str, tuple[int, int, int, float, str]] = {}
    if policy == "uniform":
        scores = {q.question_id: weights[q.question_id] for q in eligible}
    elif policy == "random":
        scores = {q.question_id: 1.0 for q in eligible}
    elif policy == "lexicographic":
        baseline_order = _top_symbols(means, len(means))
        baseline_positions = {symbol: rank for rank, symbol in enumerate(baseline_order)}
        for question in eligible:
            value = values[question.question_id]
            answer_orders = [
                _top_symbols(scenario, len(means))
                for label, scenario in value.scenario_means.items()
                if next(a for a in question.outcomes if a.label == label).probability > 0
            ]
            selections = [set(baseline_order[:top_k])]
            selections.extend(set(order[:top_k]) for order in answer_orders)
            flips = sum(len(left ^ right) for left, right in combinations(selections, 2))
            displacement = sum(
                abs(rank - baseline_positions[symbol])
                for order in answer_orders
                for rank, symbol in enumerate(order)
            )
            coverage = max(0, len(question.evidence_ids) - inspections[question.question_id])
            lexicographic_keys[question.question_id] = (
                -flips,
                -displacement,
                -coverage,
                question.reserved_cost_usd,
                question.question_id,
            )
        scores = {question_id: float(-key[0]) for question_id, key in lexicographic_keys.items()}
    elif policy == "boundary":
        ranked = _top_symbols(means, len(means))
        scores = dict.fromkeys(candidate_ids, 0.0)
        if top_k < len(ranked):
            boundary = (means[ranked[top_k - 1]] + means[ranked[top_k]]) / 2
            scores = {
                q.question_id: 1 / (1e-12 + min(abs(means[s] - boundary) for s in q.symbols))
                for q in eligible
            }
    if policy == "lexicographic" and actionable:
        winner = min(actionable, key=lambda q: lexicographic_keys[q.question_id])
        exploitation = {winner.question_id: 1.0}
    elif policy == "random" and actionable:
        exploitation = {q.question_id: 1 / len(actionable) for q in actionable}
    else:
        positive = [q for q in actionable if scores[q.question_id] > 0]
        if positive:
            winner = min(positive, key=lambda q: (-scores[q.question_id], q.question_id))
            exploitation = {winner.question_id: 1.0}
    diagnostics = {
        "configured_exploration_probability": epsilon,
        "exploration_distribution": exploration,
        "exploitation_distribution": exploitation,
        "scenario_scores": {key: value.model_dump(mode="json") for key, value in values.items()},
        "inspection_counts": dict(counts),
        "lexicographic_keys": {
            key: {
                "membership_flip_count": -value[0],
                "rank_displacement": -value[1],
                "uninspected_evidence_coverage": -value[2],
                "reserved_cost_usd": value[3],
            }
            for key, value in lexicographic_keys.items()
        },
        "seed": seed,
    }
    if not eligible:
        return AllocationDecision(
            question_id=None,
            policy=policy,
            propensity=1,
            explored=False,
            candidate_ids=[],
            candidate_scores={},
            exploration_probability=epsilon,
            stop_reason="no_eligible_questions",
            diagnostics=diagnostics,
        )
    rng = random.Random(seed)
    # A zero estimated gain is not evidence that the archive is exhausted. When
    # no greedy action exists, condition on the remaining exploratory actions
    # and log their effective probability rather than stochastically terminating.
    if not exploitation and epsilon > 0:
        epsilon = 1.0
    explored = rng.random() < epsilon
    distribution = exploration if explored else exploitation
    if not distribution:
        return AllocationDecision(
            question_id=None,
            policy=policy,
            propensity=1 - epsilon,
            explored=False,
            candidate_ids=candidate_ids,
            candidate_scores=scores,
            exploration_probability=epsilon,
            stop_reason="no_positive_exploitation_action",
            diagnostics=diagnostics,
        )
    choices = sorted(distribution)
    selected_id = rng.choices(choices, weights=[distribution[key] for key in choices], k=1)[0]
    selected = next(q for q in eligible if q.question_id == selected_id)
    value = values[selected_id]
    propensity = epsilon * exploration[selected_id] + (1 - epsilon) * exploitation.get(
        selected_id, 0.0
    )
    return AllocationDecision(
        question_id=selected_id,
        policy=policy,
        priority=scores[selected_id],
        propensity=propensity,
        explored=explored,
        reserved_cost_usd=selected.reserved_cost_usd,
        conditional_value=value.conditional_value,
        resolution_probability=selected.resolution_probability,
        candidate_ids=candidate_ids,
        candidate_scores=scores,
        exploration_probability=epsilon,
        diagnostics=diagnostics,
    )


def allocate_batch(
    questions: Sequence[ResearchQuestion],
    forecasts: Mapping[str, float],
    *,
    batch_size: int,
    top_k: int,
    policy: AllocationPolicy = "scenario",
    epsilon: float = 0.2,
    inspection_counts: Mapping[str, int] | None = None,
    seed: int = 0,
    budget_usd: float | None = None,
    excluded_event_ids: set[str] | None = None,
    excluded_question_ids: set[str] | None = None,
) -> list[AllocationDecision]:
    """Approximate a parallel batch against one forecast state, without duplicate events."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    blocked_events = set(excluded_event_ids or ())
    blocked_questions = set(excluded_question_ids or ())
    counts = dict(inspection_counts or {})
    remaining = budget_usd
    rng = random.Random(seed)
    decisions = []
    by_id = {q.question_id: q for q in questions}
    for index in range(batch_size):
        decision = allocate_question(
            questions,
            forecasts,
            top_k=top_k,
            policy=policy,
            epsilon=epsilon,
            inspection_counts=counts,
            seed=rng.randrange(2**63),
            budget_usd=remaining,
            excluded_event_ids=blocked_events,
            excluded_question_ids=blocked_questions,
        )
        decision.diagnostics.update({"batch_approximation": True, "batch_position": index})
        decisions.append(decision)
        if decision.question_id is None:
            break
        selected = by_id[decision.question_id]
        blocked_questions.add(selected.question_id)
        blocked_events.update(selected.event_ids)
        for symbol in selected.symbols:
            counts[symbol] = counts.get(symbol, 0) + 1
        if remaining is not None:
            remaining = max(0.0, remaining - selected.reserved_cost_usd)
    return decisions
