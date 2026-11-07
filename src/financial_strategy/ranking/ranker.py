"""Pure scoring functions. No model is allowed to directly assign a rank score."""

from __future__ import annotations

import re
from collections import defaultdict

from financial_strategy.config import RankingConfig
from financial_strategy.schemas import (
    EventRecord,
    FeatureRecord,
    MacroContext,
    ResearchCard,
    SymbolMacroExposure,
    Verdict,
    WatchlistEntry,
)

MARKET_FEATURE_FIELDS = (
    "median_abs_oc_return_20d",
    "realized_volatility_20d",
    "return_5d",
    "previous_session_relative_volume",
    "distance_to_ma20",
    "realized_volatility_5d",
    "realized_volatility_10d",
    "realized_volatility_60d",
    "median_abs_oc_return_5d",
    "median_abs_oc_return_10d",
    "median_abs_oc_return_60d",
    "median_range_20d",
    "atr_14d",
    "robust_volume_zscore_20d",
    "momentum_10d",
    "momentum_20d",
    "reversal_1d",
    "volatility_trend_5_20",
    "market_residual_volatility_20d",
    "industry_residual_volatility_20d",
)
ABSOLUTE_MARKET_FIELDS = {
    "return_5d",
    "distance_to_ma20",
    "robust_volume_zscore_20d",
    "momentum_10d",
    "momentum_20d",
    "reversal_1d",
    "volatility_trend_5_20",
}
LLM_RESIDUAL_FEATURE_FIELDS = (
    "target_session_relevance",
    "novelty",
    "materiality",
    "expectedness",
    "expected_abs_impact",
    "signed_bucket",
    "metric_count",
    "has_reported",
    "has_consensus",
    "has_guidance",
    "before_open",
    "during_session",
    "numeric_surprise_magnitude",
    "has_numeric_surprise",
    "event_recency",
    "scheduled_proximity",
    "event_earnings",
    "event_guidance",
    "event_capital_action",
    "event_corporate_change",
    "event_risk",
    "event_product",
)


def average_percentiles(values: dict[str, float], singleton_value: float = 0.5) -> dict[str, float]:
    """Return [0, 1] average-rank percentiles with deterministic tie handling."""

    if not values:
        return {}
    if len(values) == 1:
        return {next(iter(values)): singleton_value}
    grouped: dict[float, list[str]] = defaultdict(list)
    for symbol, value in values.items():
        grouped[value].append(symbol)
    ranks: dict[str, float] = {}
    position = 1
    for value in sorted(grouped):
        symbols = sorted(grouped[value])
        average_rank = (position + position + len(symbols) - 1) / 2
        percentile = (average_rank - 1) / (len(values) - 1)
        for symbol in symbols:
            ranks[symbol] = percentile
        position += len(symbols)
    return ranks


def calibrate_market_scores(
    scores: dict[str, float],
    method: str = "cross_sectional_percentile",
) -> dict[str, float]:
    """Put market scores on the realized cross-sectional percentile scale."""

    if method == "none":
        return dict(scores)
    if method == "cross_sectional_percentile":
        return average_percentiles(scores)
    raise ValueError(f"unknown market score calibration method: {method}")


def market_components(
    features: dict[str, FeatureRecord],
    config: RankingConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    singleton = float(config.percentile["singleton_value"])
    if config.market_feature_weights is not None:
        percentile_maps = market_feature_percentiles(features, singleton)
        score = {
            symbol: sum(
                weight * percentile_maps[field][symbol]
                for field, weight in config.market_feature_weights.items()
            )
            for symbol in features
        }
        return score, score.copy()
    if config.volatility_weights is None or config.anomaly_weights is None:
        raise ValueError("component market ranker requires volatility and anomaly weights")
    med_oc = average_percentiles(
        {symbol: item.median_abs_oc_return_20d for symbol, item in features.items()}, singleton
    )
    realized = average_percentiles(
        {symbol: item.realized_volatility_20d for symbol, item in features.items()}, singleton
    )
    return_5d = average_percentiles(
        {symbol: abs(item.return_5d) for symbol, item in features.items()}, singleton
    )
    relative_volume = average_percentiles(
        {
            symbol: item.previous_session_relative_volume
            for symbol, item in features.items()
        },
        singleton,
    )
    ma_distance = average_percentiles(
        {symbol: abs(item.distance_to_ma20) for symbol, item in features.items()}, singleton
    )
    enhanced_volatility_fields = (
        "realized_volatility_5d",
        "realized_volatility_10d",
        "realized_volatility_60d",
        "median_abs_oc_return_5d",
        "median_abs_oc_return_10d",
        "median_abs_oc_return_60d",
        "median_range_20d",
        "atr_14d",
        "market_residual_volatility_20d",
        "industry_residual_volatility_20d",
    )
    enhanced_anomaly_fields = (
        "robust_volume_zscore_20d",
        "momentum_10d",
        "momentum_20d",
        "reversal_1d",
        "volatility_trend_5_20",
    )
    enhanced_volatility = _enhanced_component(
        features,
        enhanced_volatility_fields,
        singleton,
    )
    enhanced_anomaly = _enhanced_component(
        features,
        enhanced_anomaly_fields,
        singleton,
        absolute=True,
    )

    vw = config.volatility_weights
    aw = config.anomaly_weights
    base_volatility = {
        symbol: min(
            1.0,
            max(
                0.0,
                vw.median_abs_oc_return_20d * med_oc[symbol]
                + vw.realized_volatility_20d * realized[symbol],
            ),
        )
        for symbol in features
    }
    base_anomaly = {
        symbol: min(
            1.0,
            max(
                0.0,
                aw.abs_return_5d * return_5d[symbol]
                + aw.previous_session_relative_volume * relative_volume[symbol]
                + aw.abs_distance_to_ma20 * ma_distance[symbol],
            ),
        )
        for symbol in features
    }
    volatility = {
        symbol: (
            0.5 * base_volatility[symbol] + 0.5 * enhanced_volatility[symbol]
            if enhanced_volatility
            else base_volatility[symbol]
        )
        for symbol in features
    }
    anomaly = {
        symbol: (
            0.5 * base_anomaly[symbol] + 0.5 * enhanced_anomaly[symbol]
            if enhanced_anomaly
            else base_anomaly[symbol]
        )
        for symbol in features
    }
    return volatility, anomaly


def market_feature_percentiles(
    features: dict[str, FeatureRecord],
    singleton: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Return deterministic cross-sectional inputs for the learned market ranker."""

    result: dict[str, dict[str, float]] = {}
    for field in MARKET_FEATURE_FIELDS:
        values = {
            symbol: (
                abs(value) if field in ABSOLUTE_MARKET_FIELDS else value
            )
            for symbol, item in features.items()
            if (value := getattr(item, field)) is not None
        }
        result[field] = (
            average_percentiles(values, singleton)
            if len(values) == len(features)
            else {symbol: singleton for symbol in features}
        )
    return result


def _enhanced_component(
    features: dict[str, FeatureRecord],
    fields: tuple[str, ...],
    singleton: float,
    *,
    absolute: bool = False,
) -> dict[str, float]:
    percentile_maps: list[dict[str, float]] = []
    for field in fields:
        values = {
            symbol: abs(value) if absolute else value
            for symbol, item in features.items()
            if (value := getattr(item, field)) is not None
        }
        if len(values) == len(features):
            percentile_maps.append(average_percentiles(values, singleton))
    if not percentile_maps:
        return {}
    return {
        symbol: sum(mapping[symbol] for mapping in percentile_maps)
        / len(percentile_maps)
        for symbol in features
    }


def company_event_score(events: list[EventRecord], config: RankingConfig) -> float:
    event_config = config.company_event
    scores: list[float] = []
    for event in events:
        status = event_config.status_factors.get(event.status.value, 0.0)
        event_type = max(
            (
                event_config.type_factors.get(candidate, 0.0)
                for candidate in event.event_types
            ),
            default=0.0,
        )
        confidence = event_config.confidence_factors[event.confidence.value]
        recency = 0.0
        if event.sessions_ago is not None:
            for band in event_config.recency:
                if event.sessions_ago <= band.max_sessions:
                    recency = band.factor
                    break
        scores.append(status * event_type * confidence * recency)
    return max(scores, default=0.0)


def macro_score(
    context: MacroContext,
    exposure: SymbolMacroExposure,
    config: RankingConfig,
    symbol_range: float | None = None,
) -> float:
    """Score only a cited, same-session macro assessment, scaled by this stock's range."""

    cited = {
        evidence_id: assessment
        for assessment in context.assessments
        if assessment.evidence_ids
        and assessment.target_session_relevance in {"medium", "high"}
        and assessment.target_session_timing in {"before_open", "during_session"}
        for evidence_id in assessment.evidence_ids
    }
    if not cited or symbol_range is None or symbol_range <= 0:
        return 0.0
    macro = config.macro
    range_factor = min(1.0, symbol_range / 0.04)
    scores: list[float] = []
    for event in context.events:
        assessment = cited.get(event.evidence_id)
        if assessment is None:
            continue
        if (
            context.target_session is not None
            and event.event_time is not None
            and event.event_time.date() != context.target_session
        ):
            continue
        confidence = macro.confidence_factors[assessment.confidence.value]
        surprise = assessment.standardized_surprise
        if surprise is None:
            strength = macro.scheduled_strength.get(event.status.value, 0.0)
        elif abs(surprise) < macro.surprise_bins["small_upper_exclusive"]:
            strength = macro.surprise_factors["small"]
        elif abs(surprise) < macro.surprise_bins["moderate_upper_exclusive"]:
            strength = macro.surprise_factors["moderate"]
        else:
            strength = macro.surprise_factors["large"]
        factor_exposure = max(
            (exposure.exposures.get(factor, 0.0) for factor in assessment.affected_risk_factors),
            default=0.0,
        )
        scores.append(strength * factor_exposure * confidence * range_factor)
    return max(scores, default=0.0)


def catalyst_assessment_score(card: ResearchCard) -> float:
    assessment = card.catalyst_assessment
    if assessment is None or not assessment.evidence_ids:
        return 0.0
    relevance = {"none": 0.0, "low": 0.25, "medium": 0.60, "high": 1.0}
    materiality = {"unknown": 0.0, "low": 0.25, "medium": 0.60, "high": 1.0}
    impact = {
        "unknown": 0.0,
        "under_1pct": 0.15,
        "1_to_2pct": 0.35,
        "2_to_5pct": 0.70,
        "over_5pct": 1.0,
    }
    expectedness = {
        "unknown": 0.25,
        "expected": 0.25,
        "mixed": 0.60,
        "surprising": 1.0,
    }
    return (
        relevance[assessment.target_session_relevance]
        * materiality[assessment.materiality]
        * impact[assessment.expected_abs_impact_bucket]
        * expectedness[assessment.expectedness]
    )


def llm_residual_features(card: ResearchCard) -> dict[str, float]:
    """Encode only structured, citation-bearing LLM fields into stable numbers."""

    assessment = card.catalyst_assessment
    if (
        assessment is None
        or not assessment.evidence_ids
        or assessment.target_session_relevance == "none"
        or assessment.target_session_timing not in {"before_open", "during_session"}
    ):
        return {}
    ordinal = {
        "none": 0.0,
        "unknown": 0.0,
        "low": 1 / 3,
        "medium": 2 / 3,
        "high": 1.0,
        "expected": 1 / 3,
        "mixed": 2 / 3,
        "surprising": 1.0,
        "under_1pct": 0.2,
        "1_to_2pct": 0.4,
        "2_to_5pct": 0.7,
        "over_5pct": 1.0,
    }
    bucket = {
        "strong_demote": -1.0,
        "demote": -0.5,
        "neutral": 0.0,
        "promote": 0.5,
        "strong_promote": 1.0,
    }
    signed_bucket = bucket[assessment.signed_residual_bucket]
    metric_kinds = {metric.kind for metric in assessment.metrics}
    metric_values: dict[tuple[str, str, str], dict[str, float]] = {}
    for metric in assessment.metrics:
        key = (
            re.sub(
                r"\b(reported|consensus|actual|estimate|estimated)\b",
                "",
                metric.name.lower(),
            ).strip(),
            metric.period.lower(),
            metric.unit.lower(),
        )
        metric_values.setdefault(key, {})[metric.kind] = metric.value
    numeric_surprises = [
        abs(values["reported"] - values["consensus"])
        / max(abs(values["consensus"]), 1e-6)
        for values in metric_values.values()
        if "reported" in values and "consensus" in values
    ]
    event_types = {
        event_type
        for event in card.events
        for event_type in event.event_types
    }
    released_hours = [
        max(
            0.0,
            (card.decision_cutoff - event.released_at).total_seconds() / 3600,
        )
        for event in card.events
        if event.released_at is not None
        and event.released_at <= card.decision_cutoff
    ]
    scheduled_hours = [
        max(
            0.0,
            (event.scheduled_at - card.decision_cutoff).total_seconds() / 3600,
        )
        for event in card.events
        if event.scheduled_at is not None
        and event.scheduled_at >= card.decision_cutoff
    ]
    result = {
        "target_session_relevance": ordinal[assessment.target_session_relevance],
        "novelty": ordinal[assessment.novelty],
        "materiality": ordinal[assessment.materiality],
        "expectedness": ordinal[assessment.expectedness],
        "expected_abs_impact": ordinal[assessment.expected_abs_impact_bucket],
        "signed_bucket": signed_bucket,
        "metric_count": min(len(assessment.metrics), 10) / 10,
        "has_reported": float("reported" in metric_kinds),
        "has_consensus": float("consensus" in metric_kinds),
        "has_guidance": float(
            bool({"guidance_low", "guidance_high"} & metric_kinds)
        ),
        "before_open": float(assessment.target_session_timing == "before_open"),
        "during_session": float(assessment.target_session_timing == "during_session"),
        "numeric_surprise_magnitude": min(
            max(numeric_surprises, default=0.0),
            1.0,
        ),
        "has_numeric_surprise": float(bool(numeric_surprises)),
        "event_recency": (
            1.0 / (1.0 + min(released_hours) / 24.0)
            if released_hours
            else 0.0
        ),
        "scheduled_proximity": (
            1.0 / (1.0 + min(scheduled_hours) / 24.0)
            if scheduled_hours
            else 0.0
        ),
        "event_earnings": float("earnings" in event_types),
        "event_guidance": float("guidance" in event_types),
        "event_capital_action": float(
            bool({"buyback", "dividend", "financing"} & event_types)
        ),
        "event_corporate_change": float(
            bool(
                {"merger_acquisition", "management_change"} & event_types
            )
        ),
        "event_risk": float(
            bool({"litigation_regulatory", "cybersecurity"} & event_types)
        ),
        "event_product": float("product" in event_types),
    }
    return {name: result[name] for name in LLM_RESIDUAL_FEATURE_FIELDS}


_SIGNED_BUCKET = {
    "strong_demote": -1.0,
    "demote": -0.5,
    "neutral": 0.0,
    "promote": 0.5,
    "strong_promote": 1.0,
}
_TIMING_WEIGHT = {
    "before_open": 1.0,
    "during_session": 1.0,
    "prior_session": 0.6,
    "unknown": 0.5,
    "after_close": 0.35,
}
_RELEVANCE_WEIGHT = {"none": 0.0, "low": 0.5, "medium": 0.75, "high": 1.0}


def direct_verified_signed_residual(card: ResearchCard, config: RankingConfig) -> float:
    """Apply the model's own verified range residual when no ridge model is promoted.

    The learned residual stays off unless chronological validation promotes it.
    A verified non-neutral ``signed_residual_bucket`` is still the language model's
    explicit correction to the market rank, so Prediction A uses that judgment
    directly, clipped to ``max_abs_residual``. Neutral, irrelevant, or unverified
    cards leave the market score unchanged.
    """

    assessment = card.catalyst_assessment
    if assessment is None or not assessment.evidence_ids:
        return 0.0
    signed = _SIGNED_BUCKET[assessment.signed_residual_bucket]
    relevance = _RELEVANCE_WEIGHT[assessment.target_session_relevance]
    if signed == 0.0 or relevance == 0.0:
        return 0.0
    limit = config.residual_model.max_abs_residual
    value = limit * signed * _TIMING_WEIGHT[assessment.target_session_timing] * relevance
    return max(-limit, min(limit, value))


def signed_llm_residual(card: ResearchCard, config: RankingConfig) -> float:
    model = config.residual_model
    if not model.enabled:
        return 0.0
    deterministic_pass = (
        card.deterministic_verification is not None
        and card.deterministic_verification.verdict == Verdict.PASS
    )
    semantic_pass = (
        card.semantic_verification is not None
        and card.semantic_verification.verdict == Verdict.PASS
    )
    if not deterministic_pass or not semantic_pass:
        return 0.0
    features = llm_residual_features(card)
    # A verified card may still contain no usable residual assessment. In that
    # case, applying the model intercept would modify the market rank without
    # any structured LLM signal and would disagree with the training filter.
    if not features:
        return 0.0
    value = model.intercept + sum(
        coefficient * features.get(name, 0.0)
        for name, coefficient in model.coefficients.items()
    )
    return max(-model.max_abs_residual, min(model.max_abs_residual, value))


def macro_interaction_residual(card: ResearchCard, config: RankingConfig) -> float:
    """No signed macro residual is applied until one is trained and calibrated."""

    return 0.0


def rank_cards(
    cards: list[ResearchCard],
    volatility: dict[str, float],
    anomaly: dict[str, float],
    macro_context: MacroContext,
    exposures: dict[str, SymbolMacroExposure],
    config: RankingConfig,
    top_k: int,
    variant: str = "full_system",
) -> list[WatchlistEntry]:
    weighted: list[
        tuple[str, float, float, float, float, float, float, float, float, float]
    ] = []
    weights = config.final_weights
    market_weight = weights.volatility + weights.anomaly
    raw_market_scores = {
        card.symbol: (
            weights.volatility * volatility[card.symbol]
            + weights.anomaly * anomaly[card.symbol]
        )
        / market_weight
        for card in cards
    }
    market_scores = calibrate_market_scores(
        raw_market_scores,
        (
            config.residual_model.market_score_calibration
            if config.residual_model.enabled
            else "none"
        ),
    )
    for card in cards:
        deterministic_pass = (
            card.deterministic_verification is not None
            and card.deterministic_verification.verdict == Verdict.PASS
        )
        semantic_pass = (
            variant == "no_semantic_verifier"
            or (
                card.semantic_verification is not None
                and card.semantic_verification.verdict == Verdict.PASS
            )
        )
        agent_pass = deterministic_pass and semantic_pass
        # Verification protects Agent-derived facts and score uplift. It must not
        # delete a symbol whose independently computed market features are valid.
        event_component = (
            max(
                company_event_score(card.events, config),
                catalyst_assessment_score(card),
            )
            if agent_pass
            else 0.0
        )
        macro_component = (
            macro_score(
                macro_context,
                exposures[card.symbol],
                config,
                card.features.median_range_20d,
            )
            if agent_pass
            else 0.0
        )
        if variant == "no_company_event":
            event_component = 0.0
        if variant == "no_macro":
            macro_component = 0.0
        market_score = market_scores[card.symbol]
        # Event and macro heuristic components stay on the audit record.
        # The published correction is either the promoted ridge residual or,
        # when that model is off, the verified signed residual from the
        # language-model assessment.
        if variant == "no_company_event":
            company_residual = 0.0
        elif config.residual_model.enabled:
            company_residual = signed_llm_residual(card, config)
        elif agent_pass:
            company_residual = direct_verified_signed_residual(card, config)
        else:
            company_residual = 0.0
        macro_residual = (
            macro_interaction_residual(card, config)
            if variant != "no_macro"
            else 0.0
        )
        learned_residual = company_residual + macro_residual
        score = min(1.0, max(0.0, market_score + learned_residual))
        if score >= config.minimum_score:
            weighted.append(
                (
                    card.symbol,
                    score,
                    volatility[card.symbol],
                    anomaly[card.symbol],
                    event_component,
                    macro_component,
                    market_score,
                    learned_residual,
                    company_residual,
                    macro_residual,
                )
            )

    weighted.sort(key=lambda item: (-item[1], item[0]))
    return [
        WatchlistEntry(
            rank=index,
            symbol=item[0],
            score=item[1],
            volatility_component=item[2],
            anomaly_component=item[3],
            event_component=item[4],
            macro_component=item[5],
            market_score=item[6],
            signed_residual=item[7],
            company_residual=item[8],
            macro_residual=item[9],
            variant=variant,
        )
        for index, item in enumerate(weighted[:top_k], start=1)
    ]
