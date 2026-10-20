"""Pure scoring functions. No model is allowed to directly assign a rank score."""

from __future__ import annotations

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


def market_components(
    features: dict[str, FeatureRecord],
    config: RankingConfig,
) -> tuple[dict[str, float], dict[str, float]]:
    singleton = float(config.percentile["singleton_value"])
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

    vw = config.volatility_weights
    aw = config.anomaly_weights
    volatility = {
        symbol: vw.median_abs_oc_return_20d * med_oc[symbol]
        + vw.realized_volatility_20d * realized[symbol]
        for symbol in features
    }
    anomaly = {
        symbol: aw.abs_return_5d * return_5d[symbol]
        + aw.previous_session_relative_volume * relative_volume[symbol]
        + aw.abs_distance_to_ma20 * ma_distance[symbol]
        for symbol in features
    }
    return volatility, anomaly


def company_event_score(events: list[EventRecord], config: RankingConfig) -> float:
    event_config = config.company_event
    scores: list[float] = []
    for event in events:
        status = event_config.status_factors.get(event.status.value, 0.0)
        event_type = event_config.type_factors.get(event.event_type, 0.0)
        confidence = event_config.confidence_factors[event.confidence.value]
        recency = 1.0
        if event.sessions_ago is not None:
            recency = 0.0
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
) -> float:
    macro = config.macro
    scores: list[float] = []
    for event in context.events:
        confidence = macro.confidence_factors[event.timestamp_confidence.value]
        if event.actual is not None and event.consensus is not None:
            scale = macro.surprise_scales.get(event.event_type)
            if scale is None:
                continue
            z_value = abs(event.actual - event.consensus) / scale
            if z_value < macro.surprise_bins["small_upper_exclusive"]:
                strength = macro.surprise_factors["small"]
            elif z_value < macro.surprise_bins["moderate_upper_exclusive"]:
                strength = macro.surprise_factors["moderate"]
            else:
                strength = macro.surprise_factors["large"]
        else:
            strength = macro.scheduled_strength.get(event.status.value, 0.0)
        factor_exposure = max(
            (exposure.exposures.get(factor, 0.0) for factor in event.affected_risk_factors),
            default=0.0,
        )
        scores.append(strength * factor_exposure * confidence)
    return max(scores, default=0.0)


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
    weighted: list[tuple[str, float, float, float, float, float]] = []
    weights = config.final_weights
    for card in cards:
        deterministic_pass = (
            card.deterministic_verification is not None
            and card.deterministic_verification.verdict == Verdict.PASS
        )
        semantic_pass = (
            card.semantic_verification is not None
            and card.semantic_verification.verdict == Verdict.PASS
        )
        if variant == "no_semantic_verifier":
            verified = deterministic_pass
        else:
            verified = deterministic_pass and semantic_pass
        if not verified:
            continue

        event_component = company_event_score(card.events, config)
        macro_component = macro_score(macro_context, exposures[card.symbol], config)
        if variant == "no_company_event":
            event_component = 0.0
        if variant == "no_macro":
            macro_component = 0.0
        score = (
            weights.volatility * volatility[card.symbol]
            + weights.anomaly * anomaly[card.symbol]
            + weights.company_event * event_component
            + weights.macro * macro_component
        )
        if score >= config.minimum_score:
            weighted.append(
                (
                    card.symbol,
                    score,
                    volatility[card.symbol],
                    anomaly[card.symbol],
                    event_component,
                    macro_component,
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
            variant=variant,
        )
        for index, item in enumerate(weighted[:top_k], start=1)
    ]
