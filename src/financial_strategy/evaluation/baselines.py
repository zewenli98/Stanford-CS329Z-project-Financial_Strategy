"""Precisely defined non-agent baseline rankings."""

from __future__ import annotations

from datetime import datetime

from financial_strategy.config import MVPConfig
from financial_strategy.schemas import EvidenceRecord, FeatureRecord


def baseline_rankings(
    features: dict[str, FeatureRecord],
    volatility: dict[str, float],
    anomaly: dict[str, float],
    company_evidence: list[EvidenceRecord],
    config: MVPConfig,
    cutoff: datetime | None = None,
) -> dict[str, list[str]]:
    top_k = config.task.top_k
    historical_volatility = sorted(
        features,
        key=lambda symbol: (-features[symbol].realized_volatility_20d, symbol),
    )[:top_k]
    historical_abs_oc = sorted(
        features,
        key=lambda symbol: (-features[symbol].median_abs_oc_return_20d, symbol),
    )[:top_k]

    status_factors = config.ranking.company_event.status_factors
    earnings_scores = {}
    for symbol in features:
        scores = [
            status_factors.get(item.event_status.value, 0.0)
            for item in company_evidence
            if item.symbol == symbol
            and "earnings" in item.event_types
            and item.event_status is not None
        ]
        score = max(scores, default=0.0)
        if score > 0:
            earnings_scores[symbol] = score
    fixed_earnings = sorted(
        earnings_scores,
        key=lambda symbol: (-earnings_scores[symbol], symbol),
    )[:top_k]

    weights = config.ranking.final_weights
    market_weight = weights.volatility + weights.anomaly
    no_agent_scores = {
        symbol: (
            weights.volatility * volatility[symbol]
            + weights.anomaly * anomaly[symbol]
        )
        / market_weight
        for symbol in features
    }
    no_agent = [
        symbol
        for symbol in sorted(
            no_agent_scores,
            key=lambda symbol: (-no_agent_scores[symbol], symbol),
        )
        if no_agent_scores[symbol] >= config.ranking.minimum_score
    ][:top_k]
    deterministic_scores = dict(no_agent_scores)
    event_config = config.ranking.company_event
    if cutoff is None:
        cutoff = config.task.decision_cutoff
    for item in company_evidence:
        if item.available_at > cutoff or item.symbol not in deterministic_scores:
            continue
        status = event_config.status_factors.get(item.event_status.value, 0.0) if item.event_status else 0.0
        event_type = max((event_config.type_factors.get(value, 0.0) for value in item.event_types), default=0.0)
        confidence = event_config.confidence_factors[item.timestamp_confidence.value]
        age_days = max(0, (cutoff.date() - item.available_at.date()).days)
        recency = next((band.factor for band in event_config.recency if age_days <= band.max_sessions), 0.0)
        deterministic_scores[item.symbol] += status * event_type * confidence * recency
    deterministic_event = sorted(
        deterministic_scores,
        key=lambda symbol: (-deterministic_scores[symbol], symbol),
    )[:top_k]
    return {
        "historical_volatility": historical_volatility,
        "historical_abs_oc": historical_abs_oc,
        "fixed_earnings_rule": fixed_earnings,
        "no_agent_ranker": no_agent,
        "deterministic_event_baseline": deterministic_event,
    }
