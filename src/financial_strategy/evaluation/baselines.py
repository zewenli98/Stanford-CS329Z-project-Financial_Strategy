"""Precisely defined non-agent baseline rankings."""

from __future__ import annotations

from financial_strategy.config import MVPConfig
from financial_strategy.schemas import EvidenceRecord, FeatureRecord


def baseline_rankings(
    features: dict[str, FeatureRecord],
    volatility: dict[str, float],
    anomaly: dict[str, float],
    company_evidence: list[EvidenceRecord],
    config: MVPConfig,
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

    no_agent_scores = {
        symbol: (0.35 * volatility[symbol] + 0.25 * anomaly[symbol]) / 0.60
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
    return {
        "historical_volatility": historical_volatility,
        "historical_abs_oc": historical_abs_oc,
        "fixed_earnings_rule": fixed_earnings,
        "no_agent_ranker": no_agent,
    }
