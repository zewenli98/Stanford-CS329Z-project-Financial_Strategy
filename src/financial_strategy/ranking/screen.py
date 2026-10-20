"""Low-cost deterministic candidate screening."""

from __future__ import annotations

from financial_strategy.config import ScreeningConfig


def coarse_screen(
    volatility: dict[str, float],
    anomaly: dict[str, float],
    top_k: int,
    config: ScreeningConfig,
) -> tuple[list[str], dict[str, float]]:
    scores = {
        symbol: config.volatility_weight * volatility[symbol]
        + config.anomaly_weight * anomaly[symbol]
        for symbol in volatility.keys() & anomaly.keys()
    }
    candidate_count = min(
        len(scores),
        max(config.candidate_multiplier * top_k, config.minimum_candidates),
    )
    ordered = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))
    return ordered[:candidate_count], scores
