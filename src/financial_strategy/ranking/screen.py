"""Low-cost deterministic candidate screening."""

from __future__ import annotations

import random
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from financial_strategy.config import RankingConfig, ScreeningConfig
from financial_strategy.schemas import (
    EvidenceRecord,
    MacroEvidenceRecord,
    SymbolMacroExposure,
)


class CandidateUnionAudit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool
    eligible_count: int
    total_cap: int
    candidates: list[str]
    provenance: dict[str, list[str]]
    dropped: dict[str, list[str]]


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


def multi_channel_screen(
    volatility: dict[str, float],
    anomaly: dict[str, float],
    top_k: int,
    screening: ScreeningConfig,
    ranking: RankingConfig,
    company_evidence: tuple[EvidenceRecord, ...] | list[EvidenceRecord],
    macro_evidence: tuple[MacroEvidenceRecord, ...] | list[MacroEvidenceRecord],
    exposures: tuple[SymbolMacroExposure, ...] | list[SymbolMacroExposure],
    cutoff: datetime,
) -> tuple[list[str], dict[str, float], CandidateUnionAudit]:
    """Build an audited candidate union solely from the frozen cutoff snapshot."""

    legacy, scores = coarse_screen(volatility, anomaly, top_k, screening)
    cfg = screening.candidate_union
    eligible = sorted(volatility.keys() & anomaly.keys())
    cap = min(
        len(eligible),
        cfg.total_cap or max(screening.candidate_multiplier * top_k, screening.minimum_candidates),
    )
    if not cfg.enabled:
        audit = CandidateUnionAudit(
            enabled=False,
            eligible_count=len(eligible),
            total_cap=cap,
            candidates=legacy,
            provenance={symbol: ["legacy"] for symbol in legacy},
            dropped={},
        )
        return legacy, scores, audit

    evidence = [
        item for item in company_evidence if item.available_at <= cutoff and item.symbol in eligible
    ]
    event_scores: dict[str, float] = {}
    scheduled_scores: dict[str, float] = {}
    for item in evidence:
        status = (
            ranking.company_event.status_factors.get(item.event_status.value, 0.0)
            if item.event_status
            else 0.0
        )
        event_type = max(
            (ranking.company_event.type_factors.get(value, 0.0) for value in item.event_types),
            default=0.0,
        )
        confidence = ranking.company_event.confidence_factors[item.timestamp_confidence.value]
        age_days = max(0, (cutoff.date() - item.available_at.date()).days)
        recency = next(
            (
                band.factor
                for band in ranking.company_event.recency
                if age_days <= band.max_sessions
            ),
            0.0,
        )
        value = status * event_type * confidence * recency
        event_scores[item.symbol] = max(event_scores.get(item.symbol, 0.0), value)
        if item.event_status and item.event_status.value in {
            "scheduled_before_open",
            "scheduled_during_session",
            "scheduled_after_close",
        }:
            scheduled_scores[item.symbol] = max(scheduled_scores.get(item.symbol, 0.0), value)

    market_order = sorted(eligible, key=lambda symbol: (-scores[symbol], symbol))
    edge = max(0, min(len(market_order), top_k) - 1)
    boundary_order = sorted(
        eligible,
        key=lambda symbol: (
            abs(market_order.index(symbol) - edge),
            market_order.index(symbol),
            symbol,
        ),
    )
    event_order = sorted(event_scores, key=lambda symbol: (-event_scores[symbol], symbol))
    scheduled_order = sorted(
        scheduled_scores, key=lambda symbol: (-scheduled_scores[symbol], symbol)
    )
    macro_anomalies: dict[str, float] = {}
    for item in macro_evidence:
        if item.available_at > cutoff:
            continue
        if item.actual is not None and item.consensus is not None:
            scale = ranking.macro.surprise_scales.get(item.event_type)
            if scale is None or scale <= 0:
                continue
            strength = min(1.0, abs(item.actual - item.consensus) / scale)
        else:
            strength = ranking.macro.scheduled_strength.get(item.status.value, 0.0)
        for factor in item.affected_risk_factors:
            macro_anomalies[factor] = max(macro_anomalies.get(factor, 0.0), strength)
    exposure_scores = {
        item.symbol: max(
            (
                exposure * macro_anomalies.get(factor, 0.0)
                for factor, exposure in item.exposures.items()
            ),
            default=0.0,
        )
        for item in exposures
        if item.symbol in eligible
    }
    macro_order = sorted(
        (symbol for symbol, value in exposure_scores.items() if value > 0),
        key=lambda symbol: (-exposure_scores[symbol], symbol),
    )
    random_order = list(eligible)
    random.Random(cfg.seed).shuffle(random_order)
    channels = {
        "market_top": market_order[: cfg.market_top.limit],
        "boundary": boundary_order[: cfg.boundary.limit],
        "recent_event": event_order[: cfg.recent_event.limit],
        "scheduled_catalyst": scheduled_order[: cfg.scheduled_catalyst.limit],
        "macro_exposure": macro_order[: cfg.macro_exposure.limit],
        "random_exploration": random_order[: cfg.random_exploration.limit],
    }
    provenance: dict[str, list[str]] = {}
    for channel, symbols in channels.items():
        for symbol in symbols:
            provenance.setdefault(symbol, []).append(channel)
    candidates: list[str] = []
    # Fixed round-robin precedence preserves channel quotas instead of allowing
    # the market channel to crowd every alternative source out at the final cap.
    for offset in range(max((len(symbols) for symbols in channels.values()), default=0)):
        for symbols in channels.values():
            if offset < len(symbols) and symbols[offset] not in candidates:
                candidates.append(symbols[offset])
                if len(candidates) == cap:
                    break
        if len(candidates) == cap:
            break
    dropped = {
        channel: [symbol for symbol in symbols if symbol not in candidates]
        for channel, symbols in channels.items()
        if any(symbol not in candidates for symbol in symbols)
    }
    audit = CandidateUnionAudit(
        enabled=True,
        eligible_count=len(eligible),
        total_cap=cap,
        candidates=candidates,
        provenance={symbol: provenance[symbol] for symbol in candidates},
        dropped=dropped,
    )
    return candidates, scores, audit
