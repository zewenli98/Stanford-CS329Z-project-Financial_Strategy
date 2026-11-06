"""Leakage-safe rolling macro sensitivities estimated from completed releases."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

from financial_strategy.schemas import (
    MacroEvidenceRecord,
    SymbolMacroExposure,
    SymbolMacroInteraction,
    SymbolMacroSensitivity,
)

MODEL_VERSION = "rolling_macro_ridge_v1"


def _surprise(event: MacroEvidenceRecord, scales: dict[str, float]) -> float | None:
    scale = scales.get(event.event_type)
    if (
        event.actual is None
        or event.consensus is None
        or scale is None
        or scale <= 0
    ):
        return None
    return (event.actual - event.consensus) / scale


def _coefficient(values: list[tuple[float, float]], shrinkage: float) -> float:
    numerator = sum(feature * target for feature, target in values)
    denominator = sum(feature * feature for feature, _ in values) + shrinkage
    return numerator / denominator if denominator > 0 else 0.0


def build_macro_interactions(
    *,
    target_session: date,
    target_events: list[MacroEvidenceRecord],
    history_events: list[MacroEvidenceRecord],
    returns: dict[str, dict[str, float]],
    sectors: dict[str, str],
    exposures: dict[str, SymbolMacroExposure],
    surprise_scales: dict[str, float],
    shrinkage: float = 5.0,
    minimum_symbol_events: int = 8,
    minimum_sector_events: int = 6,
    minimum_global_events: int = 6,
) -> dict[str, list[SymbolMacroInteraction]]:
    """Build interactions using only release sessions strictly before the target."""

    history = [
        event
        for event in history_events
        if event.event_time is not None
        and event.event_time.date() < target_session
        and event.status == "released_before_cutoff"
        and _surprise(event, surprise_scales) is not None
        and event.event_time.date().isoformat() in returns
    ]
    observations: dict[
        str, dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    sector_observations: dict[
        str, dict[str, list[tuple[float, float]]]
    ] = defaultdict(lambda: defaultdict(list))
    global_observations: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for event in history:
        event_session = event.event_time.date().isoformat()
        session_returns = returns[event_session]
        surprise = _surprise(event, surprise_scales)
        if surprise is None:
            continue
        by_sector: dict[str, list[float]] = defaultdict(list)
        for symbol, value in session_returns.items():
            sector = sectors.get(symbol)
            if sector is not None:
                by_sector[sector].append(value)
        sector_medians = {
            sector: _median(values) for sector, values in by_sector.items()
        }
        excess_values = []
        for symbol, value in session_returns.items():
            sector = sectors.get(symbol)
            if sector is None or sector not in sector_medians:
                continue
            excess = value - sector_medians[sector]
            observations[event.event_type][symbol].append((surprise, excess))
            sector_observations[event.event_type][sector].append((surprise, excess))
            excess_values.append(excess)
        if excess_values:
            global_observations[event.event_type].append(
                (surprise, _median(excess_values))
            )

    confidence = {"high": 1.0, "medium": 0.75, "low": 0.5}
    result: dict[str, list[SymbolMacroInteraction]] = {
        symbol: [] for symbol in sectors
    }
    for event in target_events:
        if event.event_time is None or event.event_time.date() != target_session:
            continue
        surprise = _surprise(event, surprise_scales)
        # A scheduled FOMC decision is knowable before the open even though its
        # eventual rate surprise is not. Encode event presence, never the future actual.
        if surprise is None and event.status == "scheduled_during_session":
            surprise = 1.0
        if surprise is None:
            continue
        for symbol, sector in sectors.items():
            symbol_values = observations[event.event_type].get(symbol, [])
            sector_values = sector_observations[event.event_type].get(sector, [])
            global_values = global_observations[event.event_type]
            if len(symbol_values) >= minimum_symbol_events:
                values = symbol_values
                fallback = "symbol"
            elif len(sector_values) >= minimum_sector_events:
                values = sector_values
                fallback = "sector"
            elif len(global_values) >= minimum_global_events:
                values = global_values
                fallback = "global"
            else:
                values = []
                fallback = "zero"
            coefficient = _coefficient(values, shrinkage) if values else 0.0
            sensitivity = SymbolMacroSensitivity(
                symbol=symbol,
                event_type=event.event_type,
                coefficient=coefficient,
                sample_count=len(values),
                as_of_session=target_session,
                fallback_level=fallback,
                model_version=MODEL_VERSION,
            )
            exposure = max(
                (
                    exposures[symbol].exposures.get(factor, 0.0)
                    for factor in event.affected_risk_factors
                ),
                default=0.0,
            )
            confidence_factor = confidence[event.timestamp_confidence.value]
            signed_signal = surprise * coefficient * exposure * confidence_factor
            result[symbol].append(
                SymbolMacroInteraction(
                    symbol=symbol,
                    event_type=event.event_type,
                    evidence_ids=[event.evidence_id],
                    standardized_surprise=surprise,
                    sensitivity=sensitivity,
                    exposure=exposure,
                    confidence_factor=confidence_factor,
                    signed_signal=signed_signal,
                    expected_abs_impact=abs(signed_signal),
                )
            )
    return result


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
