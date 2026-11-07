"""Typed graph state and reducers for deterministic fan-out/fan-in."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from financial_strategy.agents.roles import CompanyAgentOutput
from financial_strategy.data.gateway import PointInTimeSnapshot
from financial_strategy.llm import InvocationResult
from financial_strategy.schemas import (
    FeatureRecord,
    MacroContext,
    ResearchCard,
    ScreeningTask,
    VerificationReport,
    WatchlistEntry,
)


class SymbolResearchState(TypedDict, total=False):
    task: ScreeningTask
    symbol: str
    feature: FeatureRecord
    snapshot: PointInTimeSnapshot
    macro_context: MacroContext
    company_output: CompanyAgentOutput
    initial_card: ResearchCard
    card: ResearchCard
    deterministic_report: VerificationReport
    semantic_report: VerificationReport
    revision_round: int
    revision_feedback: list[dict[str, Any]]
    attempts: list[InvocationResult]
    retrieval_traces: list[dict[str, Any]]


class ScreeningState(TypedDict, total=False):
    task: ScreeningTask
    snapshot: PointInTimeSnapshot
    features: dict[str, FeatureRecord]
    candidates: list[str]
    volatility: dict[str, float]
    anomaly: dict[str, float]
    macro_context: MacroContext
    cards: Annotated[list[ResearchCard], operator.add]
    initial_cards: Annotated[list[ResearchCard], operator.add]
    attempts: Annotated[list[InvocationResult], operator.add]
    retrieval_traces: Annotated[list[dict[str, Any]], operator.add]
    watchlist: list[WatchlistEntry]
