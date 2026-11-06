"""Strict, mean-only contracts for the adaptive range research workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from financial_strategy.schemas.models import StrictModel

FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
NonnegativeNumber = Annotated[float, Field(ge=0, allow_inf_nan=False)]
AllocationPolicy = Literal["scenario", "uniform", "random", "boundary", "lexicographic"]
ToolName = Literal["search", "read", "features", "timeline", "cards", "finish", "abstain"]


class AnswerScenario(StrictModel):
    label: str = Field(min_length=1)
    probability: Probability
    log_shifts: dict[str, FiniteNumber] = Field(default_factory=dict)


class ResearchQuestion(StrictModel):
    question_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    symbols: list[str] = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    outcomes: list[AnswerScenario] = Field(default_factory=list)
    resolution_probability: Probability
    discovery: bool = False
    allowed_tools: list[ToolName] = Field(default_factory=list)
    reserved_cost_usd: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scenarios(self) -> ResearchQuestion:
        for name in ("symbols", "evidence_ids", "event_ids", "allowed_tools"):
            values = getattr(self, name)
            if len(values) != len(set(values)) or any(not value for value in values):
                raise ValueError(f"{name} must contain unique nonempty values")
        if not self.outcomes:
            if not self.discovery:
                raise ValueError("non-discovery questions require two or three answer scenarios")
            return self
        if len(self.outcomes) not in {2, 3}:
            raise ValueError("questions require two or three answer scenarios")
        if len({item.label for item in self.outcomes}) != len(self.outcomes):
            raise ValueError("answer labels must be unique")
        if abs(sum(item.probability for item in self.outcomes) - 1.0) > 1e-8:
            raise ValueError("answer probabilities must sum to one")
        for item in self.outcomes:
            if set(item.log_shifts) - set(self.symbols):
                raise ValueError("answer shifts may reference only affected symbols")
        return self


class SourceSpan(StrictModel):
    evidence_id: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_span(self) -> SourceSpan:
        if self.end <= self.start:
            raise ValueError("source span end must be greater than start")
        return self


class InfoCard(StrictModel):
    card_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    symbols: list[str] = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    observations: list[str] = Field(min_length=1)
    inferences: list[str] = Field(default_factory=list)
    range_mechanism: str = Field(min_length=1)
    source_spans: list[SourceSpan] = Field(min_length=1)
    available_at: datetime
    expires_at: datetime | None = None
    supersedes: list[str] = Field(default_factory=list)
    counterevidence: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    event_time: datetime | None = None
    version: int = Field(default=1, ge=1)
    status: Literal["pending", "accepted", "rejected", "superseded"] = "pending"
    verification_issues: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def aware_timestamps(self) -> InfoCard:
        for name in ("available_at", "expires_at", "event_time"):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must include a timezone")
        if len(self.symbols) != len(set(self.symbols)):
            raise ValueError("card symbols must be unique")
        if self.card_id in self.supersedes:
            raise ValueError("a card cannot supersede itself")
        return self


class RangeForecast(StrictModel):
    symbol: str = Field(min_length=1)
    baseline_range: NonnegativeNumber
    predicted_range: NonnegativeNumber
    supporting_card_ids: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    fallback: bool = True
    revision: int = Field(default=0, ge=0)


class WorkerAction(StrictModel):
    tool: ToolName
    arguments: dict[str, Any] = Field(default_factory=dict)
    cards: list[InfoCard] = Field(default_factory=list)
    reason: str = ""

    @field_validator("arguments")
    @classmethod
    def remove_unused_wire_arguments(cls, value):
        # Strict single-action provider schemas represent unused keys as null.
        # Keep meaningful falsy values such as start=0 and query="".
        return {key: item for key, item in value.items() if item is not None}


class PlannerOutput(StrictModel):
    questions: list[ResearchQuestion]

    @model_validator(mode="after")
    def unique_ids(self) -> PlannerOutput:
        ids = [item.question_id for item in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("planner question IDs must be unique")
        return self


class ForecastOutput(StrictModel):
    forecasts: list[RangeForecast]

    @model_validator(mode="after")
    def unique_symbols(self) -> ForecastOutput:
        symbols = [item.symbol for item in self.forecasts]
        if len(symbols) != len(set(symbols)):
            raise ValueError("forecast symbols must be unique")
        return self


class SingleAgentOutput(StrictModel):
    """One-shot research and forecast decision used by the single-agent control."""

    cards: list[InfoCard] = Field(default_factory=list, max_length=1)
    forecasts: list[RangeForecast] = Field(default_factory=list, max_length=1)

    @model_validator(mode="after")
    def consistent_decision(self) -> SingleAgentOutput:
        if len({item.card_id for item in self.cards}) != len(self.cards):
            raise ValueError("single-agent card IDs must be unique")
        if len({item.symbol for item in self.forecasts}) != len(self.forecasts):
            raise ValueError("single-agent forecast symbols must be unique")
        return self


class ScenarioValue(StrictModel):
    question_id: str
    conditional_value: NonnegativeNumber
    priority: NonnegativeNumber
    scenario_means: dict[str, dict[str, NonnegativeNumber]]
    baseline_selection: list[str]
    transformations: list[dict[str, Any]] = Field(default_factory=list)


class AllocationDecision(StrictModel):
    question_id: str | None
    policy: AllocationPolicy
    priority: NonnegativeNumber = 0.0
    propensity: Probability
    explored: bool
    reserved_cost_usd: NonnegativeNumber = 0.0
    conditional_value: NonnegativeNumber = 0.0
    resolution_probability: Probability = 0.0
    candidate_ids: list[str] = Field(default_factory=list)
    candidate_scores: dict[str, NonnegativeNumber] = Field(default_factory=dict)
    exploration_probability: Probability
    stop_reason: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
