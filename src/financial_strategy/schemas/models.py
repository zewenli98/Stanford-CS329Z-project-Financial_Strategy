"""Strict domain contracts shared by deterministic code and agent nodes."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class StrictModel(BaseModel):
    """Reject unknown fields so schema drift fails close instead of silently leaking through."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ClaimType(StrEnum):
    OBSERVED_FACT = "observed_fact"
    COMPUTATION = "computation"
    INFERENCE = "inference"


class EventStatus(StrEnum):
    RELEASED_BEFORE_CUTOFF = "released_before_cutoff"
    SCHEDULED_BEFORE_OPEN = "scheduled_before_open"
    SCHEDULED_DURING_SESSION = "scheduled_during_session"
    SCHEDULED_AFTER_CLOSE = "scheduled_after_close"
    TIMING_UNKNOWN = "timing_unknown"
    NO_EVENT_FOUND = "no_event_found"


class Verdict(StrEnum):
    PASS = "PASS"
    REVISE = "REVISE"
    ABSTAIN = "ABSTAIN"


class ScreeningTask(StrictModel):
    task_id: str
    universe_snapshot_id: str
    decision_cutoff: datetime
    target_session: date
    top_k: int = Field(gt=0)
    price_lookback_sessions: int = Field(ge=21)
    news_lookback_days: int = Field(gt=0)
    max_revision_rounds: int = Field(ge=0)

    @model_validator(mode="after")
    def cutoff_must_be_aware(self) -> ScreeningTask:
        if self.decision_cutoff.tzinfo is None:
            raise ValueError("decision_cutoff must be timezone-aware")
        return self


class MarketBar(StrictModel):
    symbol: str
    permanent_entity_id: str
    session: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    bar_start: datetime
    bar_end: datetime
    available_at: datetime
    feed: str
    timeframe: Literal["1d"] = "1d"
    adjustment_policy: Literal["raw_unadjusted", "split_adjusted"]
    bar_status: Literal["traded", "no_trade"] = "traded"
    snapshot_id: str

    @model_validator(mode="after")
    def validate_ohlc(self) -> MarketBar:
        if self.high < max(self.open, self.close):
            raise ValueError("high must be at least max(open, close)")
        if self.low > min(self.open, self.close):
            raise ValueError("low must be at most min(open, close)")
        if self.volume == 0 and self.bar_status != "no_trade":
            raise ValueError("zero volume requires bar_status=no_trade")
        return self


class CorporateActionRecord(StrictModel):
    action_id: str
    symbol: str
    permanent_entity_id: str
    action_type: Literal["split", "reverse_split", "cash_dividend"]
    # ratio means new shares divided by old shares. A 2-for-1 split is 2.0.
    ratio: float = Field(gt=0)
    effective_at: datetime
    available_at: datetime
    source: str


class EvidenceRecord(StrictModel):
    evidence_id: str
    scope: Literal["company", "macro"]
    symbol: str | None = None
    source_type: str
    source_name: str
    source_url: HttpUrl | None = None
    title: str
    excerpt: str
    event_time: datetime | None = None
    published_at: datetime
    first_seen_at: datetime
    retrieved_at: datetime
    available_at: datetime
    content_hash: str
    snapshot_id: str
    timestamp_confidence: Confidence
    content_path: str | None = None
    event_type: str | None = None
    event_status: EventStatus | None = None

    @model_validator(mode="after")
    def company_requires_symbol(self) -> EvidenceRecord:
        if self.scope == "company" and not self.symbol:
            raise ValueError("company evidence requires symbol")
        return self


class MacroEvidenceRecord(EvidenceRecord):
    scope: Literal["macro"] = "macro"
    event_type: str
    status: EventStatus
    actual: float | None = None
    consensus: float | None = None
    consensus_source_name: str | None = None
    consensus_source_url: HttpUrl | None = None
    consensus_published_at: datetime | None = None
    consensus_content_hash: str | None = None
    unit: str | None = None
    affected_risk_factors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consensus_has_provenance(self) -> MacroEvidenceRecord:
        provenance = (
            self.consensus_source_name,
            self.consensus_source_url,
            self.consensus_published_at,
            self.consensus_content_hash,
        )
        if self.consensus is not None and any(value is not None for value in provenance):
            if any(value is None for value in provenance):
                raise ValueError("consensus provenance must be complete")
            if self.consensus_published_at > self.available_at:
                raise ValueError("consensus must be published by evidence available_at")
        return self


class EventRecord(StrictModel):
    event_id: str
    event_type: str
    status: EventStatus
    scheduled_at: datetime | None = None
    released_at: datetime | None = None
    confidence: Confidence
    evidence_ids: list[str]
    sessions_ago: int | None = Field(default=None, ge=0)


class Claim(StrictModel):
    claim_id: str
    text: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(default_factory=list)
    feature_ids: list[str] = Field(default_factory=list)


class FeatureRecord(StrictModel):
    feature_id: str
    symbol: str
    snapshot_id: str
    formula_version: str
    median_abs_oc_return_20d: float = Field(ge=0)
    realized_volatility_20d: float = Field(ge=0)
    return_5d: float
    previous_session_relative_volume: float = Field(ge=0)
    distance_to_ma20: float
    input_columns: list[str]
    input_sessions: list[date]


class SymbolMacroExposure(StrictModel):
    symbol: str
    sector: str
    exposure_method: Literal["sector_heuristic_v1"]
    template_version: str
    exposures: dict[str, float]

    @model_validator(mode="after")
    def exposure_values_in_unit_interval(self) -> SymbolMacroExposure:
        if any(value < 0 or value > 1 for value in self.exposures.values()):
            raise ValueError("macro exposures must be in [0, 1]")
        return self


class MacroContext(StrictModel):
    task_id: str
    events: list[MacroEvidenceRecord]
    evidence_ids: list[str]


class Prediction(StrictModel):
    score_type: Literal["heuristic_rank_score"] = "heuristic_rank_score"
    score: float | None = None
    volatility_component: float | None = None
    anomaly_component: float | None = None
    event_component: float | None = None
    macro_component: float | None = None
    rationale: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


class Eligibility(StrictModel):
    eligible: bool = False
    abstain_reasons: list[str] = Field(default_factory=list)


class VerificationIssue(StrictModel):
    code: str
    severity: Literal["blocking", "warning"]
    field: str
    required_action: str


class VerificationReport(StrictModel):
    verdict: Verdict
    issues: list[VerificationIssue] = Field(default_factory=list)
    attempt: int = Field(default=0, ge=0)


class ResearchCard(StrictModel):
    symbol: str
    task_id: str
    decision_cutoff: datetime
    target_session: date
    features: FeatureRecord
    events: list[EventRecord] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    prediction: Prediction = Field(default_factory=Prediction)
    deterministic_verification: VerificationReport | None = None
    semantic_verification: VerificationReport | None = None
    eligibility: Eligibility = Field(default_factory=Eligibility)
    revision_round: int = Field(default=0, ge=0)


class WatchlistEntry(StrictModel):
    rank: int = Field(gt=0)
    symbol: str
    score: float = Field(ge=0, le=1)
    volatility_component: float = Field(ge=0, le=1)
    anomaly_component: float = Field(ge=0, le=1)
    event_component: float = Field(ge=0, le=1)
    macro_component: float = Field(ge=0, le=1)
    variant: str = "full_system"


class AttemptRecord(StrictModel):
    role: str
    mode: str
    request_hash: str
    response_source: str
    attempt_type: str
    error_code: str | None = None
    latency_seconds: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
