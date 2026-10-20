"""Validated YAML configuration and canonical hashing."""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_strategy.errors import ConfigError


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskConfig(ConfigModel):
    universe_snapshot_id: str
    decision_cutoff: datetime
    target_session: date
    top_k: int = Field(gt=0)
    price_lookback_sessions: int = Field(ge=21)
    news_lookback_days: int = Field(gt=0)


class DataConfig(ConfigModel):
    universe: Path
    market_bars: Path
    corporate_actions: Path
    company_evidence: Path
    macro_events: Path
    macro_documents: Path
    sector_exposures: Path
    input_adjustment_policy: Literal["raw_unadjusted", "split_adjusted"]
    snapshot_adjustment_policy: Literal["split_adjusted"]
    zero_volume_policy: Literal["require_explicit_no_trade_status"]


class WeightConfig(ConfigModel):
    volatility: float
    anomaly: float
    company_event: float
    macro: float

    @model_validator(mode="after")
    def sum_to_one(self) -> WeightConfig:
        if abs(sum(self.model_dump().values()) - 1.0) > 1e-9:
            raise ValueError("final weights must sum to 1")
        return self


class PairWeightConfig(ConfigModel):
    median_abs_oc_return_20d: float
    realized_volatility_20d: float

    @model_validator(mode="after")
    def sum_to_one(self) -> PairWeightConfig:
        if abs(sum(self.model_dump().values()) - 1.0) > 1e-9:
            raise ValueError("volatility weights must sum to 1")
        return self


class AnomalyWeightConfig(ConfigModel):
    abs_return_5d: float
    previous_session_relative_volume: float
    abs_distance_to_ma20: float

    @model_validator(mode="after")
    def sum_to_one(self) -> AnomalyWeightConfig:
        if abs(sum(self.model_dump().values()) - 1.0) > 1e-9:
            raise ValueError("anomaly weights must sum to 1")
        return self


class RecencyBand(ConfigModel):
    max_sessions: int = Field(gt=0)
    factor: float = Field(ge=0, le=1)


class CompanyEventConfig(ConfigModel):
    status_factors: dict[str, float]
    type_factors: dict[str, float]
    confidence_factors: dict[str, float]
    recency: list[RecencyBand]

    @model_validator(mode="after")
    def validate_factors(self) -> CompanyEventConfig:
        factor_maps = (
            self.status_factors,
            self.type_factors,
            self.confidence_factors,
        )
        if any(not 0 <= value <= 1 for factors in factor_maps for value in factors.values()):
            raise ValueError("company event factors must be in [0, 1]")
        expected_statuses = {
            "released_before_cutoff",
            "scheduled_before_open",
            "scheduled_during_session",
            "scheduled_after_close",
            "timing_unknown",
            "no_event_found",
        }
        if set(self.status_factors) != expected_statuses:
            raise ValueError("company status factors must define every supported status")
        if set(self.confidence_factors) != {"high", "medium", "low"}:
            raise ValueError("company confidence factors must define high, medium, and low")
        boundaries = [band.max_sessions for band in self.recency]
        if not boundaries or boundaries != sorted(set(boundaries)):
            raise ValueError("recency boundaries must be present and strictly increasing")
        return self


class MacroRankingConfig(ConfigModel):
    exposure_method: Literal["sector_heuristic_v1"]
    supported_risk_factors: list[str]
    confidence_factors: dict[str, float]
    surprise_factors: dict[str, float]
    surprise_bins: dict[str, float]
    surprise_scales: dict[str, float]
    scheduled_strength: dict[str, float]

    @model_validator(mode="after")
    def validate_macro(self) -> MacroRankingConfig:
        if len(set(self.supported_risk_factors)) != len(self.supported_risk_factors):
            raise ValueError("supported risk factors must be unique")
        if any(value <= 0 for value in self.surprise_scales.values()):
            raise ValueError("surprise scales must be positive")
        factor_maps = (
            self.confidence_factors,
            self.surprise_factors,
            self.scheduled_strength,
        )
        if any(not 0 <= value <= 1 for factors in factor_maps for value in factors.values()):
            raise ValueError("macro factors must be in [0, 1]")
        if set(self.confidence_factors) != {"high", "medium", "low"}:
            raise ValueError("macro confidence factors must define high, medium, and low")
        if set(self.surprise_factors) != {"small", "moderate", "large"}:
            raise ValueError("macro surprise factors must define small, moderate, and large")
        small = self.surprise_bins.get("small_upper_exclusive")
        moderate = self.surprise_bins.get("moderate_upper_exclusive")
        if small is None or moderate is None or not 0 < small < moderate:
            raise ValueError("macro surprise bin boundaries must be positive and increasing")
        return self


class RankingConfig(ConfigModel):
    minimum_score: float = Field(ge=0, le=1)
    final_weights: WeightConfig
    volatility_weights: PairWeightConfig
    anomaly_weights: AnomalyWeightConfig
    percentile: dict[str, str | float]
    company_event: CompanyEventConfig
    macro: MacroRankingConfig

    @model_validator(mode="after")
    def validate_percentile(self) -> RankingConfig:
        if self.percentile.get("tie_method") != "average":
            raise ValueError("percentile tie_method must be average")
        singleton = self.percentile.get("singleton_value")
        if not isinstance(singleton, (int, float)) or not 0 <= singleton <= 1:
            raise ValueError("percentile singleton_value must be in [0, 1]")
        return self


class RoleConfig(ConfigModel):
    model: str | None = None
    prompt_version: str
    max_tokens: int = Field(gt=0)


class AgentsConfig(ConfigModel):
    mode: Literal["scripted", "recorded", "live"]
    provider: Literal["openai", "openrouter"] = "openai"
    model: str
    recordings_path: Path
    cache_path: Path
    max_parallel_symbols: int = Field(gt=0)
    macro_max_tool_calls: int = Field(gt=0)
    company_max_tool_calls_per_symbol: int = Field(gt=0)
    max_model_calls_per_symbol: int = Field(gt=0)
    max_total_tool_calls: int = Field(gt=0)
    max_total_model_calls: int = Field(gt=0)
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def validate_roles(self) -> AgentsConfig:
        expected = {"macro_evidence", "company_evidence", "synthesis", "verifier"}
        if set(self.roles) != expected:
            raise ValueError(f"agents.roles must define exactly {sorted(expected)}")
        if self.mode == "live" and self.provider == "openrouter":
            invalid = [
                self.model_for(role)
                for role in expected
                if "/" not in self.model_for(role)
            ]
            if invalid:
                raise ValueError("OpenRouter model IDs must use provider/model format")
        return self

    def model_for(self, role: str) -> str:
        return self.roles[role].model or self.model


class RetrievalConfig(ConfigModel):
    algorithm: Literal["bm25"]
    top_n: int = Field(gt=0)
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    tokenizer_version: str


class ScreeningConfig(ConfigModel):
    candidate_multiplier: int = Field(gt=0)
    minimum_candidates: int = Field(gt=0)
    volatility_weight: float = Field(ge=0, le=1)
    anomaly_weight: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> ScreeningConfig:
        if abs(self.volatility_weight + self.anomaly_weight - 1.0) > 1e-9:
            raise ValueError("screening weights must sum to 1")
        return self


class MVPConfig(ConfigModel):
    task: TaskConfig
    calendar: dict[str, str]
    data: DataConfig
    features: dict[str, int | str]
    screening: ScreeningConfig
    ranking: RankingConfig
    retrieval: RetrievalConfig
    agents: AgentsConfig
    retry: dict[str, int | list[float]]
    revision: dict[str, int]
    output: dict[str, Path]

    @model_validator(mode="after")
    def validate_loose_sections(self) -> MVPConfig:
        if self.features != {
            "required_complete_sessions": 21,
            "imputation": "none",
        }:
            raise ValueError("MVP features must require 21 sessions with no imputation")
        if set(self.retry) != {
            "technical_max_retries",
            "structured_repair_max_retries",
            "backoff_seconds",
        }:
            raise ValueError("retry config has missing or unknown keys")
        if int(self.retry["technical_max_retries"]) < 0:
            raise ValueError("technical_max_retries must be non-negative")
        if int(self.retry["structured_repair_max_retries"]) not in {0, 1}:
            raise ValueError("structured_repair_max_retries must be 0 or 1")
        backoffs = self.retry["backoff_seconds"]
        if not isinstance(backoffs, list) or not backoffs or any(value < 0 for value in backoffs):
            raise ValueError("backoff_seconds must be a non-empty list of non-negative values")
        if set(self.revision) != {"max_rounds"} or self.revision["max_rounds"] < 0:
            raise ValueError("revision.max_rounds must be non-negative")
        if set(self.output) != {"runs_path"}:
            raise ValueError("output must define exactly runs_path")
        return self


class EvaluationBody(ConfigModel):
    sessions: list[date]
    cutoff_time: str
    timezone: str
    labels: Path
    large_move_threshold: float = Field(gt=0)
    variants: list[str]
    ndcg: dict[str, str | None]


class EvaluationConfig(ConfigModel):
    screening_config: Path
    evaluation: EvaluationBody


def _load_yaml(path: Path) -> dict:
    try:
        value = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError("CONFIG_READ_FAILED", f"Unable to read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigError("CONFIG_INVALID", f"{path} must contain a mapping")
    return value


def load_mvp_config(path: str | Path) -> MVPConfig:
    path = Path(path)
    try:
        return MVPConfig.model_validate(_load_yaml(path))
    except ValueError as exc:
        raise ConfigError("CONFIG_INVALID", str(exc)) from exc


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    path = Path(path)
    try:
        return EvaluationConfig.model_validate(_load_yaml(path))
    except ValueError as exc:
        raise ConfigError("CONFIG_INVALID", str(exc)) from exc


def canonical_json(value: object) -> bytes:
    """Serialize values deterministically for cache keys and audit hashes."""

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()


def content_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
