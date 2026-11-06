"""Strict configuration for the opt-in adaptive range workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from financial_strategy.config import MVPConfig, _load_yaml
from financial_strategy.errors import ConfigError

Policy = Literal[
    "scenario", "uniform", "random", "boundary", "lexicographic", "baseline", "history_only"
]


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)
    request_usd: float = Field(default=0, ge=0)


class AdaptiveSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    policy: Policy = "scenario"
    daily_budget_usd: float = Field(default=2.0, gt=0)
    max_workers: int = Field(default=2, ge=1, le=8)
    max_batches: int = Field(default=8, ge=0)
    exploration_probability: float = Field(default=0.2, ge=0, le=1)
    max_worker_model_calls: int = Field(default=3, ge=1)
    max_worker_tool_calls: int = Field(default=4, ge=1)
    seed: int = 0
    replicate_id: str = Field(default="0", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    memory_days: int = Field(default=7, ge=1)
    memory_path: Path | None = None
    max_input_bytes: int = Field(default=60000, ge=4000)
    research_candidate_limit: int | None = Field(default=None, ge=1)
    compact_research_prompts: bool = False
    research_protocol: Literal["legacy", "guided", "packet"] = "legacy"
    candidate_method: Literal["volatility", "hybrid", "event_uncertainty"] = "volatility"
    baseline_method: Literal["mean20", "mean5", "mean10", "median20", "ewma10"] = "mean20"
    evidence_previous_day_only: bool = False
    forecast_blend: float = Field(default=1.0, ge=0, le=1)
    prices: dict[str, ModelPrice] = Field(default_factory=dict)
    pricing_as_of: str | None = None
    # Offline simulations use explicit reference prices, never claims about actual provider rates.
    offline_price: ModelPrice = Field(
        default_factory=lambda: ModelPrice(input_per_million=1, output_per_million=4)
    )


class AdaptiveConfig(MVPConfig):
    workflow: Literal["adaptive_range_v1"] = "adaptive_range_v1"
    adaptive: AdaptiveSettings = Field(default_factory=AdaptiveSettings)

    @model_validator(mode="after")
    def validate_live_prices(self):
        if self.agents.mode == "live" and self.adaptive.policy != "baseline":
            models = {self.agents.model_for(role) for role in self.agents.roles}
            if not self.adaptive.pricing_as_of or models - self.adaptive.prices.keys():
                raise ValueError("live adaptive runs require dated prices for every exact model")
        return self


def load_adaptive_config(path: str | Path) -> AdaptiveConfig:
    path = Path(path)
    try:
        return AdaptiveConfig.model_validate(_load_yaml(path))
    except ValueError as exc:
        raise ConfigError("CONFIG_INVALID", str(exc)) from exc


def is_adaptive_config(path: str | Path) -> bool:
    return _load_yaml(Path(path)).get("workflow") == "adaptive_range_v1"
