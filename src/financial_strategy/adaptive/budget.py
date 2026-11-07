"""Atomic dollar reservations and per-attempt metering, including cached responses."""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict
from threading import RLock, local
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from financial_strategy.adaptive.config import AdaptiveConfig, ModelPrice
from financial_strategy.config import canonical_json
from financial_strategy.errors import StrategyError
from financial_strategy.llm.client import (
    ControlledClient,
    LiveClient,
    RecordedClient,
    ScriptedClient,
)


class BudgetLedger:
    """Hold worst-case charges before dispatch; unused reservations are refundable."""

    def __init__(self, cap: float):
        if not math.isfinite(cap) or cap <= 0:
            raise ValueError("budget cap must be positive and finite")
        self.cap = cap
        self.spent = 0.0
        self.actual = 0.0
        self.holds: dict[str, float] = {}
        self.events: list[dict] = []
        self.lock = RLock()
        self.blocked = False

    @property
    def remaining(self) -> float:
        with self.lock:
            return (
                0.0 if self.blocked else max(0.0, self.cap - self.spent - sum(self.holds.values()))
            )

    def reserve(self, amount: float, *, parent: str | None = None, label: str = "") -> str:
        with self.lock:
            if not math.isfinite(amount) or amount < 0:
                raise ValueError("reservation must be finite and nonnegative")
            available = self.holds.get(parent, 0.0) if parent else self.remaining
            if self.blocked or amount > available + 1e-12:
                raise StrategyError("TASK_DOLLAR_BUDGET_EXHAUSTED", label)
            if parent:
                self.holds[parent] -= amount
            token = uuid4().hex
            self.holds[token] = amount
            self.events.append(
                {
                    "event": "reserve",
                    "id": token,
                    "parent": parent,
                    "label": label,
                    "reserved_usd": amount,
                }
            )
            return token

    def settle(
        self, token: str, normalized: float, actual: float = 0.0, *, parent: str | None = None
    ) -> None:
        with self.lock:
            if not math.isfinite(normalized) or not math.isfinite(actual) or actual < 0:
                self.blocked = True
                raise ValueError("settlement charges must be finite and nonnegative")
            reserved = self.holds.pop(token)
            if normalized > reserved + 1e-9 or normalized < 0:
                # Keep the conservative bound spent and stop, never admit more work.
                self.spent += reserved
                self.actual += actual
                self.blocked = True
                raise StrategyError("COST_BOUND_VIOLATION", "provider usage exceeds reservation")
            self.spent += normalized
            self.actual += actual
            if parent in self.holds:
                self.holds[parent] += reserved - normalized
            self.events.append(
                {"event": "settle", "id": token, "normalized_usd": normalized, "actual_usd": actual}
            )

    def release(self, token: str) -> None:
        with self.lock:
            amount = self.holds.pop(token, 0.0)
            self.events.append({"event": "release", "id": token, "released_usd": amount})

    def summary(self) -> dict:
        with self.lock:
            return {
                "cap_usd": self.cap,
                "normalized_uncached_usd": self.spent,
                "actual_usd": self.actual,
                "reserved_usd": sum(self.holds.values()),
                "remaining_usd": self.remaining,
                "blocked": self.blocked,
            }


class MeteredClient:
    """Wrap the individual attempt, inside ControlledClient's retry loop."""

    def __init__(self, base, config: AdaptiveConfig, ledger: BudgetLedger):
        self.base, self.config, self.ledger = base, config, ledger
        self.context = local()
        self.records: list[dict] = []
        self.lock = RLock()
        self.replaying = getattr(base, "is_replay", False)
        self.worker_attempts: dict[str, int] = {}

    def price(self, role: str) -> ModelPrice:
        model = self.config.agents.model_for(role)
        return self.config.adaptive.prices.get(model, self.config.adaptive.offline_price)

    def bound(self, role: str) -> float:
        # At most one token per UTF-8 byte for the supported byte-based models;
        # schema and framing are additionally bounded and checked below.
        price = self.price(role)
        return (
            (self.config.adaptive.max_input_bytes + 32768) * price.input_per_million / 1e6
            + self.config.agents.roles[role].max_tokens * price.output_per_million / 1e6
            + price.request_usd
        )

    @contextmanager
    def using(self, reservation: str):
        previous = getattr(self.context, "parent", None)
        self.context.parent = reservation
        try:
            yield
        finally:
            self.context.parent = previous

    def generate(
        self, role: str, payload: dict, schema: type[BaseModel], *, attempt_type: str = "initial"
    ):
        payload_bytes = len(canonical_json(payload))
        schema_bytes = len(canonical_json(schema.model_json_schema()))
        if payload_bytes > self.config.adaptive.max_input_bytes or schema_bytes > 24576:
            raise StrategyError("INPUT_LIMIT_EXCEEDED", "bounded request context exceeded")
        parent = getattr(self.context, "parent", None)
        if parent is not None and role == "company_evidence":
            with self.lock:
                attempts = self.worker_attempts.get(parent, 0)
                if attempts >= self.config.adaptive.max_worker_model_calls:
                    raise StrategyError("WORKER_MODEL_LIMIT", "assignment attempt limit reached")
                self.worker_attempts[parent] = attempts + 1
        amount = self.bound(role)
        token = self.ledger.reserve(amount, parent=parent, label=role)
        record: dict[str, Any] = {
            "role": role,
            "payload": payload,
            "schema_name": schema.__name__,
            "attempt_type": attempt_type,
        }
        try:
            result = self.base.generate(role, payload, schema, attempt_type=attempt_type)
        except Exception as exc:
            # A transport failure can have been billed. Preserve its entire charge bound.
            charge = amount if self.config.agents.mode == "live" else 0.0
            actual = charge if not self.replaying else 0.0
            self.ledger.settle(token, charge, actual, parent=parent)
            record.update(
                status="failed",
                error_code=getattr(exc, "code", type(exc).__name__),
                error_has_code=hasattr(exc, "code"),
                error_type=type(exc).__name__,
                error_message=str(exc),
                normalized_usd=charge,
                actual_usd=actual,
                usage_estimated=True,
            )
            if hasattr(exc, "errors"):
                record["validation_errors"] = exc.errors(include_context=False)
            with self.lock:
                self.records.append(record)
            raise
        price = self.price(role)
        if result.input_tokens or result.output_tokens:
            charge = (
                result.input_tokens * price.input_per_million
                + result.output_tokens * price.output_per_million
            ) / 1e6 + price.request_usd
            estimated = False
        elif self.config.agents.mode == "live":
            # Missing usage is not permission to treat a response/cache hit as free.
            charge, estimated = amount, True
        else:
            # Explicit synthetic cost reference for offline policy/budget integration tests.
            input_estimate = (payload_bytes + schema_bytes + 2048 + 3) // 4
            output_estimate = (len(canonical_json(result.raw)) + 3) // 4
            charge = (
                input_estimate * price.input_per_million
                + output_estimate * price.output_per_million
            ) / 1e6 + price.request_usd
            charge, estimated = min(charge, amount), True
        actual = charge if result.source == "live" and not self.replaying else 0.0
        record.update(
            status="completed",
            result=asdict(result),
            normalized_usd=charge,
            actual_usd=actual,
            usage_estimated=estimated,
        )
        try:
            self.ledger.settle(token, charge, actual, parent=parent)
        except StrategyError as exc:
            record.update(status="cost_bound_violation", error_code=exc.code)
            with self.lock:
                self.records.append(record)
            raise
        # Pydantic objects are serialized by the audit writer.
        with self.lock:
            self.records.append(record)
        return result


class AdaptiveClient:
    def __init__(self, config: AdaptiveConfig, responder, ledger: BudgetLedger, base=None):
        if base is not None:
            pass
        elif config.agents.mode == "scripted":
            base = ScriptedClient(config.agents, responder)
        elif config.agents.mode == "recorded":
            base = RecordedClient(config.agents)
        else:
            base = LiveClient(config.agents)
        self.meter = MeteredClient(base, config, ledger)
        self.controlled = ControlledClient(self.meter, config.agents, config.retry)
        self.config = config

    def generate(self, *args, **kwargs):
        if len(args) >= 2:
            args = (args[0], {**args[1], "_workflow": "adaptive_range_v1"}, *args[2:])
        else:
            kwargs["payload"] = {**kwargs["payload"], "_workflow": "adaptive_range_v1"}
        return self.controlled.generate(*args, **kwargs)

    def bound(self, role: str) -> float:
        multiplier = (
            1
            + int(self.config.retry["technical_max_retries"])
            + int(self.config.retry["structured_repair_max_retries"])
        )
        if self.config.agents.mode != "live":
            multiplier = 1 + int(self.config.retry["structured_repair_max_retries"])
        return multiplier * self.meter.bound(role)

    def assignment_bound(self) -> float:
        return self.config.adaptive.max_worker_model_calls * self.meter.bound(
            "company_evidence"
        ) + 2 * self.bound("verifier")

    def using(self, reservation):
        return self.meter.using(reservation)

    @property
    def attempt_log(self):
        return self.controlled.attempt_log
