"""Point-in-time filtering, split adjustment, and immutable snapshot preparation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from importlib.metadata import version
from uuid import uuid4

import polars as pl

from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.data.providers import EvidenceProvider, MarketDataProvider
from financial_strategy.errors import DataError
from financial_strategy.schemas import (
    CorporateActionRecord,
    EvidenceRecord,
    MacroEvidenceRecord,
    MarketBar,
    ScreeningTask,
    SymbolMacroExposure,
)


@dataclass(frozen=True)
class PointInTimeSnapshot:
    snapshot_id: str
    bars: pl.DataFrame
    company_evidence: tuple[EvidenceRecord, ...]
    macro_evidence: tuple[MacroEvidenceRecord, ...]
    corporate_actions: tuple[CorporateActionRecord, ...]
    exposures: tuple[SymbolMacroExposure, ...]
    calendar_schedule: tuple[dict, ...]
    calendar_version: str


class PointInTimeGateway:
    """The only component allowed to decide whether an input was available at cutoff."""

    def __init__(
        self,
        market: MarketDataProvider,
        evidence: EvidenceProvider,
        calendar: TradingCalendar,
        input_adjustment_policy: str | None = None,
    ) -> None:
        self.market = market
        self.evidence = evidence
        self.calendar = calendar
        self.input_adjustment_policy = input_adjustment_policy

    def freeze_inputs(
        self,
        task: ScreeningTask,
        exposures: list[SymbolMacroExposure],
    ) -> PointInTimeSnapshot:
        snapshot_id = f"snapshot_{uuid4().hex}"
        sessions = self.calendar.completed_sessions_before(
            task.decision_cutoff, task.price_lookback_sessions
        )
        required_sessions = set(sessions)
        raw = self.market.bars()
        records = raw.to_dicts()

        actions = tuple(self._eligible_actions(task, self.market.corporate_actions()))
        adjusted_rows: list[dict] = []
        source_adjustment_policy: str | None = None
        for row in records:
            if row["session"] not in required_sessions:
                continue
            if row["available_at"] > task.decision_cutoff:
                continue
            if row["bar_end"] >= task.decision_cutoff:
                continue
            row_policy = row["adjustment_policy"]
            if row_policy not in {"raw_unadjusted", "split_adjusted"}:
                raise DataError(
                    "MIXED_ADJUSTMENT_POLICY",
                    "source bars must be raw_unadjusted or split_adjusted",
                )
            if (
                self.input_adjustment_policy is not None
                and row_policy != self.input_adjustment_policy
            ):
                raise DataError(
                    "ADJUSTMENT_POLICY_MISMATCH",
                    f"configured {self.input_adjustment_policy}, found {row_policy}",
                )
            if source_adjustment_policy is None:
                source_adjustment_policy = row_policy
            elif row_policy != source_adjustment_policy:
                raise DataError(
                    "MIXED_ADJUSTMENT_POLICY",
                    "all source bars must use one adjustment policy",
                )
            adjusted = (
                self._apply_split_adjustments(row, actions)
                if row_policy == "raw_unadjusted"
                else dict(row)
            )
            adjusted["adjustment_policy"] = "split_adjusted"
            adjusted["snapshot_id"] = snapshot_id
            # Pydantic enforces OHLC consistency, finite positive prices, and zero-volume policy.
            validated = MarketBar.model_validate(adjusted)
            adjusted_rows.append(validated.model_dump())

        company = tuple(
            self._dedupe(
                [
                    item
                    for item in self.evidence.company_evidence()
                    if item.available_at <= task.decision_cutoff
                    and item.available_at
                    >= task.decision_cutoff - timedelta(days=task.news_lookback_days)
                ]
            )
        )
        macro = tuple(
            self._dedupe(
                [
                    item
                    for item in self.evidence.macro_evidence()
                    if item.available_at <= task.decision_cutoff
                ]
            )
        )
        schedule = tuple(self.calendar.schedule_snapshot(sessions[0], task.target_session))
        return PointInTimeSnapshot(
            snapshot_id=snapshot_id,
            bars=pl.DataFrame(adjusted_rows),
            company_evidence=company,
            macro_evidence=macro,
            corporate_actions=actions,
            exposures=tuple(exposures),
            calendar_schedule=schedule,
            calendar_version=version("exchange-calendars"),
        )

    @staticmethod
    def _eligible_actions(
        task: ScreeningTask,
        actions: list[CorporateActionRecord],
    ) -> list[CorporateActionRecord]:
        return [
            action
            for action in actions
            if action.action_type in {"split", "reverse_split"}
            and action.available_at <= task.decision_cutoff
            and action.effective_at <= task.decision_cutoff
        ]

    @staticmethod
    def _apply_split_adjustments(
        row: dict,
        actions: tuple[CorporateActionRecord, ...],
    ) -> dict:
        adjusted = dict(row)
        ratio = 1.0
        for action in actions:
            if (
                action.permanent_entity_id == row["permanent_entity_id"]
                and row["bar_end"] < action.effective_at
            ):
                # ratio is new shares / old shares. Price uses the inverse; volume uses ratio.
                ratio *= action.ratio
        if ratio != 1.0:
            for field in ("open", "high", "low", "close"):
                adjusted[field] = adjusted[field] / ratio
            adjusted["volume"] = adjusted["volume"] * ratio
        return adjusted

    @staticmethod
    def _dedupe(items: list[EvidenceRecord]) -> list:
        by_hash: dict[str, EvidenceRecord] = {}
        for item in sorted(items, key=lambda value: (value.available_at, value.evidence_id)):
            by_hash.setdefault(item.content_hash, item)
        return list(by_hash.values())
