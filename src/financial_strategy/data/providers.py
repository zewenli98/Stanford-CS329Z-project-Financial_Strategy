"""Provider protocols and local file implementations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import polars as pl

from financial_strategy.schemas import CorporateActionRecord, EvidenceRecord, MacroEvidenceRecord


class MarketDataProvider(Protocol):
    def bars(self) -> pl.DataFrame: ...

    def corporate_actions(self) -> list[CorporateActionRecord]: ...


class EvidenceProvider(Protocol):
    def company_evidence(self) -> list[EvidenceRecord]: ...

    def macro_evidence(self) -> list[MacroEvidenceRecord]: ...


class UniverseProvider(Protocol):
    def universe(self) -> dict: ...

    def sector_exposures(self) -> dict: ...


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class LocalMarketDataProvider:
    def __init__(self, bars_path: Path, actions_path: Path) -> None:
        self.bars_path = bars_path
        self.actions_path = actions_path

    def bars(self) -> pl.DataFrame:
        return pl.read_parquet(self.bars_path)

    def corporate_actions(self) -> list[CorporateActionRecord]:
        return [
            CorporateActionRecord.model_validate(item) for item in _read_jsonl(self.actions_path)
        ]


class LocalEvidenceProvider:
    def __init__(self, company_path: Path, macro_path: Path) -> None:
        self.company_path = company_path
        self.macro_path = macro_path

    def company_evidence(self) -> list[EvidenceRecord]:
        return [EvidenceRecord.model_validate(item) for item in _read_jsonl(self.company_path)]

    def macro_evidence(self) -> list[MacroEvidenceRecord]:
        return [
            MacroEvidenceRecord.model_validate(item) for item in _read_jsonl(self.macro_path)
        ]


class LocalUniverseProvider:
    def __init__(self, universe_path: Path, exposures_path: Path) -> None:
        self.universe_path = universe_path
        self.exposures_path = exposures_path

    def universe(self) -> dict:
        return json.loads(self.universe_path.read_text())

    def sector_exposures(self) -> dict:
        return json.loads(self.exposures_path.read_text())
