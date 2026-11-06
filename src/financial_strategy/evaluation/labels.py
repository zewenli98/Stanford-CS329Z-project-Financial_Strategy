"""Shared point-in-time ranking label definitions."""

from __future__ import annotations

import math
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

RankLabelMode = Literal["intraday_range"]


def range_ratio(high: float, low: float) -> float | None:
    """Return the shared regular-session range target, `(high - low) / low`."""

    if not math.isfinite(high) or not math.isfinite(low) or low <= 0 or high < low:
        return None
    return (high - low) / low


def load_rank_labels(
    dataset: Path,
    mode: RankLabelMode = "intraday_range",
) -> dict[str, dict[str, float]]:
    """Load `(high - low) / low` labels without mutating the frozen snapshot."""

    if mode != "intraday_range":
        raise ValueError(f"unknown rank label mode: {mode}")
    panel = pl.read_parquet(dataset / "evaluation/labels.parquet").select(
        "session", "symbol"
    )
    bars = pl.read_parquet(dataset / "market/daily_bars.parquet")
    return intraday_range_labels(bars, panel)


def intraday_range_labels(
    bars: pl.DataFrame,
    panel: pl.DataFrame,
) -> dict[str, dict[str, float]]:
    """Map each evaluation panel row to `(high - low) / low` from frozen bars."""

    required = {"session", "symbol", "high", "low"}
    if not required <= set(bars.columns):
        raise ValueError("intraday range labels require session, symbol, high, and low")
    if not {"session", "symbol"} <= set(panel.columns):
        raise ValueError("label panel requires session and symbol")
    allowed = {
        (_session_text(row["session"]), row["symbol"])
        for row in panel.select("session", "symbol").iter_rows(named=True)
    }
    labels: dict[str, dict[str, float]] = {}
    for row in bars.select("session", "symbol", "high", "low").iter_rows(named=True):
        session = _session_text(row["session"])
        symbol = row["symbol"]
        if (session, symbol) not in allowed:
            continue
        try:
            ratio = range_ratio(float(row["high"]), float(row["low"]))
        except (TypeError, ValueError):
            continue
        if ratio is None:
            continue
        labels.setdefault(session, {})[symbol] = ratio
    return labels


def _session_text(value: date | datetime | str) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)
