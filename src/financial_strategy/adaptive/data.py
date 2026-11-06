"""Point-in-time range features and separate, explicitly covered range labels.

All range quantities are dimensionless ratios, not percentage points. This module
never fills a missing target-session high/low with an open/close proxy.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field

from financial_strategy.errors import DataError
from financial_strategy.features.engine import compute_features
from financial_strategy.schemas import FeatureRecord
from financial_strategy.schemas.models import StrictModel


class RangeFeature(StrictModel):
    feature_id: str
    symbol: str
    snapshot_id: str
    mean_range_20d: float = Field(ge=0, allow_inf_nan=False)
    median_range_20d: float = Field(ge=0, allow_inf_nan=False)
    latest_range: float = Field(ge=0, allow_inf_nan=False)
    historical_context: FeatureRecord


def compute_range_features(
    bars: pl.DataFrame,
    expected_sessions: list[date],
    required_sessions: int = 21,
) -> tuple[dict[str, RangeFeature], dict[str, str]]:
    """Build history-only features from a gateway-filtered, complete daily history.

    The caller must supply the point-in-time gateway's bars, not unfiltered source
    bars. Existing market-context features preserve their original definitions.
    Twenty high/low ranges use the last twenty completed sessions; the extra session
    is required by the existing close-to-close volatility calculation.
    """
    if required_sessions < 21:
        raise ValueError("Range features require at least 21 completed sessions")
    if len(expected_sessions) < required_sessions:
        raise ValueError("expected_sessions must contain the required completed sessions")
    context, rejected = compute_features(bars, expected_sessions, required_sessions)
    features: dict[str, RangeFeature] = {}
    for symbol, historical in context.items():
        rows = bars.filter(pl.col("symbol") == symbol).sort("session").tail(20)
        highs = rows["high"].to_numpy().astype(float)
        lows = rows["low"].to_numpy().astype(float)
        if (
            not np.isfinite(highs).all()
            or not np.isfinite(lows).all()
            or np.any(lows <= 0)
            or np.any(highs < lows)
        ):
            rejected[symbol] = "INVALID_HIGH_LOW_RANGE"
            continue
        ranges = (highs - lows) / lows
        features[symbol] = RangeFeature(
            feature_id=f"{historical.snapshot_id}:{symbol}:range_features_v1",
            symbol=symbol,
            snapshot_id=historical.snapshot_id,
            mean_range_20d=float(np.mean(ranges)),
            median_range_20d=float(np.median(ranges)),
            latest_range=float(ranges[-1]),
            historical_context=historical,
        )
    return features, rejected


def baseline_predictions(
    features: dict[str, RangeFeature],
    method: Literal["mean_range_20d", "median_range_20d", "latest_range"] = "mean_range_20d",
) -> dict[str, float]:
    """Return range-valued predictions; no cross-sectional percentile rescaling."""
    if method not in {"mean_range_20d", "median_range_20d", "latest_range"}:
        raise ValueError(f"Unsupported range baseline: {method}")
    return {symbol: float(getattr(item, method)) for symbol, item in sorted(features.items())}


def derive_range_labels(
    bars_or_path: pl.DataFrame | str | Path,
    sessions: list[date],
    symbols: list[str] | None = None,
) -> tuple[pl.DataFrame, dict]:
    """Derive labels only from existing source high/low, reporting missing coverage.

    Call only after publishing predictions. Source bars are read without mutation.
    Unlike feature construction, the evaluator intentionally selects target-session
    bars; no such row may be passed back into the prediction workflow.
    """
    bars = bars_or_path if isinstance(bars_or_path, pl.DataFrame) else pl.read_parquet(bars_or_path)
    requested = sorted(set(sessions))
    required = {"symbol", "session", "high", "low"}
    if not required <= set(bars.columns):
        raise DataError("RANGE_LABEL_COLUMNS_MISSING", "Range labels require session/high/low")
    universe = sorted(set(symbols if symbols is not None else bars["symbol"].to_list()))
    allowed = set(universe)
    found: dict[tuple[date, str], dict] = {}
    seen: set[tuple[date, str]] = set()
    invalid: list[dict] = []
    for row in bars.filter(pl.col("session").is_in(requested)).to_dicts():
        if row["symbol"] not in allowed:
            continue
        key = (row["session"], row["symbol"])
        if key in seen:
            raise DataError("DUPLICATE_RANGE_LABEL", f"Duplicate target bar: {key}")
        seen.add(key)
        try:
            high, low = float(row["high"]), float(row["low"])
        except (TypeError, ValueError):
            invalid.append({"session": row["session"].isoformat(), "symbol": row["symbol"]})
            continue
        if not math.isfinite(high) or not math.isfinite(low) or low <= 0 or high < low:
            invalid.append({"session": row["session"].isoformat(), "symbol": row["symbol"]})
            continue
        found[key] = {
            "session": row["session"],
            "symbol": row["symbol"],
            "high": high,
            "low": low,
            "range": (high - low) / low,
        }
    coverage = {
        "target": "(high-low)/low",
        "source": "existing_frozen_market_bars",
        "requested_session_count": len(requested),
        "requested_symbol_count": len(universe),
        "expected_label_count": len(requested) * len(universe),
        "available_label_count": len(found),
        "missing_sessions": [
            session.isoformat()
            for session in requested
            if not any(key[0] == session for key in found)
        ],
        "missing_by_session": {
            session.isoformat(): [symbol for symbol in universe if (session, symbol) not in found]
            for session in requested
            if any((session, symbol) not in found for symbol in universe)
        },
        "invalid_bars": invalid,
        "note": "Missing high/low labels are excluded explicitly; open/close is never substituted.",
    }
    frame = pl.DataFrame(
        [found[key] for key in sorted(found)],
        schema={
            "session": pl.Date,
            "symbol": pl.String,
            "high": pl.Float64,
            "low": pl.Float64,
            "range": pl.Float64,
        },
    )
    return frame, coverage
