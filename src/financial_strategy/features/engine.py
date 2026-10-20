"""Recomputable feature formulas with explicit input lineage."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from financial_strategy.schemas import FeatureRecord

FORMULA_VERSION = "mvp_features_v1"


def compute_features(
    bars: pl.DataFrame,
    expected_sessions: list[date],
    required_sessions: int = 21,
) -> tuple[dict[str, FeatureRecord], dict[str, str]]:
    """Compute all scoring features or reject a symbol without partial-weight fallback."""

    expected = expected_sessions[-required_sessions:]
    features: dict[str, FeatureRecord] = {}
    rejected: dict[str, str] = {}

    if bars.is_empty():
        return features, rejected

    for symbol in bars.get_column("symbol").unique().sort().to_list():
        frame = bars.filter(pl.col("symbol") == symbol).sort("session")
        rows = frame.tail(required_sessions).to_dicts()
        actual_sessions = [row["session"] for row in rows]
        if len(rows) < required_sessions:
            rejected[symbol] = "INSUFFICIENT_PRICE_HISTORY"
            continue
        if actual_sessions != expected:
            rejected[symbol] = "MISSING_REQUIRED_SESSION"
            continue

        opens = np.asarray([row["open"] for row in rows], dtype=float)
        closes = np.asarray([row["close"] for row in rows], dtype=float)
        volumes = np.asarray([row["volume"] for row in rows], dtype=float)
        finite_inputs = (
            np.isfinite(opens).all()
            and np.isfinite(closes).all()
            and np.isfinite(volumes).all()
        )
        if not finite_inputs:
            rejected[symbol] = "INVALID_OHLCV"
            continue

        daily_simple_returns = closes[1:] / closes[:-1] - 1.0
        abs_oc = np.abs((closes[-20:] - opens[-20:]) / opens[-20:])
        denominator = float(np.median(volumes[:-1]))
        if denominator <= 0:
            rejected[symbol] = "INVALID_RELATIVE_VOLUME_DENOMINATOR"
            continue

        ma20 = float(np.mean(closes[-20:]))
        record = FeatureRecord(
            feature_id=f"{rows[-1]['snapshot_id']}:{symbol}:{FORMULA_VERSION}",
            symbol=symbol,
            snapshot_id=rows[-1]["snapshot_id"],
            formula_version=FORMULA_VERSION,
            median_abs_oc_return_20d=float(np.median(abs_oc)),
            realized_volatility_20d=float(np.std(daily_simple_returns, ddof=1)),
            return_5d=float(closes[-1] / closes[-6] - 1.0),
            previous_session_relative_volume=float(volumes[-1] / denominator),
            distance_to_ma20=float((closes[-1] - ma20) / ma20),
            input_columns=["open", "close", "volume"],
            input_sessions=actual_sessions,
        )
        features[symbol] = record

    return features, rejected
