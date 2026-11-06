"""Recomputable feature formulas with explicit input lineage."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from financial_strategy.schemas import FeatureRecord

FORMULA_VERSION = "market_features_v2"


def compute_features(
    bars: pl.DataFrame,
    expected_sessions: list[date],
    required_sessions: int = 21,
    symbol_sectors: dict[str, str] | None = None,
) -> tuple[dict[str, FeatureRecord], dict[str, str]]:
    """Compute all scoring features or reject a symbol without partial-weight fallback."""

    expected = expected_sessions[-required_sessions:]
    features: dict[str, FeatureRecord] = {}
    rejected: dict[str, str] = {}

    if bars.is_empty():
        return features, rejected

    pending: dict[str, dict] = {}
    for symbol in bars.get_column("symbol").unique().sort().to_list():
        frame = bars.filter(pl.col("symbol") == symbol).sort("session")
        rows = frame.to_dicts()
        required_rows = rows[-required_sessions:]
        actual_sessions = [row["session"] for row in required_rows]
        if len(required_rows) < required_sessions:
            rejected[symbol] = "INSUFFICIENT_PRICE_HISTORY"
            continue
        if actual_sessions != expected:
            rejected[symbol] = "MISSING_REQUIRED_SESSION"
            continue

        opens = np.asarray([row["open"] for row in rows], dtype=float)
        highs = np.asarray([row["high"] for row in rows], dtype=float)
        lows = np.asarray([row["low"] for row in rows], dtype=float)
        closes = np.asarray([row["close"] for row in rows], dtype=float)
        volumes = np.asarray([row["volume"] for row in rows], dtype=float)
        finite_inputs = (
            np.isfinite(opens).all()
            and np.isfinite(highs).all()
            and np.isfinite(lows).all()
            and np.isfinite(closes).all()
            and np.isfinite(volumes).all()
        )
        if not finite_inputs:
            rejected[symbol] = "INVALID_OHLCV"
            continue

        daily_simple_returns = closes[1:] / closes[:-1] - 1.0
        abs_oc = np.abs((closes[-20:] - opens[-20:]) / opens[-20:])
        denominator = float(np.median(volumes[-21:-1]))
        if denominator <= 0:
            rejected[symbol] = "INVALID_RELATIVE_VOLUME_DENOMINATOR"
            continue

        ma20 = float(np.mean(closes[-20:]))
        volume_history = volumes[-21:-1]
        volume_median = float(np.median(volume_history))
        volume_mad = float(np.median(np.abs(volume_history - volume_median)))
        robust_volume_zscore = (
            float((volumes[-1] - volume_median) / (1.4826 * volume_mad))
            if volume_mad > 0
            else 0.0
        )
        true_ranges = np.maximum.reduce(
            (
                highs[1:] - lows[1:],
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1]),
            )
        )
        volatility_20 = float(np.std(daily_simple_returns[-20:], ddof=1))
        volatility_5 = _sample_std(daily_simple_returns, 5)
        pending[symbol] = {
            "rows": rows,
            "required_sessions": actual_sessions,
            "returns": daily_simple_returns,
            "values": {
                "median_abs_oc_return_20d": float(np.median(abs_oc)),
                "realized_volatility_20d": volatility_20,
                "return_5d": float(closes[-1] / closes[-6] - 1.0),
                "previous_session_relative_volume": float(volumes[-1] / denominator),
                "distance_to_ma20": float((closes[-1] - ma20) / ma20),
                "realized_volatility_5d": volatility_5,
                "realized_volatility_10d": _sample_std(daily_simple_returns, 10),
                "realized_volatility_60d": _sample_std(daily_simple_returns, 60),
                "median_abs_oc_return_5d": _median_abs_oc(opens, closes, 5),
                "median_abs_oc_return_10d": _median_abs_oc(opens, closes, 10),
                "median_abs_oc_return_60d": _median_abs_oc(opens, closes, 60),
                "median_range_20d": float(
                    np.median((highs[-20:] - lows[-20:]) / opens[-20:])
                ),
                "atr_14d": (
                    float(np.mean(true_ranges[-14:]) / closes[-1])
                    if len(true_ranges) >= 14
                    else None
                ),
                "robust_volume_zscore_20d": robust_volume_zscore,
                "momentum_10d": _return_horizon(closes, 10),
                "momentum_20d": _return_horizon(closes, 20),
                "reversal_1d": (
                    float(-daily_simple_returns[-1]) if len(daily_simple_returns) else None
                ),
                "volatility_trend_5_20": (
                    volatility_5 / volatility_20
                    if volatility_5 is not None and volatility_20 > 0
                    else 0.0
                ),
            },
        }

    returns_by_session: dict[date, list[float]] = {}
    for item in pending.values():
        sessions = [row["session"] for row in item["rows"]][1:]
        for session, value in zip(sessions, item["returns"], strict=True):
            returns_by_session.setdefault(session, []).append(float(value))
    market_returns = {
        session: float(np.median(values))
        for session, values in returns_by_session.items()
    }
    industry_returns: dict[tuple[str, date], float] = {}
    if symbol_sectors:
        grouped: dict[tuple[str, date], list[float]] = {}
        for symbol, item in pending.items():
            sector = symbol_sectors.get(symbol)
            if sector is None:
                continue
            sessions = [row["session"] for row in item["rows"]][1:]
            for session, value in zip(sessions, item["returns"], strict=True):
                grouped.setdefault((sector, session), []).append(float(value))
        industry_returns = {
            key: float(np.median(values)) for key, values in grouped.items()
        }

    for symbol, item in pending.items():
        rows = item["rows"]
        sessions = [row["session"] for row in rows][1:]
        residuals = np.asarray(
            [
                value - market_returns[session]
                for session, value in zip(sessions, item["returns"], strict=True)
            ],
            dtype=float,
        )
        values = dict(item["values"])
        values["market_residual_volatility_20d"] = _sample_std(residuals, 20)
        sector = symbol_sectors.get(symbol) if symbol_sectors else None
        has_industry_history = sector is not None and all(
            (sector, session) in industry_returns for session in sessions
        )
        if has_industry_history:
            industry_residuals = np.asarray(
                [
                    value - industry_returns[(sector, session)]
                    for session, value in zip(
                        sessions,
                        item["returns"],
                        strict=True,
                    )
                ],
                dtype=float,
            )
            values["industry_residual_volatility_20d"] = _sample_std(
                industry_residuals,
                20,
            )
        else:
            values["industry_residual_volatility_20d"] = None
        values["day_of_week"] = rows[-1]["session"].weekday()
        values["missing_feature_count"] = sum(
            value is None for value in values.values()
        )
        features[symbol] = FeatureRecord(
            feature_id=f"{rows[-1]['snapshot_id']}:{symbol}:{FORMULA_VERSION}",
            symbol=symbol,
            snapshot_id=rows[-1]["snapshot_id"],
            formula_version=FORMULA_VERSION,
            **values,
            input_columns=["open", "high", "low", "close", "volume"],
            input_sessions=item["required_sessions"],
        )

    return features, rejected


def _sample_std(values: np.ndarray, window: int) -> float | None:
    if len(values) < window:
        return None
    return float(np.std(values[-window:], ddof=1))


def _median_abs_oc(opens: np.ndarray, closes: np.ndarray, window: int) -> float | None:
    if len(opens) < window:
        return None
    return float(np.median(np.abs((closes[-window:] - opens[-window:]) / opens[-window:])))


def _return_horizon(closes: np.ndarray, window: int) -> float | None:
    if len(closes) <= window:
        return None
    return float(closes[-1] / closes[-window - 1] - 1.0)
