"""Small deterministic rank learners used inside time-isolated training folds."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def transform_rank_target(
    values: np.ndarray,
    scheme: str,
    *,
    large_move_threshold: float = 0.04,
) -> np.ndarray:
    """Apply one of the four pre-registered absolute-return target definitions."""

    target = np.asarray(values, dtype=float)
    if scheme == "raw_magnitude":
        return target.copy()
    if scheme == "winsorized_magnitude":
        lower, upper = np.quantile(target, [0.01, 0.99])
        return np.clip(target, lower, upper)
    if scheme == "session_percentile":
        order = np.argsort(target, kind="stable")
        ranks = np.empty(len(target), dtype=float)
        ranks[order] = np.arange(len(target), dtype=float)
        return ranks / max(1, len(target) - 1)
    if scheme == "large_move_then_magnitude":
        magnitude_scale = max(float(target.max(initial=0.0)), large_move_threshold)
        return (target >= large_move_threshold).astype(float) + target / magnitude_scale
    raise ValueError(f"unknown rank target scheme: {scheme}")


def fit_nonnegative_linear_ranker(
    features: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    *,
    shrinkage: float = 0.1,
) -> np.ndarray:
    """Fit nonnegative simplex weights with L2 shrinkage toward an anchor."""

    x, y, prior = _validated_inputs(features, target, anchor)

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.mean(residual**2) + shrinkage * np.sum((weights - prior) ** 2))

    return _solve_simplex(objective, prior)


def fit_pairwise_ranker(
    features: np.ndarray,
    target: np.ndarray,
    session_ids: np.ndarray,
    anchor: np.ndarray,
    *,
    shrinkage: float = 0.1,
    maximum_pairs_per_session: int = 5000,
) -> np.ndarray:
    """Fit a session-grouped pairwise logistic ranker on a nonnegative simplex."""

    x, y, prior = _validated_inputs(features, target, anchor)
    sessions = np.asarray(session_ids)
    if sessions.ndim != 1 or len(sessions) != len(y):
        raise ValueError("session_ids must contain one value per feature row")
    differences: list[np.ndarray] = []
    for session in np.unique(sessions):
        indices = np.flatnonzero(sessions == session)
        added = 0
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                if y[left] == y[right]:
                    continue
                winner, loser = (left, right) if y[left] > y[right] else (right, left)
                differences.append(x[winner] - x[loser])
                added += 1
                if added >= maximum_pairs_per_session:
                    break
            if added >= maximum_pairs_per_session:
                break
    if not differences:
        return prior.copy()
    pair_differences = np.asarray(differences, dtype=float)

    def objective(weights: np.ndarray) -> float:
        margins = pair_differences @ weights
        logistic_loss = np.logaddexp(0.0, -margins).mean()
        return float(logistic_loss + shrinkage * np.sum((weights - prior) ** 2))

    return _solve_simplex(objective, prior)


def fit_signed_ridge_residual(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    shrinkage: float = 1.0,
    coefficient_bound: float = 0.25,
) -> tuple[float, np.ndarray]:
    """Fit a bounded signed correction without changing the frozen market model."""

    x = np.asarray(features, dtype=float)
    y = np.asarray(targets, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or not len(y):
        raise ValueError("residual features and targets must be non-empty and aligned")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("residual training data must be finite")
    design = np.column_stack([np.ones(len(x)), x])
    penalty = np.eye(design.shape[1]) * shrinkage
    penalty[0, 0] = 0.0
    parameters = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    intercept = float(np.clip(parameters[0], -coefficient_bound, coefficient_bound))
    coefficients = np.clip(
        parameters[1:],
        -coefficient_bound,
        coefficient_bound,
    )
    return intercept, coefficients


def _validated_inputs(
    features: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    y = np.asarray(target, dtype=float)
    prior = np.asarray(anchor, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or x.shape[0] != len(y):
        raise ValueError("features must be 2D and aligned with the 1D target")
    if prior.shape != (x.shape[1],):
        raise ValueError("anchor must contain one weight per feature")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("training data must be finite")
    if np.any(prior < 0) or prior.sum() <= 0:
        raise ValueError("anchor must contain nonnegative weights with positive sum")
    return x, y, prior / prior.sum()


def _solve_simplex(objective: object, anchor: np.ndarray) -> np.ndarray:
    result = minimize(
        objective,
        anchor,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(anchor),
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"rank optimization failed: {result.message}")
    weights = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    return weights / weights.sum()
