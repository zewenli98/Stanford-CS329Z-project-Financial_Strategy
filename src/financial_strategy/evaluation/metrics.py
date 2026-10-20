"""Explicit ranking metrics with zero-filled missing positions."""

from __future__ import annotations

import math


def evaluate_ranking(
    symbols: list[str],
    labels: dict[str, float],
    *,
    k: int,
    eligible_count: int,
    large_move_threshold: float,
) -> dict[str, float | int | None]:
    k_effective = min(k, eligible_count)
    if k_effective == 0:
        return {
            "k_effective": 0,
            "selection_count": len(symbols),
            "selection_coverage": None,
            "precision_at_k": None,
            "mean_absolute_return_at_k": None,
            "ndcg_at_k": None,
        }
    # Preserve rank positions when a selected symbol has no valid label. Filtering
    # first would incorrectly move lower-ranked names upward and inflate DCG.
    selected_returns = [labels.get(symbol, 0.0) for symbol in symbols[:k_effective]]
    selected_returns.extend([0.0] * (k_effective - len(selected_returns)))
    hits = sum(value >= large_move_threshold for value in selected_returns)
    precision = hits / k_effective
    mean_return = sum(selected_returns) / k_effective

    dcg = sum(
        value / math.log2(rank + 1)
        for rank, value in enumerate(selected_returns, start=1)
    )
    ideal = sorted(labels.values(), reverse=True)[:k_effective]
    idcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(ideal, start=1))
    ndcg = None if idcg == 0 else dcg / idcg
    return {
        "k_effective": k_effective,
        "selection_count": min(len(symbols), k_effective),
        "selection_coverage": min(len(symbols), k_effective) / k_effective,
        "precision_at_k": precision,
        "mean_absolute_return_at_k": mean_return,
        "ndcg_at_k": ndcg,
    }
