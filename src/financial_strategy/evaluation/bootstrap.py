"""Deterministic paired moving-block bootstrap over chronological sessions."""

from __future__ import annotations

import math
import random


def paired_moving_block_bootstrap(
    challenger: dict[str, float | None],
    baseline: dict[str, float | None],
    *,
    seed: int = 329,
    samples: int = 2_000,
    block_sizes: list[int] | tuple[int, ...] = (5, 10, 20),
    confidence: float = 0.95,
) -> dict:
    """Return challenger-minus-baseline inference on paired dated observations."""

    sessions = sorted(challenger.keys() & baseline.keys())
    pairs = [
        (session, challenger[session], baseline[session])
        for session in sessions
        if challenger[session] is not None and baseline[session] is not None
    ]
    deltas = [float(left) - float(right) for _, left, right in pairs]
    if not deltas:
        return {
            "paired_session_count": 0,
            "mean_delta": None,
            "confidence_interval": None,
            "effective_block": None,
            "block_sensitivity": {},
        }
    sensitivity = {}
    for block in block_sizes:
        effective = min(int(block), len(deltas))
        rng = random.Random(f"{seed}:{block}:{len(deltas)}")
        means = []
        for _ in range(samples):
            sampled: list[float] = []
            while len(sampled) < len(deltas):
                start = rng.randrange(len(deltas))
                sampled.extend(
                    deltas[(start + offset) % len(deltas)] for offset in range(effective)
                )
            means.append(sum(sampled[: len(deltas)]) / len(deltas))
        means.sort()
        alpha = (1 - confidence) / 2
        lower = means[max(0, math.floor(alpha * (samples - 1)))]
        upper = means[min(samples - 1, math.ceil((1 - alpha) * (samples - 1)))]
        sensitivity[str(block)] = {
            "effective_block": effective,
            "confidence_interval": [lower, upper],
        }
    requested = list(block_sizes)
    selected = requested[len(requested) // 2]
    return {
        "paired_session_count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "confidence_interval": sensitivity[str(selected)]["confidence_interval"],
        "effective_block": sensitivity[str(selected)]["effective_block"],
        "block_sensitivity": sensitivity,
    }
