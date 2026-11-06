"""Shared point-in-time ranking label definitions."""

from __future__ import annotations

import json
from datetime import date, datetime
from math import prod
from pathlib import Path
from typing import Literal

import polars as pl

RankLabelMode = Literal["open_to_close", "prev_close_to_close"]


def load_rank_labels(
    dataset: Path,
    mode: RankLabelMode = "open_to_close",
) -> dict[str, dict[str, float]]:
    """Load absolute-return labels without mutating the frozen snapshot."""

    if mode == "open_to_close":
        labels: dict[str, dict[str, float]] = {}
        frame = pl.read_parquet(dataset / "evaluation/labels.parquet")
        for row in frame.to_dicts():
            labels.setdefault(row["session"].isoformat(), {})[row["symbol"]] = abs(
                row["close"] / row["open"] - 1.0
            )
        return labels
    if mode != "prev_close_to_close":
        raise ValueError(f"unknown rank label mode: {mode}")

    split_actions: dict[str, list[tuple[date, float]]] = {}
    action_path = dataset / "market/corporate_actions.jsonl"
    for line in action_path.read_text().splitlines():
        if not line:
            continue
        action = json.loads(line)
        if action["action_type"] not in {"split", "reverse_split"}:
            continue
        effective_date = datetime.fromisoformat(
            action["effective_at"].replace("Z", "+00:00")
        ).date()
        split_actions.setdefault(action["symbol"], []).append(
            (effective_date, float(action["ratio"]))
        )

    panel = {
        (row["session"], row["symbol"])
        for row in pl.read_parquet(
            dataset / "evaluation/labels.parquet"
        ).select("session", "symbol").iter_rows(named=True)
    }
    bars = pl.read_parquet(dataset / "market/daily_bars.parquet").sort(
        ["symbol", "session"]
    )
    previous_close: dict[str, float] = {}
    previous_session: dict[str, date] = {}
    labels: dict[str, dict[str, float]] = {}
    for row in bars.select("symbol", "session", "close").iter_rows(named=True):
        symbol = row["symbol"]
        session = row["session"]
        close = float(row["close"])
        prior = previous_close.get(symbol)
        if prior is not None and (session, symbol) in panel:
            ratio = prod(
                action_ratio
                for effective_date, action_ratio in split_actions.get(symbol, [])
                if previous_session[symbol] < effective_date <= session
            )
            adjusted_prior = prior / ratio
            labels.setdefault(session.isoformat(), {})[symbol] = abs(
                close / adjusted_prior - 1.0
            )
        previous_close[symbol] = close
        previous_session[symbol] = session
    return labels
