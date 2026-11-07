from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from financial_strategy.evaluation.labels import load_rank_labels


def _write_snapshot(tmp_path: Path) -> Path:
    dataset = tmp_path / "snapshot"
    (dataset / "evaluation").mkdir(parents=True)
    (dataset / "market").mkdir()
    pl.DataFrame(
        [
            {"session": date(2026, 1, 5), "symbol": "AAA", "open": 51.0, "close": 52.0},
            {"session": date(2026, 1, 5), "symbol": "BBB", "open": 20.0, "close": 21.0},
        ]
    ).write_parquet(dataset / "evaluation/labels.parquet")
    pl.DataFrame(
        [
            {
                "session": date(2026, 1, 2),
                "symbol": "AAA",
                "close": 100.0,
                "high": 101.0,
                "low": 99.0,
            },
            {
                "session": date(2026, 1, 5),
                "symbol": "AAA",
                "close": 52.0,
                "high": 55.0,
                "low": 50.0,
            },
            {
                "session": date(2026, 1, 2),
                "symbol": "BBB",
                "close": 20.0,
                "high": 21.0,
                "low": 19.0,
            },
            {
                "session": date(2026, 1, 5),
                "symbol": "BBB",
                "close": 21.0,
                "high": 22.0,
                "low": 20.0,
            },
        ]
    ).write_parquet(dataset / "market/daily_bars.parquet")
    return dataset


def test_default_rank_labels_use_shared_intraday_range(tmp_path: Path) -> None:
    labels = load_rank_labels(_write_snapshot(tmp_path))
    assert labels["2026-01-05"]["AAA"] == pytest.approx((55 - 50) / 50)
    assert labels["2026-01-05"]["BBB"] == pytest.approx((22 - 20) / 20)
    assert "2026-01-02" not in labels
