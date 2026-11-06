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
            {"session": date(2026, 1, 2), "symbol": "AAA", "close": 100.0},
            {"session": date(2026, 1, 5), "symbol": "AAA", "close": 52.0},
            {"session": date(2026, 1, 2), "symbol": "BBB", "close": 20.0},
            {"session": date(2026, 1, 5), "symbol": "BBB", "close": 21.0},
        ]
    ).write_parquet(dataset / "market/daily_bars.parquet")
    (dataset / "market/corporate_actions.jsonl").write_text(
        '{"symbol":"AAA","action_type":"split","ratio":2.0,'
        '"effective_at":"2026-01-05T14:30:00Z"}\n'
        '{"symbol":"BBB","action_type":"cash_dividend","ratio":5.0,'
        '"effective_at":"2026-01-05T14:30:00Z"}\n'
    )
    return dataset


def test_prev_close_labels_adjust_splits_but_not_dividends(tmp_path: Path) -> None:
    labels = load_rank_labels(_write_snapshot(tmp_path), "prev_close_to_close")
    assert labels["2026-01-05"]["AAA"] == pytest.approx(0.04)
    assert labels["2026-01-05"]["BBB"] == pytest.approx(0.05)


def test_open_to_close_labels_remain_available(tmp_path: Path) -> None:
    labels = load_rank_labels(_write_snapshot(tmp_path), "open_to_close")
    assert labels["2026-01-05"]["AAA"] == pytest.approx(abs(52 / 51 - 1))
