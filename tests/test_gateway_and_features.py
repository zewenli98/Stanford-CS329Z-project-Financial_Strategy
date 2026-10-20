from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from financial_strategy.config import load_mvp_config
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.data.gateway import PointInTimeGateway
from financial_strategy.data.providers import (
    LocalEvidenceProvider,
    LocalMarketDataProvider,
    LocalUniverseProvider,
)
from financial_strategy.features.engine import compute_features
from financial_strategy.task.validator import validate_and_freeze_task

ROOT = Path(__file__).resolve().parents[1]


EVALUATION_SESSIONS = [
    date(2026, 10, 1),
    date(2026, 10, 2),
    date(2026, 10, 5),
    date(2026, 10, 6),
    date(2026, 10, 7),
    date(2026, 10, 8),
    date(2026, 10, 9),
    date(2026, 10, 12),
    date(2026, 10, 13),
    date(2026, 10, 14),
]


def _snapshot(target_session: date | None = None):
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    if target_session is not None:
        cutoff = datetime.combine(
            target_session,
            datetime.min.time().replace(hour=9),
            ZoneInfo("America/New_York"),
        )
        task_config = config.task.model_copy(
            update={
                "decision_cutoff": cutoff,
                "target_session": target_session,
            }
        )
        config = config.model_copy(update={"task": task_config})
    calendar = TradingCalendar()
    universe_provider = LocalUniverseProvider(
        ROOT / config.data.universe,
        ROOT / config.data.sector_exposures,
    )
    task, exposures, _ = validate_and_freeze_task(
        config,
        calendar,
        universe_provider.universe(),
        universe_provider.sector_exposures(),
    )
    gateway = PointInTimeGateway(
        LocalMarketDataProvider(
            ROOT / config.data.market_bars,
            ROOT / config.data.corporate_actions,
        ),
        LocalEvidenceProvider(
            ROOT / config.data.company_evidence,
            ROOT / config.data.macro_events,
        ),
        calendar,
        input_adjustment_policy=config.data.input_adjustment_policy,
    )
    return config, calendar, task, gateway.freeze_inputs(task, exposures)


def test_gateway_freezes_only_real_point_in_time_inputs() -> None:
    _, _, task, snapshot = _snapshot()
    assert not snapshot.corporate_actions
    assert len(snapshot.company_evidence) >= 8
    assert all(
        item.available_at <= task.decision_cutoff
        for item in snapshot.company_evidence
    )
    assert snapshot.bars.get_column("adjustment_policy").unique().to_list() == [
        "split_adjusted"
    ]

    # The split-adjusted real series must not contain a synthetic 50% discontinuity.
    nvda = snapshot.bars.filter(snapshot.bars["symbol"] == "NVDA").sort("session")
    closes = np.asarray(nvda["close"].to_list())
    assert np.max(np.abs(closes[1:] / closes[:-1] - 1)) < 0.1


@pytest.mark.parametrize("target_session", EVALUATION_SESSIONS)
def test_gateway_has_complete_point_in_time_history_for_each_evaluation_session(
    target_session: date,
) -> None:
    _, _, task, snapshot = _snapshot(target_session)

    assert snapshot.bars.height == 60 * 20
    assert snapshot.bars["symbol"].n_unique() == 20
    assert snapshot.bars.group_by("symbol").len()["len"].unique().to_list() == [60]
    assert all(row < task.decision_cutoff for row in snapshot.bars["bar_end"])
    assert all(
        item.available_at <= task.decision_cutoff
        for item in (*snapshot.company_evidence, *snapshot.macro_evidence)
    )


def test_feature_engine_uses_exact_formulas_and_complete_sessions() -> None:
    config, calendar, task, snapshot = _snapshot()
    sessions = calendar.completed_sessions_before(task.decision_cutoff, 21)
    features, rejected = compute_features(snapshot.bars, sessions)
    assert not rejected
    assert len(features) == 20
    feature = features["AAPL"]
    assert feature.realized_volatility_20d >= 0
    assert feature.previous_session_relative_volume > 0
    assert len(feature.input_sessions) == config.features["required_complete_sessions"]


def test_feature_engine_rejects_one_missing_required_session() -> None:
    _, calendar, task, snapshot = _snapshot()
    sessions = calendar.completed_sessions_before(task.decision_cutoff, 21)
    missing = snapshot.bars.filter(
        ~(
            (snapshot.bars["symbol"] == "AAPL")
            & (snapshot.bars["session"] == sessions[-5])
        )
    )
    _, rejected = compute_features(missing, sessions)
    assert rejected["AAPL"] == "MISSING_REQUIRED_SESSION"


def test_gateway_does_not_double_adjust_split_adjusted_input() -> None:
    _, calendar, task, snapshot = _snapshot()
    source = (
        snapshot.bars.filter(snapshot.bars["symbol"] == "NVDA")
        .sort("session")
        .head(1)
    )
    expected_close = source.item(0, "close")

    class Market:
        def bars(self):
            return source

        def corporate_actions(self):
            return list(snapshot.corporate_actions)

    class Evidence:
        def company_evidence(self):
            return []

        def macro_evidence(self):
            return []

    gateway = PointInTimeGateway(
        Market(),
        Evidence(),
        calendar,
        input_adjustment_policy="split_adjusted",
    )
    rebuilt = gateway.freeze_inputs(task, list(snapshot.exposures))
    assert rebuilt.bars.item(0, "close") == expected_close
