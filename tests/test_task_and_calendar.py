import json
from datetime import date, datetime
from pathlib import Path

import pytest

from financial_strategy.config import load_mvp_config
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.errors import ConfigError
from financial_strategy.task.validator import validate_and_freeze_task

ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    universe = json.loads((ROOT / config.data.universe).read_text())
    exposures = json.loads((ROOT / config.data.sector_exposures).read_text())
    return config, universe, exposures


def test_calendar_returns_exact_completed_session_count() -> None:
    config, _, _ = _inputs()
    calendar = TradingCalendar()
    sessions = calendar.completed_sessions_before(config.task.decision_cutoff, 21)
    assert len(sessions) == 21
    assert sessions[-1] == date(2026, 10, 13)


def test_task_validator_accepts_premarket_common_stock_universe() -> None:
    config, universe, exposures = _inputs()
    task, materialized, warnings = validate_and_freeze_task(
        config, TradingCalendar(), universe, exposures
    )
    assert task.target_session == date(2026, 10, 14)
    assert len(materialized) == len(universe["securities"])
    assert not warnings
    assert set(materialized[0].exposures) == {
        "inflation",
        "interest_rates",
        "oil_prices",
        "usd",
    }


def test_task_validator_rejects_cutoff_after_open() -> None:
    config, universe, exposures = _inputs()
    invalid = config.model_copy(
        update={
            "task": config.task.model_copy(
                update={"decision_cutoff": datetime.fromisoformat("2026-10-14T10:00:00-04:00")}
            )
        }
    )
    with pytest.raises(ConfigError, match="must be before"):
        validate_and_freeze_task(invalid, TradingCalendar(), universe, exposures)


def test_missing_exposure_factor_fails_closed() -> None:
    config, universe, exposures = _inputs()
    exposures["sectors"]["technology"].pop("usd")
    with pytest.raises(ConfigError) as error:
        validate_and_freeze_task(config, TradingCalendar(), universe, exposures)
    assert error.value.code == "INCOMPLETE_EXPOSURE_TEMPLATE"
