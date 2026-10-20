import json
from pathlib import Path

import pytest
import yaml

from financial_strategy.audit.replay import replay_run
from financial_strategy.config import load_mvp_config
from financial_strategy.service import run_screening

ROOT = Path(__file__).resolve().parents[1]


def test_screening_freezes_responses_and_replays_without_cache(tmp_path: Path) -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    data = config.model_dump(mode="json")
    for key in (
        "universe",
        "market_bars",
        "corporate_actions",
        "company_evidence",
        "macro_events",
        "macro_documents",
        "sector_exposures",
    ):
        data["data"][key] = str(ROOT / data["data"][key])
    data["agents"]["mode"] = "scripted"
    data["agents"]["model"] = "scripted"
    data["agents"]["recordings_path"] = str(ROOT / data["agents"]["recordings_path"])
    data["agents"]["cache_path"] = str(tmp_path / "cache")
    data["output"]["runs_path"] = str(tmp_path / "runs")
    config_path = tmp_path / "real.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=True))

    run_dir, watchlist = run_screening(config_path, run_id="e2e")
    assert watchlist
    assert list((run_dir / "llm_responses").glob("*.json"))
    assert (run_dir / "snapshot/universe.json").exists()
    assert (run_dir / "snapshot/sector_exposures.json").exists()
    response = json.loads(next((run_dir / "llm_responses").glob("*.json")).read_text())
    assert response["validation_status"] == "valid"
    assert response["response_content_hash"]
    assert replay_run(run_dir) == watchlist


def test_agent_failure_publishes_inputs_without_watchlist(tmp_path: Path) -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    data = config.model_dump(mode="json")
    for key in (
        "universe",
        "market_bars",
        "corporate_actions",
        "company_evidence",
        "macro_events",
        "macro_documents",
        "sector_exposures",
    ):
        data["data"][key] = str(ROOT / data["data"][key])
    data["agents"]["mode"] = "recorded"
    data["agents"]["recordings_path"] = str(tmp_path / "empty_recordings")
    data["output"]["runs_path"] = str(tmp_path / "runs")
    config_path = tmp_path / "failing.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=True))

    with pytest.raises(RuntimeError, match="No recording exists"):
        run_screening(config_path, run_id="failed_e2e")

    run_dir = tmp_path / "runs/failed_e2e"
    assert json.loads((run_dir / "status.json").read_text())["status"] == "failed"
    assert (run_dir / "snapshot/market.parquet").exists()
    assert not (run_dir / "watchlist.json").exists()
