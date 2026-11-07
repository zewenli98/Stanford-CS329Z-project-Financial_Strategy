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
    published = json.loads((run_dir / "published_research_cards.json").read_text())
    assert all(
        card["deterministic_verification"]["verdict"] == "PASS"
        and card["semantic_verification"]["verdict"] == "PASS"
        for card in published
    )
    response = json.loads(next((run_dir / "llm_responses").glob("*.json")).read_text())
    assert response["validation_status"] == "valid"
    assert response["response_content_hash"]
    assert replay_run(run_dir) == watchlist


@pytest.mark.parametrize("error_type", [ConnectionError, TimeoutError])
def test_one_provider_failure_retains_other_verified_cards(monkeypatch, error_type):
    from types import SimpleNamespace

    from test_corrective_retrieval import _task
    from test_llm_residual import _feature

    from financial_strategy.schemas import (
        MacroContext,
        ResearchCard,
        SymbolMacroExposure,
        VerificationReport,
    )
    from financial_strategy.workflow import graph

    task = _task()
    features = {s: _feature().model_copy(update={"symbol": s}) for s in ("AAPL", "MSFT")}
    exposures = [
        SymbolMacroExposure(
            symbol=s,
            sector="technology",
            exposure_method="sector_heuristic_v1",
            template_version="v1",
            exposures={},
        )
        for s in features
    ]

    class Symbols:
        def invoke(self, state):
            if state["symbol"] == "AAPL":
                raise error_type("injected provider outage")
            card = ResearchCard(
                symbol=state["symbol"],
                task_id=task.task_id,
                decision_cutoff=task.decision_cutoff,
                target_session=task.target_session,
                features=state["feature"],
                deterministic_verification=VerificationReport(verdict="PASS"),
                semantic_verification=VerificationReport(verdict="PASS"),
            )
            return {"card": card}

    class Agents:
        def macro(self, *args):
            return MacroContext(task_id=task.task_id, events=[], evidence_ids=[]), None, []

    monkeypatch.setattr(graph, "build_symbol_graph", lambda agents: Symbols())
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    result = graph.build_screening_graph(config, Agents()).invoke(
        {
            "task": task,
            "features": features,
            "candidates": list(features),
            "snapshot": SimpleNamespace(
                snapshot_id="snapshot", company_evidence=[], macro_evidence=[], exposures=exposures
            ),
            "volatility": {"AAPL": 0.6, "MSFT": 0.5},
            "anomaly": {"AAPL": 0.6, "MSFT": 0.5},
            "cards": [],
            "initial_cards": [],
            "attempts": [],
            "retrieval_traces": [],
        }
    )
    cards = {c.symbol: c for c in result["cards"]}
    assert cards["AAPL"].eligibility.eligible is False
    assert cards["AAPL"].semantic_verification.verdict == "ABSTAIN"
    assert cards["AAPL"].eligibility.abstain_reasons == [error_type.__name__]
    assert cards["MSFT"].semantic_verification.verdict == "PASS"
    assert {r.symbol for r in result["watchlist"]} == set(features)
    assert all(r.score == r.market_score for r in result["watchlist"])


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
