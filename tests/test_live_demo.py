"""Both demo heads must share their control without confusing scores with ranges."""

import csv
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
import yaml

from financial_strategy.adaptive.config import AdaptiveConfig
from financial_strategy.adaptive.service import run_adaptive
from financial_strategy.config import MVPConfig
from scripts.demo_head_a import (
    ground_truth_after_publication,
    prepare_head_a,
    run_head_a,
    write_comparison,
)
from scripts.run_live_demo import credential_file, parse_args

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tmp_path(tmp_path_factory):
    # Audit filenames contain hashes, so keep the surrounding Windows path short.
    return tmp_path_factory.mktemp("d")


def raw_fixture(tmp_path):
    raw = yaml.safe_load((ROOT / "configs/adaptive_range_100_compact_live.yaml").read_text())
    fixture = yaml.safe_load((ROOT / "configs/adaptive_range_v1.yaml").read_text())
    raw["data"] = fixture["data"]
    raw["task"] = fixture["task"]
    for key, value in raw["data"].items():
        if key in {
            "universe",
            "market_bars",
            "corporate_actions",
            "company_evidence",
            "macro_events",
            "macro_documents",
            "sector_exposures",
        }:
            raw["data"][key] = str(ROOT / value)
    raw["agents"]["mode"] = "scripted"
    raw["agents"]["cache_path"] = str(tmp_path / "cache")
    raw["agents"]["recordings_path"] = str(tmp_path / "recordings")
    raw["output"]["runs_path"] = str(tmp_path / "b")
    raw["adaptive"]["policy"] = "baseline"
    raw["adaptive"]["replicate_id"] = "0"
    return raw


def test_head_a_runs_research_but_preserves_unpromoted_market_rank(tmp_path, monkeypatch):
    def fail_live(*args, **kwargs):
        raise AssertionError("Offline comparison must not construct a live provider.")

    monkeypatch.setattr("financial_strategy.llm.client.LiveClient", fail_live)
    raw = raw_fixture(tmp_path)
    _, baseline = run_adaptive(AdaptiveConfig.model_validate(raw), run_id="m")
    launch = tmp_path / "launch"
    launch.mkdir()
    path, config = prepare_head_a(raw, launch, budget=0.4)
    prepared = yaml.safe_load(path.read_text())
    prepared["output"]["runs_path"] = str(tmp_path / "a")
    path.write_text(yaml.safe_dump(prepared))
    config = MVPConfig.model_validate(prepared)
    result = run_head_a(path, config, "fixture", baseline)

    assert result["status"] == "completed", result.get("error")
    assert result["research_card_count"] > 0
    assert result["model_attempt_count"] > 0
    assert result["live_provider_responses"] == 0
    assert result["reported_live_cost_usd"] == 0
    assert result["residual_enabled"] is False
    assert [row["symbol"] for row in result["watchlist"]] == [row["symbol"] for row in baseline]
    assert all(row["score"] == pytest.approx(row["market_score"]) for row in result["watchlist"])
    assert prepared["data"] == raw["data"]
    assert prepared["task"] == raw["task"]
    assert config.agents.max_cost_usd == 0.4
    assert config.agents.cache_path != AdaptiveConfig.model_validate(raw).agents.cache_path
    assert "predicted_range" not in result["watchlist"][0]


def test_failed_head_a_leaves_its_comparison_blank_and_preserves_b(tmp_path, monkeypatch):
    raw = raw_fixture(tmp_path)
    path, config = prepare_head_a(raw, tmp_path, budget=0.4)

    def fail(*args, **kwargs):
        raise RuntimeError("test provider failure")

    monkeypatch.setattr("scripts.demo_head_a.run_screening", fail)
    baseline = [{"rank": 1, "symbol": "M"}]
    head_a = run_head_a(path, config, "failure", baseline)
    assert head_a["status"] == "failed"
    assert head_a["watchlist"] == []
    assert head_a["reported_live_cost_usd"] is None
    head_b = {"watchlist": [{"rank": 1, "symbol": "B", "predicted_range": 0.04}]}
    output = write_comparison(tmp_path, baseline, head_a, head_b)
    rows = list(csv.DictReader(output.open(encoding="utf-8")))
    assert rows[0]["head_a_symbol"] == ""
    assert rows[0]["head_a_score"] == ""
    assert rows[0]["head_b_symbol"] == "B"
    assert float(rows[0]["head_b_predicted_range_ratio"]) == 0.04
    json.dumps(head_a)


def test_live_defaults_use_august_27_and_three_batches():
    args = parse_args([])
    assert args.date == "2026-08-27"
    assert args.max_batches == 3
    assert args.head == "Both"
    assert parse_args(["--require-differences"]).require_differences
    assert parse_args(["--report-run-id", "saved"]).report_run_id == "saved"


@pytest.mark.parametrize("missing", [False, True])
def test_ground_truth_uses_full_prediction_universe_after_integrity_checks(
    tmp_path, monkeypatch, capsys, missing
):
    session = date(2026, 8, 27)
    bars = pl.DataFrame(
        {
            "session": [session] * 4,
            "symbol": ["A", "B", "C", "OUTSIDE"],
            "high": [110.0, 220.0, 125.0, 900.0],
            "low": [100.0, 200.0, 100.0, 100.0],
        }
    )
    if missing:
        bars = bars.filter(pl.col("symbol") != "B")
    bars_path = tmp_path / "bars.parquet"
    bars.write_parquet(bars_path)
    config = SimpleNamespace(
        data=SimpleNamespace(market_bars=bars_path),
        task=SimpleNamespace(target_session=session, top_k=2),
    )
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    predictions_path = baseline_dir / "range_predictions.json"
    predictions_path.write_text(json.dumps([{"symbol": s} for s in ["A", "B", "C"]]))
    original = predictions_path.read_bytes()
    checked = []
    monkeypatch.setattr("scripts.demo_head_a.verify_manifest", lambda p: checked.append(p.name))
    from financial_strategy.adaptive.data import derive_range_labels

    def checked_labels(*args):
        assert checked == ["baseline", "head_a", "head_b"]
        return derive_range_labels(*args)

    monkeypatch.setattr("financial_strategy.adaptive.data.derive_range_labels", checked_labels)
    head_a = {"status": "completed", "run_dir": str(tmp_path / "head_a")}
    head_b = {"status": "completed", "run_dir": str(tmp_path / "head_b")}
    truth = ground_truth_after_publication(config, baseline_dir, head_a, head_b)
    assert predictions_path.read_bytes() == original
    assert truth["coverage"]["expected_label_count"] == 3
    if missing:
        assert truth["status"] == "incomplete_coverage"
        assert truth["watchlist"] == []
        assert truth["coverage"]["missing_by_session"] == {"2026-08-27": ["B"]}
    else:
        assert truth["status"] == "completed"
        assert [row["symbol"] for row in truth["watchlist"]] == ["C", "A"]
        assert truth["watchlist"][0]["realized_range"] == pytest.approx(0.25)
    output = write_comparison(tmp_path, [{"rank": 1, "symbol": "A"}], ground_truth=truth)
    row = list(csv.DictReader(output.open(encoding="utf-8")))[0]
    assert row["ground_truth_symbol"] == ("" if missing else "C")
    printed = capsys.readouterr().out
    assert "Ground truth" in printed
    assert ("incomplete session high/low coverage" if missing else "25.00%") in printed


def test_ground_truth_does_not_load_outcomes_when_publication_check_fails(tmp_path, monkeypatch):
    def corrupt_run(*args):
        raise ValueError("publication integrity failed")

    def forbidden_labels(*args):
        raise AssertionError("Outcomes must not be opened before publication checks")

    monkeypatch.setattr("scripts.demo_head_a.verify_manifest", corrupt_run)
    monkeypatch.setattr("financial_strategy.adaptive.data.derive_range_labels", forbidden_labels)
    with pytest.raises(ValueError, match="publication integrity failed"):
        ground_truth_after_publication(None, tmp_path)


def test_saved_report_refresh_needs_no_provider_and_preserves_predictions(tmp_path, monkeypatch):
    import scripts.run_live_demo as demo

    raw = raw_fixture(tmp_path)
    baseline_dir, _ = run_adaptive(AdaptiveConfig.model_validate(raw), run_id="saved_baseline")
    symbols = [
        row["symbol"]
        for row in json.loads((baseline_dir / "range_predictions.json").read_text())
    ]
    outcomes = tmp_path / "outcomes.parquet"
    pl.DataFrame(
        {
            "session": [date.fromisoformat(raw["task"]["target_session"])] * len(symbols),
            "symbol": symbols,
            "high": [110.0 + i for i in range(len(symbols))],
            "low": [100.0] * len(symbols),
        }
    ).write_parquet(outcomes)
    raw["data"]["market_bars"] = str(outcomes)
    launch = tmp_path / "runs/demo_launches/saved"
    launch.mkdir(parents=True)
    combined = {
        "session": raw["task"]["target_session"],
        "baseline_run_dir": str(baseline_dir),
        "heads": {},
    }
    (launch / "summary.json").write_text(json.dumps(combined))
    (launch / "config.yaml").write_text(yaml.safe_dump(raw))
    preserved = [baseline_dir / "range_predictions.json", launch / "summary.json"]
    before = [path.read_bytes() for path in preserved]

    def no_provider(*args, **kwargs):
        raise AssertionError("Saved comparison must not use credentials or model inference")

    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(demo, "credential_file", no_provider)
    monkeypatch.setattr(demo, "run_adaptive", no_provider)
    monkeypatch.setattr(demo, "run_head_a", no_provider)
    monkeypatch.setattr("financial_strategy.llm.client.LiveClient", no_provider)
    assert demo.refresh_saved_comparison("saved") == 0
    truth = json.loads((launch / "ground_truth.json").read_text())
    assert truth["status"] == "completed"
    assert truth["watchlist"]
    assert (launch / "comparison.csv").is_file()
    assert [path.read_bytes() for path in preserved] == before


def test_launcher_uses_its_own_source_even_with_stale_pythonpath(tmp_path):
    stale = tmp_path / "stale" / "financial_strategy"
    stale.mkdir(parents=True)
    (stale / "__init__.py").write_text('raise AssertionError("stale runtime loaded")')
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-c",
            "import scripts.run_live_demo; import financial_strategy; "
            "print(financial_strategy.__file__)",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(stale.parent)},
        capture_output=True,
        text=True,
        check=True,
    )
    assert Path(result.stdout.strip()).resolve().is_relative_to(ROOT / "src")


def test_credentials_prefer_checkout_then_shared_repo(tmp_path, monkeypatch):
    import scripts.run_live_demo as demo

    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shared = tmp_path / "main"
    (shared / ".git").mkdir(parents=True)
    (shared / ".env").write_text("TEST_ONLY=shared")
    monkeypatch.setattr(demo, "ROOT", checkout)
    monkeypatch.setattr(demo.subprocess, "check_output", lambda *a, **k: str(shared / ".git"))
    assert credential_file() == shared / ".env"
    (checkout / ".env").write_text("TEST_ONLY=local")
    assert credential_file() == checkout / ".env"
    assert credential_file(shared / ".env") == shared / ".env"
    with pytest.raises(ValueError, match="Credential file not found"):
        credential_file(tmp_path / "missing.env")


def test_head_a_reference_resolution_is_automatic_in_normal_run(tmp_path, monkeypatch):
    from financial_strategy.agents.roles import scripted_responder

    def source_references(role, payload):
        result = scripted_responder(role, payload)
        if role == "synthesis":
            supported = next(
                (c for c in payload.get("company_claims", []) if c["evidence_ids"]), None
            )
            if supported:
                result["catalyst_assessment"].update(
                    evidence_ids=[supported["claim_id"]],
                    signed_residual_bucket="promote",
                    target_session_relevance="high",
                    target_session_timing="unknown",
                )
        return result

    monkeypatch.setattr("financial_strategy.service.scripted_responder", source_references)
    raw = raw_fixture(tmp_path)
    _, baseline = run_adaptive(AdaptiveConfig.model_validate(raw), run_id="m_auto")
    launch = tmp_path / "auto"
    launch.mkdir()
    path, config = prepare_head_a(raw, launch, budget=0.4)
    prepared = yaml.safe_load(path.read_text())
    prepared["output"]["runs_path"] = str(tmp_path / "a_auto")
    path.write_text(yaml.safe_dump(prepared))
    result = run_head_a(path, MVPConfig.model_validate(prepared), "auto", baseline)
    assert result["status"] == "completed", result.get("error")
    assert result["changed_top_10_scores"]
    cards = json.loads((Path(result["run_dir"]) / "research_cards.json").read_text())
    supported = [
        c
        for c in cards
        if c.get("catalyst_assessment")
        and c["catalyst_assessment"]["signed_residual_bucket"] == "promote"
    ]
    assert supported
    for card in supported:
        actual_ids = {i for claim in card["claims"] for i in claim["evidence_ids"]}
        assert set(card["catalyst_assessment"]["evidence_ids"]) <= actual_ids
        assert card["catalyst_assessment"]["evidence_ids"]
    assert config.agents.roles["company_evidence"].max_tokens >= 4096
    assert config.agents.roles["verifier"].max_tokens >= 4096
