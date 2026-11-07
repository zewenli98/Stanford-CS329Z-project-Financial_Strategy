"""Offline integration checks for audited adaptive screening on frozen real inputs."""

from __future__ import annotations

import json
import re
from pathlib import Path

import polars as pl
import pytest
import yaml

from financial_strategy.adaptive.config import load_adaptive_config
from financial_strategy.adaptive.service import run_adaptive
from financial_strategy.audit.writer import verify_manifest

ROOT = Path(__file__).resolve().parents[1]


def adaptive_config(tmp_path, *, policy="scenario", budget=2.0, batches=2):
    raw = yaml.safe_load((ROOT / "configs/screening_2026-10-14.yaml").read_text())
    raw["workflow"] = "adaptive_range_v1"
    for key in (
        "universe",
        "market_bars",
        "corporate_actions",
        "company_evidence",
        "macro_events",
        "macro_documents",
        "sector_exposures",
    ):
        raw["data"][key] = str(ROOT / raw["data"][key])
    raw["agents"]["mode"] = "scripted"
    raw["agents"]["model"] = "scripted"
    raw["agents"]["cache_path"] = str(tmp_path / "unused_cache")
    raw["agents"]["recordings_path"] = str(tmp_path / "unused_recordings")
    raw["agents"]["roles"]["macro_evidence"]["max_tokens"] = 8192
    raw["agents"]["roles"]["synthesis"]["max_tokens"] = 4096
    raw["output"]["runs_path"] = str(tmp_path / "runs")
    raw["adaptive"] = {
        "policy": policy,
        "daily_budget_usd": budget,
        "max_workers": 2,
        "max_batches": batches,
        "exploration_probability": 0.2,
        "seed": 31,
    }
    path = tmp_path / f"{policy}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=True))
    return load_adaptive_config(path)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_adaptive_screen_records_research_budget_and_opaque_model_trace(tmp_path):
    config = adaptive_config(tmp_path)
    run_dir, watchlist = run_adaptive(config, run_id="adaptive_real")
    assert watchlist
    assert len(watchlist) <= config.task.top_k
    assert all(item["predicted_range"] >= 0 for item in watchlist)
    assert [item["rank"] for item in watchlist] == list(range(1, len(watchlist) + 1))
    assert verify_manifest(run_dir)["status"] == "completed"
    budget = json.loads((run_dir / "budget.json").read_text())
    assert 0 < budget["normalized_uncached_usd"] <= config.adaptive.daily_budget_usd
    assert budget["actual_usd"] == 0
    assert budget["reserved_usd"] == 0
    assert rows(run_dir / "budget_ledger.jsonl")
    assert rows(run_dir / "tool_traces.jsonl")
    assert rows(run_dir / "allocation_decisions.jsonl")
    calls = rows(run_dir / "model_calls.jsonl")
    operations = {call["payload"]["op"] for call in calls}
    assert {"adaptive_plan", "adaptive_research", "adaptive_verify"} <= operations
    assert "adaptive_forecast" in operations
    cards = json.loads((run_dir / "research_cards.json").read_text())
    assert cards and all(card["status"] == "accepted" for card in cards)
    assert all(card["source_spans"] for card in cards)
    private = json.loads((run_dir / "private_mapping.json").read_text())
    assert "AAPL" in private["companies"].values()
    visible = json.dumps([call["payload"] for call in calls])
    assert not re.search(r"\b(?:AAPL|MSFT|NVDA|Apple|Microsoft|NVIDIA)\b", visible, re.I)
    assert "https://" not in visible
    assert "C:\\\\" not in visible
    assert "company_" in visible


@pytest.mark.parametrize("memory_days,named_timezone", [(7, False), (30, False), (30, True)])
def test_adaptive_replay_uses_frozen_calls_without_live_client_or_cache(
    tmp_path, monkeypatch, memory_days, named_timezone
):
    from financial_strategy.adaptive.replay import replay_adaptive

    config = adaptive_config(tmp_path, batches=1)
    # Thirty calendar days from October 14 crosses the November DST boundary.
    config = config.model_copy(
        update={"adaptive": config.adaptive.model_copy(update={"memory_days": memory_days})}
    )
    if named_timezone:
        from zoneinfo import ZoneInfo

        config = config.model_copy(
            update={
                "task": config.task.model_copy(
                    update={
                        "decision_cutoff": config.task.decision_cutoff.astimezone(
                            ZoneInfo("America/New_York")
                        )
                    }
                )
            }
        )
    run_dir, expected = run_adaptive(config, run_id="replay_source")

    def fail_live(*args, **kwargs):
        raise AssertionError("replay must not construct a live client")

    monkeypatch.setattr("financial_strategy.adaptive.budget.LiveClient", fail_live)
    # External source paths are irrelevant after publication: replay uses its
    # snapshot, source copies, initial memory, and exact recorded model-call tape.
    assert not config.agents.cache_path.exists()
    assert replay_adaptive(run_dir) == expected


def test_baseline_makes_zero_model_calls_and_history_control_one(tmp_path):
    baseline_dir, baseline = run_adaptive(
        adaptive_config(tmp_path, policy="baseline"), run_id="baseline"
    )
    assert baseline
    assert rows(baseline_dir / "model_calls.jsonl") == []
    assert all(item["baseline_range"] == item["predicted_range"] for item in baseline)
    assert json.loads((baseline_dir / "budget.json").read_text())["normalized_uncached_usd"] == 0
    history_dir, history = run_adaptive(
        adaptive_config(tmp_path, policy="history_only"), run_id="history"
    )
    calls = rows(history_dir / "model_calls.jsonl")
    assert len(calls) == 1
    assert calls[0]["payload"]["op"] == "adaptive_history"
    assert all(item["baseline_range"] == item["predicted_range"] for item in history)
    assert all(not item["supporting_card_ids"] for item in history)


@pytest.mark.parametrize("error_name", ["RuntimeError", "LengthFinishReasonError"])
def test_replay_retains_forecast_failure_labels_without_inventing_error_codes(
    tmp_path, monkeypatch, error_name
):
    from financial_strategy.adaptive import service
    from financial_strategy.adaptive.replay import replay_adaptive

    original = service.deterministic_responder
    error_type = type(error_name, (Exception,), {})

    def failing_forecast(role, payload):
        if payload["op"] == "adaptive_forecast":
            raise error_type("injected output failure")
        return original(role, payload)

    monkeypatch.setattr(service, "deterministic_responder", failing_forecast)
    source, expected = run_adaptive(adaptive_config(tmp_path, batches=1), run_id="output_failure")
    revisions = rows(source / "forecast_revisions.jsonl")
    assert revisions and all(row["error"] == "INVALID_FORECAST" for row in revisions)
    assert replay_adaptive(source) == expected


def test_no_accepted_evidence_preserves_every_market_baseline(tmp_path, monkeypatch):
    from financial_strategy.adaptive import service

    original = service.deterministic_responder

    def abstaining(role, payload):
        if payload["op"] == "adaptive_research":
            return {"tool": "abstain", "reason": "No usable additional archive evidence."}
        return original(role, payload)

    monkeypatch.setattr(service, "deterministic_responder", abstaining)
    run_dir, _ = run_adaptive(adaptive_config(tmp_path), run_id="abstention")
    predictions = json.loads((run_dir / "range_predictions.json").read_text())
    assert len(predictions) == 20
    assert all(item["predicted_range"] == item["baseline_range"] for item in predictions)
    assert all(item["fallback"] for item in predictions)
    assert json.loads((run_dir / "research_cards.json").read_text()) == []
    assert not any(
        call["payload"]["op"] == "adaptive_forecast" for call in rows(run_dir / "model_calls.jsonl")
    )


def test_screening_cannot_read_labels_or_target_session_prices(tmp_path, monkeypatch):
    config = adaptive_config(tmp_path, policy="baseline")
    original_read = pl.read_parquet
    reads = []

    def guarded_read(path, *args, **kwargs):
        reads.append(str(path))
        assert "label" not in str(path).lower(), "forecast path must never read evaluation labels"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", guarded_read)
    before_dir, before = run_adaptive(config, run_id="before_future_bar")
    frame = original_read(config.data.market_bars)
    latest = frame.sort("session").group_by("symbol", maintain_order=True).tail(1)
    target = latest.with_columns(
        pl.lit(config.task.target_session).alias("session"),
        pl.lit(config.task.decision_cutoff.replace(hour=16)).alias("bar_end"),
        pl.lit(config.task.decision_cutoff.replace(hour=16, minute=5)).alias("available_at"),
        (pl.col("high") * 1000).alias("high"),
    )
    changed_bars = tmp_path / "market_with_future.parquet"
    pl.concat([frame, target]).write_parquet(changed_bars)
    changed = config.model_copy(
        update={"data": config.data.model_copy(update={"market_bars": changed_bars})}
    )
    after_dir, after = run_adaptive(changed, run_id="after_future_bar")
    assert before == after
    assert json.loads((before_dir / "range_predictions.json").read_text()) == json.loads(
        (after_dir / "range_predictions.json").read_text()
    )
    assert reads and all("label" not in path.lower() for path in reads)


def test_unexpected_planner_failure_keeps_partial_model_and_budget_audit(tmp_path, monkeypatch):
    from financial_strategy.adaptive import service

    original = service.deterministic_responder

    def failing(role, payload):
        if payload["op"] == "adaptive_plan":
            raise RuntimeError("injected planner failure")
        return original(role, payload)

    monkeypatch.setattr(service, "deterministic_responder", failing)
    config = adaptive_config(tmp_path)
    with pytest.raises(RuntimeError, match="injected planner failure"):
        run_adaptive(config, run_id="failed_planner")
    run_dir = Path(config.output["runs_path"]) / "failed_planner"
    assert verify_manifest(run_dir)["status"] == "failed"
    assert (run_dir / "snapshot/market.parquet").exists()
    assert not (run_dir / "watchlist.json").exists()
    attempts = rows(run_dir / "model_calls.jsonl")
    assert attempts and attempts[-1]["status"] == "failed"
    assert attempts[-1]["error_type"] == "RuntimeError"
    assert (run_dir / "budget.json").exists()
    assert rows(run_dir / "budget_ledger.jsonl")
