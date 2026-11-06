"""Real-data acceptance checks for bounded research over a 100-stock universe."""

import json
import math
from pathlib import Path

import polars as pl
import pytest

from financial_strategy.adaptive.config import load_adaptive_config
from financial_strategy.adaptive.data import derive_range_labels
from financial_strategy.adaptive.evaluation import evaluate_range_cutoffs
from financial_strategy.adaptive.replay import replay_adaptive
from financial_strategy.adaptive.service import run_adaptive
from financial_strategy.audit.writer import verify_manifest
from financial_strategy.schemas import EvidenceRecord, MacroEvidenceRecord

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/2023-11-01_2026-10-30"


@pytest.fixture(scope="module")
def study(tmp_path_factory):
    if not (ROOT / "data/research100_v1/preparation.json").exists():
        pytest.skip("Run scripts/prepare_research100.py to materialize the research subset")
    config = load_adaptive_config(ROOT / "configs/adaptive_range_100_offline.yaml")
    directory = tmp_path_factory.mktemp("research100")
    config = config.model_copy(
        update={
            "output": {"runs_path": directory / "runs"},
            "adaptive": config.adaptive.model_copy(
                update={"max_batches": 1, "compact_research_prompts": True}
            ),
        }
    )
    output, watchlist = run_adaptive(config, run_id="scripted")
    return config, output, watchlist, directory


def test_research_screen_limits_context_but_keeps_full_universe_and_replay(study):
    config, output, watchlist, _ = study
    assert verify_manifest(output)["status"] == "completed"
    forecasts = json.loads((output / "range_predictions.json").read_text())
    screening = json.loads((output / "research_screening.json").read_text())
    assert len(forecasts) == 100 and len(watchlist) == 10
    assert screening["eligible_count"] == 100 and screening["candidate_count"] == 20
    candidates = set(screening["candidate_symbols"])
    assert len([r for r in forecasts if r["symbol"] not in candidates]) == 80
    assert all(
        r["predicted_range"] == r["baseline_range"]
        for r in forecasts
        if r["symbol"] not in candidates
    )
    calls = [json.loads(s) for s in (output / "model_calls.jsonl").read_text().splitlines()]
    assert calls and all(c["result"]["source"] == "scripted" for c in calls)
    plans = [c["payload"] for c in calls if c["payload"]["op"] == "adaptive_plan"]
    assert plans and all(len(p["context"]["companies"]) == 20 for p in plans)
    assert json.loads((output / "budget.json").read_text())["actual_usd"] == 0
    assert replay_adaptive(output) == watchlist


def test_future_target_prices_cannot_change_forecasts_and_no_label_file_is_read(study, monkeypatch):
    config, _, _, directory = study
    baseline = config.model_copy(
        update={"adaptive": config.adaptive.model_copy(update={"policy": "baseline"})}
    )
    target = baseline.task.target_session
    original_read = pl.read_parquet
    bars = original_read(config.data.market_bars)
    changed = bars.with_columns(
        *[
            pl.when(pl.col("session") >= target)
            .then(pl.col(field) * 1000)
            .otherwise(pl.col(field))
            .alias(field)
            for field in ["open", "high", "low", "close"]
        ]
    )
    alternate = directory / "future_poisoned.parquet"
    changed.write_parquet(alternate)

    def guard(path, *args, **kwargs):
        assert "labels.parquet" not in str(path)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", guard)
    before, watch_before = run_adaptive(baseline, run_id="before")
    after, watch_after = run_adaptive(
        baseline.model_copy(
            update={"data": baseline.data.model_copy(update={"market_bars": alternate})}
        ),
        run_id="after",
    )
    assert watch_before == watch_after
    assert (before / "range_predictions.json").read_text() == (
        after / "range_predictions.json"
    ).read_text()
    frozen = original_read(after / "snapshot/market.parquet")
    assert frozen["session"].max() < target
    assert frozen["available_at"].max() <= baseline.task.decision_cutoff
    assert len(json.loads((after / "range_features.json").read_text())) == 100


def test_scores_use_full_100_stock_oracle_at_each_cutoff(study):
    config, output, _, _ = study
    predictions = json.loads((output / "range_predictions.json").read_text())
    # Scores are computed only after the forecast manifest has been published.
    assert verify_manifest(output)["status"] == "completed"
    labels, coverage = derive_range_labels(
        config.data.market_bars, [config.task.target_session], [p["symbol"] for p in predictions]
    )
    assert coverage["available_label_count"] == 100
    actual = {row["symbol"]: row["range"] for row in labels.to_dicts()}
    metrics = evaluate_range_cutoffs(predictions, actual, top_ks=[3, 5, 10])
    ordered = sorted(predictions, key=lambda p: (-p["predicted_range"], p["symbol"]))
    for k in [3, 5, 10]:
        selected = [actual[p["symbol"]] for p in ordered[:k]]
        ideal = sorted(actual.values(), reverse=True)[:k]
        expected = sum(v / math.log2(i + 2) for i, v in enumerate(selected)) / sum(
            v / math.log2(i + 2) for i, v in enumerate(ideal)
        )
        assert metrics[str(k)]["ndcg_at_k"] == pytest.approx(expected)
        assert metrics[str(k)]["label_count"] == 100


def test_native_sec_metadata_and_macro_confidence_are_preserved():
    records = [json.loads(s) for s in (SOURCE / "company/evidence.jsonl").read_text().splitlines()]
    multi = next(r for r in records if len(r["event_types"]) > 1)
    parsed = EvidenceRecord.model_validate(multi)
    assert parsed.event_types == multi["event_types"]
    assert parsed.event_type == multi["event_types"][0]
    assert parsed.filing_form == multi["filing_form"]
    assert parsed.chunks[0].content_hash == multi["chunks"][0]["content_hash"]
    macro = [
        MacroEvidenceRecord.model_validate(json.loads(s))
        for s in (SOURCE / "macro/events.jsonl").read_text().splitlines()
    ]
    assert len(macro) == 171 and all(r.timestamp_confidence.value == "low" for r in macro)
    assert (ROOT / "data/research100_v1/macro/events.jsonl").read_text() == ""
