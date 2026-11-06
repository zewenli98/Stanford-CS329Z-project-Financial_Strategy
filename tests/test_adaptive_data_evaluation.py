from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from financial_strategy.adaptive.data import (
    baseline_predictions,
    compute_range_features,
    derive_range_labels,
)
from financial_strategy.adaptive.evaluation import (
    evaluate_range_cutoffs,
    evaluate_range_ranking,
    load_published_predictions,
    run_adaptive_evaluation,
    summarize_run_integrity,
)
from financial_strategy.audit import RunWriter
from financial_strategy.audit.writer import file_sha256
from financial_strategy.errors import AuditError, ConfigError

ROOT = Path(__file__).resolve().parents[1]


def _bars() -> tuple[pl.DataFrame, list[date]]:
    sessions = [date(2026, 8, 1) + timedelta(days=index) for index in range(21)]
    rows = [
        {
            "symbol": "A",
            "session": session,
            "open": 100.0,
            "close": 100.0 + index,
            "volume": 1000.0 + index,
            "high": 110.0 + index,
            "low": 100.0,
            "snapshot_id": "frozen",
        }
        for index, session in enumerate(sessions)
    ]
    return pl.DataFrame(rows), sessions


def test_range_features_use_twenty_completed_high_low_ranges_and_context():
    bars, sessions = _bars()
    features, rejected = compute_range_features(bars, sessions)
    assert not rejected
    expected_ranges = [(10 + index) / 100 for index in range(1, 21)]
    feature = features["A"]
    assert feature.mean_range_20d == pytest.approx(sum(expected_ranges) / 20)
    assert feature.median_range_20d == pytest.approx((0.20 + 0.21) / 2)
    assert feature.latest_range == pytest.approx(0.30)
    assert feature.historical_context.return_5d == pytest.approx(120 / 115 - 1)
    assert baseline_predictions(features) == {"A": feature.mean_range_20d}
    assert baseline_predictions(features, "latest_range") == {"A": 0.30}


def test_range_features_reject_bad_range_and_missing_session():
    bars, sessions = _bars()
    broken = bars.with_columns(
        pl.when(pl.col("session") == sessions[-1])
        .then(99.0)
        .otherwise(pl.col("high"))
        .alias("high")
    )
    features, rejected = compute_range_features(broken, sessions)
    assert not features
    assert rejected["A"] == "INVALID_HIGH_LOW_RANGE"
    features, rejected = compute_range_features(bars.slice(1), sessions)
    assert not features
    assert rejected["A"] == "INSUFFICIENT_PRICE_HISTORY"


def test_derive_labels_reports_missing_october_fourteenth_without_source_changes():
    source = ROOT / "data/2026-10-01_2026-10-14/market/daily_bars.parquet"
    before = file_sha256(source)
    labels, coverage = derive_range_labels(source, [date(2026, 10, 13), date(2026, 10, 14)])
    assert coverage["missing_sessions"] == ["2026-10-14"]
    assert coverage["available_label_count"] == 20
    assert coverage["expected_label_count"] == 40
    assert labels["session"].unique().to_list() == [date(2026, 10, 13)]
    for row in labels.to_dicts():
        assert row["range"] == pytest.approx((row["high"] - row["low"]) / row["low"])
    assert file_sha256(source) == before


def test_range_metrics_regret_and_order_are_independently_checkable():
    forecasts = [
        {"symbol": "A", "predicted_range": 0.09},
        {"symbol": "B", "predicted_range": 0.08},
        {"symbol": "C", "predicted_range": 0.01},
    ]
    metrics = evaluate_range_ranking(forecasts, {"A": 0.01, "B": 0.05, "C": 0.09}, k=2)
    assert metrics["mean_selected_range"] == pytest.approx(0.03)
    assert metrics["oracle_mean_range"] == pytest.approx(0.07)
    assert metrics["top_k_regret"] == pytest.approx(0.04)
    assert metrics["top_k_overlap"] == 0.5
    assert metrics["mae"] == pytest.approx((0.08 + 0.03 + 0.08) / 3)
    assert metrics["large_range_precision_at_k"] == 0.5


def test_range_metrics_do_not_promote_positions_with_missing_labels():
    forecasts = [
        {"symbol": "missing", "predicted_range": 0.20},
        {"symbol": "A", "predicted_range": 0.10},
        {"symbol": "B", "predicted_range": 0.05},
    ]
    metrics = evaluate_range_ranking(forecasts, {"A": 0.08, "B": 0.04}, k=2)
    assert metrics["mean_selected_range"] == 0.04
    assert metrics["top_k_regret"] == pytest.approx(0.02)
    assert metrics["missing_selected_label_count"] == 1
    empty = evaluate_range_ranking(forecasts, {}, k=2)
    assert empty["top_k_regret"] is None
    assert empty["mae"] is None


def test_cutoffs_use_forecast_prefixes_and_oracles_from_the_full_universe():
    forecasts = [
        {"symbol": symbol, "predicted_range": value}
        for symbol, value in [("A", 0.10), ("B", 0.09), ("C", 0.08), ("D", 0.07)]
    ]
    labels = {"A": 0.01, "B": 0.02, "C": 0.03, "D": 0.20}
    metrics = evaluate_range_cutoffs(list(reversed(forecasts)), labels, top_ks=[1, 2, 3])
    assert metrics["1"]["mean_selected_range"] == 0.01
    assert metrics["1"]["oracle_mean_range"] == 0.20
    assert metrics["2"]["mean_selected_range"] == pytest.approx(0.015)
    assert metrics["2"]["top_k_overlap"] == 0
    assert metrics["3"]["mean_selected_range"] == pytest.approx(0.02)
    assert metrics["3"]["top_k_overlap"] == pytest.approx(2 / 3)
    assert metrics["1"]["mae"] == metrics["3"]["mae"]


def test_prediction_loader_requires_manifested_completed_forecasts(tmp_path):
    failed = RunWriter(tmp_path, "failed")
    failed.write_json("range_predictions.json", [])
    failed.publish("failed")
    with pytest.raises(AuditError, match="failed"):
        load_published_predictions(tmp_path / "failed")
    completed = RunWriter(tmp_path, "missing")
    completed.publish("completed")
    with pytest.raises(AuditError, match="Missing manifested"):
        load_published_predictions(tmp_path / "missing")


def test_integrity_cost_resolution_and_rejection_proxies_have_explicit_denominators(tmp_path):
    writer = RunWriter(tmp_path, "diagnostics")
    writer.write_json(
        "budget.json",
        {
            "normalized_uncached_usd": 0.6,
            "actual_usd": 0.2,
            "cap_usd": 1.0,
            "blocked": False,
        },
    )
    writer.write_jsonl("model_calls.jsonl", [{"usage_estimated": False}] * 3)
    writer.write_jsonl(
        "model_attempts.jsonl",
        [
            {"status": "completed", "latency_seconds": 0.5, "input_tokens": 100},
            {"status": "failed", "latency_seconds": 0.2},
            {"status": "completed", "latency_seconds": 0.3, "output_tokens": 20},
        ],
    )
    prior = {"card_id": "prior", "status": "accepted", "observations": ["old"]}
    writer.write_json("memory_initial.json", [prior])
    writer.write_json(
        "research_cards.json",
        [
            prior,
            {
                "card_id": "new",
                "status": "accepted",
                "observations": ["one", "two"],
            },
        ],
    )
    writer.write_json(
        "rejected_cards.json",
        [
            {"observations": ["duplicate"], "verification_issues": ["DUPLICATE_EVENT"]},
            {
                "observations": ["bad", "also bad"],
                "verification_issues": ["UNSUPPORTED_NUMERICAL_FACT", "MISSING_EVIDENCE"],
            },
        ],
    )
    writer.write_jsonl(
        "research_results.jsonl",
        [
            {"accepted_card_ids": ["new"]},
            {"accepted_card_ids": []},
        ],
    )
    writer.write_jsonl(
        "forecast_revisions.jsonl",
        [
            {
                "before": [{"symbol": "A", "predicted_range": 0.01}],
                "after": [{"symbol": "A", "predicted_range": 0.02}],
            },
            {"error": "budget_exhausted"},
        ],
    )
    run_dir = writer.publish("completed")
    stats = summarize_run_integrity(run_dir, [{"revision": 1}])
    assert stats["normalized_uncached_usd"] == 0.6
    assert stats["actual_usd"] == 0.2
    assert stats["model_latency_seconds"] == 1.0
    assert stats["model_call_count"] == 3
    assert stats["failed_model_attempt_count"] == 1
    assert stats["new_unique_accepted_card_count"] == 1
    assert stats["accepted_card_yield_per_normalized_usd"] == pytest.approx(1 / 0.6)
    assert stats["accepted_evidence_resolution_rate"] == 0.5
    assert stats["unsupported_observation_proxy_rate"] == pytest.approx(2 / 6)
    assert stats["unsupported_rejected_card_count"] == 1
    assert stats["duplicate_rejection_count"] == 1
    assert stats["changed_forecast_count"] == 1
    assert stats["failed_forecast_update_count"] == 1


def test_partial_labels_are_reported_and_aggregate_metric_denominators_are_explicit(
    tmp_path,
    monkeypatch,
):
    import financial_strategy.adaptive.service as service

    path = _evaluation_files(tmp_path)
    spec = yaml.safe_load(path.read_text())
    spec["evaluation"].update({"policies": ["baseline"], "budgets_usd": [0.5], "replicates": 1})
    path.write_text(yaml.safe_dump(spec))
    market = tmp_path / "market.parquet"
    pl.read_parquet(market).filter(pl.col("symbol") == "A").write_parquet(market)

    def fake_run(config, *, run_id, task_overrides):
        writer = RunWriter(Path(config.output["runs_path"]), run_id)
        forecasts = [
            {"symbol": "A", "predicted_range": 0.01},
            {"symbol": "B", "predicted_range": 0.10},
        ]
        writer.write_json("range_predictions.json", forecasts)
        writer.write_json("budget.json", {"normalized_uncached_usd": 0.0, "actual_usd": 0.0})
        return writer.publish("completed"), forecasts

    monkeypatch.setattr(service, "run_adaptive", fake_run)
    _, report = run_adaptive_evaluation(path)
    assert all(row["status"] == "partial_range_labels" for row in report["sessions"])
    summary = report["summary"]["baseline_budget_0.5_replicate_0"]
    assert summary["complete_label_session_count"] == 0
    assert summary["partial_label_session_count"] == 2
    assert summary["scored_session_count"] == 2
    assert summary["top_k_regret_defined_session_count"] == 2
    assert summary["total_actual_usd"] == 0.0


def _evaluation_files(tmp_path, *, live=False, aggregate=None):
    raw = yaml.safe_load((ROOT / "configs/screening_2026-10-14.yaml").read_text())
    raw["workflow"] = "adaptive_range_v1"
    raw["agents"]["mode"] = "live" if live else "scripted"
    raw["output"]["runs_path"] = str(tmp_path / "runs")
    raw["adaptive"] = {"seed": 7}
    if live:
        raw["adaptive"].update(
            {
                "pricing_as_of": "2026-11-04",
                "prices": {
                    raw["agents"]["model"]: {
                        "input_per_million": 1,
                        "output_per_million": 4,
                    }
                },
            }
        )
    market = tmp_path / "market.parquet"
    pl.DataFrame(
        [
            {"session": day, "symbol": symbol, "high": high, "low": 100.0}
            for day in [date(2026, 10, 1), date(2026, 10, 2)]
            for symbol, high in [("A", 101.0), ("B", 110.0)]
        ]
    ).write_parquet(market)
    raw["data"]["market_bars"] = str(market)
    screening = tmp_path / "screening.yaml"
    screening.write_text(yaml.safe_dump(raw))
    spec = {
        "screening_config": str(screening),
        "evaluation": {
            "sessions": ["2026-10-02", "2026-10-01"],
            "policies": ["baseline", "scenario"],
            "budgets_usd": [0.5, 1],
            "replicates": 2,
        },
    }
    if aggregate is not None:
        spec["evaluation"]["aggregate_budget_usd"] = aggregate
    path = tmp_path / "evaluation.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def test_comparisons_execute_independent_chronological_trajectories_after_freeze(
    tmp_path,
    monkeypatch,
):
    import financial_strategy.adaptive.evaluation as evaluation
    import financial_strategy.adaptive.service as service

    calls = []
    original_labels = evaluation.derive_range_labels

    def fake_run(config, *, run_id, task_overrides):
        session = task_overrides["target_session"]
        calls.append(
            (
                config.adaptive.policy,
                config.adaptive.daily_budget_usd,
                config.adaptive.replicate_id,
                session,
                config.adaptive.memory_path,
            )
        )
        writer = RunWriter(Path(config.output["runs_path"]), run_id)
        predictions = [
            {"symbol": "A", "predicted_range": 0.01},
            {
                "symbol": "B",
                "predicted_range": (0.10 if config.adaptive.policy == "scenario" else 0.001),
            },
        ]
        writer.write_json("range_predictions.json", predictions)
        return writer.publish("completed"), predictions

    def checked_labels(*args, **kwargs):
        # Every label read follows publication of this exact day's prediction.
        assert len(list((tmp_path / "runs").glob("*/range_predictions.json"))) == len(calls)
        return original_labels(*args, **kwargs)

    monkeypatch.setattr(service, "run_adaptive", fake_run)
    monkeypatch.setattr(evaluation, "derive_range_labels", checked_labels)
    output, report = run_adaptive_evaluation(_evaluation_files(tmp_path))
    assert output.exists()
    assert len(calls) == 16
    assert len({call[4] for call in calls}) == 8
    for arm in {call[:3] for call in calls}:
        days = [call[3] for call in calls if call[:3] == arm]
        assert days == sorted(days)
    assert len(report["sessions"]) == 16
    assert all(row["status"] == "completed" for row in report["sessions"])
    assert report["report_top_ks"] == [3, 5, 10]
    assert set(report["sessions"][0]["metrics_by_k"]) == {"3", "5", "10"}
    for name in ["daily_costs.csv", "daily_metrics.csv", "daily_watchlists.csv"]:
        assert (output.parent / name).is_file()


@pytest.mark.parametrize("aggregate", [None, 1.0])
def test_live_evaluation_requires_adequate_aggregate_cap_before_execution(tmp_path, aggregate):
    path = _evaluation_files(tmp_path, live=True, aggregate=aggregate)
    with pytest.raises(ConfigError) as raised:
        run_adaptive_evaluation(path)
    assert raised.value.code in {"AGGREGATE_BUDGET_REQUIRED", "AGGREGATE_BUDGET_TOO_SMALL"}
    assert not (tmp_path / "runs").exists()


def test_default_reporting_cutoffs_preserve_smaller_existing_watchlists(tmp_path):
    path = _evaluation_files(tmp_path)
    spec = yaml.safe_load(path.read_text())
    screening = Path(spec["screening_config"])
    raw = yaml.safe_load(screening.read_text())
    raw["task"]["top_k"] = 2
    screening.write_text(yaml.safe_dump(raw))
    spec["evaluation"].update({"policies": ["baseline"], "budgets_usd": [0.5], "replicates": 1})
    path.write_text(yaml.safe_dump(spec))
    _, report = run_adaptive_evaluation(path)
    assert report["report_top_ks"] == [2]
    spec["evaluation"]["report_top_ks"] = [3]
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ConfigError) as raised:
        run_adaptive_evaluation(path)
    assert raised.value.code == "INVALID_REPORT_CUTOFF"
