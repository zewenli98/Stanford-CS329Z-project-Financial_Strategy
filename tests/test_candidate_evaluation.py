from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from financial_strategy.adaptive.evaluation import (
    AdaptiveEvaluationBody,
    summarize_adaptive_paired_comparisons,
)
from financial_strategy.config import BootstrapConfig, EvaluationConfig, load_mvp_config
from financial_strategy.evaluation.baselines import baseline_rankings
from financial_strategy.evaluation.bootstrap import paired_moving_block_bootstrap
from financial_strategy.evaluation.runner import summarize_paired_comparisons
from financial_strategy.ranking.screen import multi_channel_screen
from financial_strategy.ranking.ranker import rank_cards
from financial_strategy.schemas import EvidenceRecord, FeatureRecord, MacroContext, MacroEvidenceRecord, ResearchCard, SymbolMacroExposure

ROOT = Path(__file__).resolve().parents[1]


def _feature(symbol: str) -> FeatureRecord:
    return FeatureRecord(
        feature_id=symbol, symbol=symbol, snapshot_id="snapshot", formula_version="test",
        median_abs_oc_return_20d=0.01, realized_volatility_20d=0.02, return_5d=0.0,
        previous_session_relative_volume=1.0, distance_to_ma20=0.0,
        input_columns=["close"], input_sessions=[],
    )


def _evidence(symbol: str, available_at: datetime) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=f"e-{symbol}-{available_at.timestamp()}", scope="company", symbol=symbol,
        source_type="test", source_name="test", title="earnings", excerpt="earnings",
        published_at=available_at, first_seen_at=available_at, retrieved_at=available_at,
        available_at=available_at, content_hash=f"h-{symbol}-{available_at.timestamp()}",
        snapshot_id="snapshot", timestamp_confidence="high", event_types=["earnings"],
        event_status="released_before_cutoff",
    )


def _macro(available_at: datetime, actual: float = 2.0) -> MacroEvidenceRecord:
    return MacroEvidenceRecord(
        evidence_id=f"macro-{available_at.timestamp()}", scope="macro",
        source_type="test", source_name="test", title="CPI", excerpt="CPI",
        published_at=available_at, first_seen_at=available_at, retrieved_at=available_at,
        available_at=available_at, content_hash=f"mh-{available_at.timestamp()}",
        snapshot_id="snapshot", timestamp_confidence="high", event_type="cpi_yoy",
        event_status="released_before_cutoff", status="released_before_cutoff",
        actual=actual, consensus=1.0, consensus_source_name="survey",
        consensus_source_url="https://example.com", consensus_published_at=available_at,
        consensus_content_hash="consensus", affected_risk_factors=["inflation"],
    )


def test_bootstrap_is_deterministic_and_challenger_positive() -> None:
    challenger = {str(index): index + 1.0 for index in range(12)}
    baseline = {str(index): index for index in range(12)}
    left = paired_moving_block_bootstrap(challenger, baseline, samples=200)
    right = paired_moving_block_bootstrap(challenger, baseline, samples=200)
    assert left == right
    assert left["paired_session_count"] == 12
    assert left["mean_delta"] == pytest.approx(1.0)
    assert left["effective_block"] == 10


def test_disabled_residual_ranking_is_market_only() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    feature_a, feature_b = _feature("A"), _feature("B")
    cards = [
        ResearchCard(symbol="A", task_id="t", decision_cutoff=config.task.decision_cutoff, target_session=config.task.target_session, features=feature_a),
        ResearchCard(symbol="B", task_id="t", decision_cutoff=config.task.decision_cutoff, target_session=config.task.target_session, features=feature_b),
    ]
    exposures = {
        symbol: SymbolMacroExposure(symbol=symbol, sector="technology", exposure_method="sector_heuristic_v1", template_version="v1", exposures={})
        for symbol in "AB"
    }
    entries = rank_cards(cards, {"A": 0.9, "B": 0.1}, {"A": 0.9, "B": 0.1}, MacroContext(task_id="t", events=[], evidence_ids=[]), exposures, config.ranking, 2)
    assert [entry.symbol for entry in entries] == ["A", "B"]
    assert all(entry.score == pytest.approx(entry.market_score) for entry in entries)
    for variant in ("no_company_event", "no_macro"):
        ablated = rank_cards(
            cards, {"A": 0.9, "B": 0.1}, {"A": 0.9, "B": 0.1},
            MacroContext(task_id="t", events=[], evidence_ids=[]),
            exposures, config.ranking, 2, variant,
        )
        assert [entry.symbol for entry in ablated] == ["A", "B"]
        assert [entry.score for entry in ablated] == pytest.approx(
            [entry.score for entry in entries]
        )


def test_candidate_union_is_deterministic_and_audited() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    union = config.screening.candidate_union.model_validate({
        "enabled": True, "total_cap": 3,
        "market_top": {"limit": 2}, "recent_event": {"limit": 2},
        "random_exploration": {"limit": 2},
    })
    screening = config.screening.model_copy(update={"candidate_union": union})
    volatility = {symbol: value for symbol, value in zip("ABCD", (0.9, 0.8, 0.2, 0.1))}
    anomaly = {symbol: value for symbol, value in zip("ABCD", (0.9, 0.8, 0.2, 0.1))}
    cutoff = datetime(2026, 10, 14, 13, tzinfo=UTC)
    evidence = [_evidence("D", cutoff - timedelta(hours=1))]
    exposures = [
        SymbolMacroExposure(symbol=symbol, sector="technology", exposure_method="sector_heuristic_v1", template_version="v1", exposures={"inflation": 0.5})
        for symbol in "ABCD"
    ]
    first = multi_channel_screen(volatility, anomaly, 2, screening, config.ranking, evidence, [], exposures, cutoff)
    second = multi_channel_screen(volatility, anomaly, 2, screening, config.ranking, evidence, [], exposures, cutoff)
    assert first[0] == second[0]
    assert first[2] == second[2]
    assert all(first[2].provenance[symbol] for symbol in first[0])
    assert "D" in first[0]


def test_macro_channel_uses_cutoff_anomaly_and_exposure() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    union = config.screening.candidate_union.model_validate({
        "enabled": True, "total_cap": 1, "macro_exposure": {"limit": 2},
    })
    screening = config.screening.model_copy(update={"candidate_union": union})
    cutoff = datetime(2026, 10, 14, 13, tzinfo=UTC)
    values = {"A": 0.9, "B": 0.1}
    exposures = [
        SymbolMacroExposure(symbol="A", sector="technology", exposure_method="sector_heuristic_v1", template_version="v1", exposures={"inflation": 0.1}),
        SymbolMacroExposure(symbol="B", sector="technology", exposure_method="sector_heuristic_v1", template_version="v1", exposures={"inflation": 1.0}),
    ]
    before = multi_channel_screen(values, values, 1, screening, config.ranking, [], [_macro(cutoff - timedelta(seconds=1))], exposures, cutoff)
    after = multi_channel_screen(values, values, 1, screening, config.ranking, [], [_macro(cutoff + timedelta(seconds=1))], exposures, cutoff)
    assert before[0] == ["B"]
    assert after[0] == []


def test_ranking_paired_summary_keeps_replicates_separate() -> None:
    records = []
    for replicate, delta in (("0", 0.1), ("1", 0.3)):
        for session in ("2026-01-01", "2026-01-02"):
            records.append({
                "replicate_id": replicate, "session": session,
                "metrics": {
                    "no_agent_ranker": {"ndcg_at_k": 0.5, "mean_absolute_return_at_k": 0.2, "precision_at_k": 0.4},
                    "full_system": {"ndcg_at_k": 0.5 + delta, "mean_absolute_return_at_k": 0.2, "precision_at_k": 0.4},
                    "deterministic_event_baseline": {"ndcg_at_k": 0.6, "mean_absolute_return_at_k": 0.2, "precision_at_k": 0.4},
                },
            })
    summary = summarize_paired_comparisons(records, BootstrapConfig(samples=100))
    ndcg = summary["full_system_vs_no_agent_ranker"]["ndcg_at_k"]
    assert [item["paired_session_count"] for item in ndcg["replicates"]] == [2, 2]
    assert ndcg["aggregate_replicate_deltas"]["mean"] == pytest.approx(0.2)
    assert ndcg["aggregate_replicate_deltas"]["min"] == pytest.approx(0.1)


def test_adaptive_paired_summary_orients_regret_and_aggregates() -> None:
    records = []
    for replicate, challenger_regret in (("0", 0.1), ("1", 0.2)):
        for policy, regret in (("baseline", 0.4), ("scenario", challenger_regret)):
            for session in ("2026-01-01", "2026-01-02"):
                records.append({
                    "policy": policy, "budget_usd": 1.0, "replicate_id": replicate,
                    "session": session, "status": "completed",
                    "metrics": {"top_k_regret": regret, "mean_selected_range": 0.1, "ndcg_at_k": 0.5},
                })
    body = AdaptiveEvaluationBody(
        sessions=[datetime(2026, 1, 1).date()], policies=["baseline", "scenario"],
        budgets_usd=[1.0], bootstrap_samples=100,
    )
    summary = summarize_adaptive_paired_comparisons(records, body)
    regret = summary["scenario_budget_1_vs_baseline"]["top_k_regret"]
    assert [item["mean_delta"] for item in regret["replicates"]] == pytest.approx([0.3, 0.2])
    assert regret["aggregate_replicate_deltas"]["mean"] == pytest.approx(0.25)


def test_deterministic_event_baseline_fails_closed_after_cutoff() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    cutoff = datetime(2026, 10, 14, 13, tzinfo=UTC)
    features = {symbol: _feature(symbol) for symbol in "AB"}
    volatility = {"A": 0.8, "B": 0.2}
    anomaly = dict(volatility)
    future = baseline_rankings(features, volatility, anomaly, [_evidence("B", cutoff + timedelta(seconds=1))], config, cutoff)
    prior = baseline_rankings(features, volatility, anomaly, [_evidence("B", cutoff - timedelta(seconds=1))], config, cutoff)
    assert future["deterministic_event_baseline"][0] == "A"
    assert prior["deterministic_event_baseline"][0] == "B"


def test_evaluation_bootstrap_config_is_strict() -> None:
    with pytest.raises(ValueError, match="block sizes"):
        EvaluationConfig.model_validate({
            "screening_config": "x.yaml",
            "evaluation": {
                "sessions": ["2026-01-01"], "cutoff_time": "09:00:00", "timezone": "UTC",
                "labels": "labels.parquet", "large_move_threshold": 0.04, "variants": ["full_system"],
                "ndcg": {"relevance": "intraday_range"}, "bootstrap": {"block_sizes": [5, 5]},
            },
        })
