from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from financial_strategy.config import load_mvp_config
from financial_strategy.errors import ConfigError
from financial_strategy.evaluation.holdout import (
    assess_holdout_promotion,
    mark_holdout_consumed,
    prepare_future_holdout,
    require_holdout_ready,
)
from financial_strategy.features.engine import compute_features
from financial_strategy.ranking.ranker import rank_cards
from financial_strategy.schemas import (
    CatalystAssessment,
    Eligibility,
    EvidenceRecord,
    FeatureRecord,
    MacroContext,
    ResearchCard,
    SymbolMacroExposure,
    VerificationIssue,
    VerificationReport,
)
from financial_strategy.tuning.learning import (
    fit_nonnegative_linear_ranker,
    fit_pairwise_ranker,
)
from financial_strategy.tuning.runner import (
    _block_bootstrap_ci,
    _expanding_test_folds,
)
from financial_strategy.verification.deterministic import verify_card

ROOT = Path(__file__).resolve().parents[1]


def test_expanding_folds_leave_one_session_embargo() -> None:
    sessions = [f"s{index:03}" for index in range(101)]
    folds = _expanding_test_folds(sessions, minimum_train=40, test_size=20, embargo=1)
    assert folds[0] == sessions[41:61]
    assert sessions[40] not in folds[0]
    assert folds[1] == sessions[61:81]


def test_moving_block_bootstrap_is_deterministic_and_finite() -> None:
    first = _block_bootstrap_ci([0.1, -0.2, 0.3, 0.0], 329, 200, 2)
    second = _block_bootstrap_ci([0.1, -0.2, 0.3, 0.0], 329, 200, 2)
    assert first == second
    assert first is not None and first[0] <= first[1]


def test_regularized_rankers_respect_nonnegative_simplex() -> None:
    features = np.asarray(
        [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9]],
        dtype=float,
    )
    targets = np.asarray([1.0, 0.8, 0.1, 0.2])
    anchor = np.asarray([0.5, 0.5])
    linear = fit_nonnegative_linear_ranker(features, targets, anchor)
    pairwise = fit_pairwise_ranker(
        features,
        targets,
        np.asarray(["a", "a", "b", "b"]),
        anchor,
    )
    for weights in (linear, pairwise):
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights >= 0)
    assert linear[0] > linear[1]
    assert pairwise[0] > pairwise[1]


def test_enhanced_features_include_industry_residual_and_missing_marker() -> None:
    sessions: list[date] = []
    current = date(2026, 7, 1)
    while len(sessions) < 61:
        if current.weekday() < 5:
            sessions.append(current)
        current += timedelta(days=1)
    rows = []
    for symbol, slope in (("A", 0.5), ("B", 0.2)):
        for index, session in enumerate(sessions):
            close = 100 + slope * index + (index % 3)
            rows.append(
                {
                    "symbol": symbol,
                    "session": session,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "volume": 1_000 + index,
                    "snapshot_id": "snapshot",
                }
            )
    features, rejected = compute_features(
        pl.DataFrame(rows),
        sessions[-21:],
        symbol_sectors={"A": "technology", "B": "technology"},
    )
    assert rejected == {}
    assert features["A"].realized_volatility_60d is not None
    assert features["A"].industry_residual_volatility_20d is not None
    assert features["A"].day_of_week == sessions[-1].weekday()
    assert features["A"].missing_feature_count == 0


def _feature() -> FeatureRecord:
    return FeatureRecord(
        feature_id="feature",
        symbol="AAPL",
        snapshot_id="snapshot",
        formula_version="v2",
        median_abs_oc_return_20d=0.01,
        realized_volatility_20d=0.02,
        return_5d=0.03,
        previous_session_relative_volume=1.2,
        distance_to_ma20=0.01,
        input_columns=["close"],
        input_sessions=[date(2026, 10, 30)],
    )


def test_semantic_verdict_does_not_change_market_only_score() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    ranking = config.ranking.model_copy(update={"minimum_score": 0.0})
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        deterministic_verification=VerificationReport(verdict="PASS"),
        semantic_verification=VerificationReport(
            verdict="REVISE",
            issues=[
                VerificationIssue(
                    code="BAD_INFERENCE",
                    severity="blocking",
                    field="catalyst_assessment",
                    required_action="revise_assessment",
                )
            ],
        ),
        eligibility=Eligibility(eligible=True),
    )
    context = MacroContext(task_id="task", events=[], evidence_ids=[])
    exposure = SymbolMacroExposure(
        symbol="AAPL",
        sector="technology",
        exposure_method="sector_heuristic_v1",
        template_version="v1",
        exposures={
            "inflation": 0.0,
            "interest_rates": 0.0,
            "oil_prices": 0.0,
            "usd": 0.0,
        },
    )
    full = rank_cards(
        [card],
        {"AAPL": 0.8},
        {"AAPL": 0.6},
        context,
        {"AAPL": exposure},
        ranking,
        1,
    )
    no_semantic = rank_cards(
        [card],
        {"AAPL": 0.8},
        {"AAPL": 0.6},
        context,
        {"AAPL": exposure},
        ranking,
        1,
        variant="no_semantic_verifier",
    )
    assert [entry.symbol for entry in full] == ["AAPL"]
    assert full[0].score == pytest.approx(no_semantic[0].score)
    failed = card.model_copy(
        update={"deterministic_verification": VerificationReport(verdict="ABSTAIN")}
    )
    retained = rank_cards(
        [failed],
        {"AAPL": 0.8},
        {"AAPL": 0.6},
        context,
        {"AAPL": exposure},
        ranking,
        1,
    )
    assert [entry.symbol for entry in retained] == ["AAPL"]


def test_verifier_code_normalization_and_warning_only_pass() -> None:
    report = VerificationReport.model_validate(
        {
            "verdict": "REVISE",
            "issues": [
                {
                    "code": "FREE_FORM_CODE",
                    "severity": "blocking",
                    "field": "rationale",
                    "required_action": "rewrite",
                }
            ],
        }
    )
    assert report.verdict == "PASS"
    assert report.issues[0].code == "SEMANTIC_WARNING"
    assert report.issues[0].detail_code == "FREE_FORM_CODE"


def test_catalyst_evidence_is_part_of_deterministic_verification() -> None:
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 11, 1, tzinfo=UTC),
        target_session=date(2026, 11, 2),
        features=_feature(),
        catalyst_assessment=CatalystAssessment(
            target_session_relevance="high",
            materiality="high",
            expected_abs_impact_bucket="over_5pct",
            evidence_ids=["missing"],
        ),
    )
    report = verify_card(card, [])
    assert {issue.code for issue in report.issues} == {"MISSING_EVIDENCE"}

    timestamp = datetime(2026, 10, 31, tzinfo=UTC)
    evidence = EvidenceRecord(
        evidence_id="missing",
        scope="company",
        symbol="AAPL",
        source_type="company_release",
        source_name="test",
        title="Catalyst",
        excerpt="Catalyst",
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash="hash",
        snapshot_id="snapshot",
        timestamp_confidence="high",
    )
    assert verify_card(card, [evidence]).issues == []


def test_future_holdout_cannot_be_consumed_early_or_twice(tmp_path: Path) -> None:
    challenger = tmp_path / "challenger.yaml"
    challenger.write_text("frozen: true\n")
    manifest_path = tmp_path / "holdout.json"
    early = prepare_future_holdout(
        manifest_path,
        challenger,
        start_session=date(2026, 10, 15),
        as_of=date(2026, 10, 30),
        minimum_sessions=60,
    )
    assert early["session_count"] == 12
    with pytest.raises(ConfigError, match="12/60"):
        require_holdout_ready(manifest_path)

    ready = prepare_future_holdout(
        manifest_path,
        challenger,
        start_session=date(2026, 10, 15),
        as_of=date(2027, 2, 1),
        minimum_sessions=60,
    )
    assert ready["ready_for_one_shot_evaluation"]
    require_holdout_ready(manifest_path)
    report = tmp_path / "report.json"
    report.write_text("{}")
    mark_holdout_consumed(
        manifest_path,
        evaluation_report=report,
        consumed_at="2027-02-01T00:00:00+00:00",
    )
    with pytest.raises(ConfigError) as error:
        require_holdout_ready(manifest_path)
    assert error.value.code == "HOLDOUT_ALREADY_CONSUMED"


def test_holdout_promotion_requires_positive_paired_lower_bound() -> None:
    sessions = [
        {
            "status": "completed",
            "metrics": {
                "full_system": {"ndcg_at_k": 0.7, "selection_coverage": 1.0},
                "no_agent_ranker": {"ndcg_at_k": 0.6, "selection_coverage": 1.0},
            },
            "reliability": {
                "time_boundary_violation_rate": 0.0,
                "unsupported_claim_rate": 0.0,
            },
        }
        for _ in range(60)
    ]
    report = {
        "sessions": sessions,
        "summary": {
            "variants": {
                "full_system": {"average_mean_absolute_return_at_k": 0.03},
                "no_agent_ranker": {"average_mean_absolute_return_at_k": 0.02},
            }
        },
    }
    decision = assess_holdout_promotion(report)
    assert decision["promote"]
    assert decision["paired_ndcg_delta_95ci"][0] > 0

    sessions[0]["reliability"]["unsupported_claim_rate"] = 0.01
    assert not assess_holdout_promotion(report)["promote"]
