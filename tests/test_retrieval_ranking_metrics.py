import math
from datetime import UTC, datetime
from pathlib import Path

from financial_strategy.config import load_mvp_config
from financial_strategy.evaluation.metrics import evaluate_ranking
from financial_strategy.ranking.ranker import average_percentiles
from financial_strategy.retrieval.bm25 import search
from financial_strategy.schemas import (
    Claim,
    EventRecord,
    EvidenceRecord,
    FeatureRecord,
    ResearchCard,
)
from financial_strategy.verification.deterministic import verify_card

ROOT = Path(__file__).resolve().parents[1]


def _evidence(evidence_id: str, text: str) -> EvidenceRecord:
    timestamp = datetime(2026, 10, 18, tzinfo=UTC)
    return EvidenceRecord(
        evidence_id=evidence_id,
        scope="company",
        symbol="AAPL",
        source_type="company_release",
        source_name="unit_test",
        title=text,
        excerpt=text,
        published_at=timestamp,
        first_seen_at=timestamp,
        retrieved_at=timestamp,
        available_at=timestamp,
        content_hash=evidence_id,
        snapshot_id="snapshot",
        timestamp_confidence="high",
    )


def test_bm25_ties_use_evidence_id_order() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    hits = search(
        [_evidence("ev_b", "earnings"), _evidence("ev_a", "earnings")],
        "earnings",
        config.retrieval,
        scope="company",
        symbol="AAPL",
    )
    assert [hit.evidence.evidence_id for hit in hits] == ["ev_a", "ev_b"]


def test_average_percentiles_handle_ties_and_singletons() -> None:
    assert average_percentiles({"A": 1.0}) == {"A": 0.5}
    values = average_percentiles({"A": 1.0, "B": 2.0, "C": 2.0, "D": 4.0})
    assert values["A"] == 0
    assert values["B"] == values["C"] == 0.5
    assert values["D"] == 1


def test_metrics_zero_fill_unselected_positions_and_use_five_percent_threshold() -> None:
    metrics = evaluate_ranking(
        ["A"],
        {"A": 0.06, "B": 0.04},
        k=2,
        eligible_count=2,
        large_move_threshold=0.05,
    )
    assert metrics["precision_at_k"] == 0.5
    assert metrics["mean_absolute_return_at_k"] == 0.03
    assert metrics["selection_coverage"] == 0.5


def test_empty_watchlist_and_zero_idcg() -> None:
    empty = evaluate_ranking(
        [],
        {"A": 0.06},
        k=1,
        eligible_count=1,
        large_move_threshold=0.05,
    )
    assert empty["precision_at_k"] == 0
    assert empty["ndcg_at_k"] == 0
    zero = evaluate_ranking(
        [],
        {"A": 0.0},
        k=1,
        eligible_count=1,
        large_move_threshold=0.05,
    )
    assert zero["ndcg_at_k"] is None


def test_missing_label_keeps_its_original_rank_position() -> None:
    metrics = evaluate_ranking(
        ["MISSING", "A"],
        {"A": 0.10, "B": 0.05},
        k=2,
        eligible_count=2,
        large_move_threshold=0.05,
    )
    expected_dcg = 0.10 / math.log2(3)
    ideal_dcg = 0.10 + 0.05 / math.log2(3)
    assert metrics["ndcg_at_k"] == expected_dcg / ideal_dcg


def test_deterministic_verifier_rejects_unsupported_event_status() -> None:
    evidence = _evidence("ev", "earnings").model_copy(
        update={
            "event_types": ["earnings"],
            "event_status": "scheduled_after_close",
        }
    )
    feature = FeatureRecord(
        feature_id="feature_A",
        symbol="AAPL",
        snapshot_id="snapshot",
        formula_version="v1",
        median_abs_oc_return_20d=0.01,
        realized_volatility_20d=0.01,
        return_5d=0.01,
        previous_session_relative_volume=1.0,
        distance_to_ma20=0.01,
        input_columns=["close"],
        input_sessions=[datetime(2026, 10, 16, tzinfo=UTC).date()],
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 10, 19, 13, tzinfo=UTC),
        target_session=datetime(2026, 10, 19, tzinfo=UTC).date(),
        features=feature,
        events=[
            EventRecord(
                event_id="event",
                event_types=["earnings"],
                status="released_before_cutoff",
                confidence="high",
                evidence_ids=["ev"],
            )
        ],
    )
    report = verify_card(card, [evidence])
    assert {issue.code for issue in report.issues} == {"UNSUPPORTED_EVENT_STATUS"}


def test_deterministic_verifier_accepts_feature_backed_observed_fact() -> None:
    feature = FeatureRecord(
        feature_id="feature_A",
        symbol="AAPL",
        snapshot_id="snapshot",
        formula_version="v1",
        median_abs_oc_return_20d=0.01,
        realized_volatility_20d=0.01,
        return_5d=0.01,
        previous_session_relative_volume=1.0,
        distance_to_ma20=0.01,
        input_columns=["close"],
        input_sessions=[datetime(2026, 10, 16, tzinfo=UTC).date()],
    )
    card = ResearchCard(
        symbol="AAPL",
        task_id="task",
        decision_cutoff=datetime(2026, 10, 19, 13, tzinfo=UTC),
        target_session=datetime(2026, 10, 19, tzinfo=UTC).date(),
        features=feature,
        claims=[
            Claim(
                claim_id="feature_claim",
                text="The five-day return was 1%.",
                claim_type="observed_fact",
                feature_ids=["feature_A"],
            )
        ],
    )

    assert verify_card(card, []).issues == []
