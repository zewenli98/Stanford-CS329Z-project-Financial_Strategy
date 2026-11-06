from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

from financial_strategy.config import load_mvp_config
from financial_strategy.macro import build_macro_interactions
from financial_strategy.ranking.ranker import macro_interaction_residual
from financial_strategy.schemas import (
    FeatureRecord,
    MacroContext,
    MacroEvidenceRecord,
    ResearchCard,
    SymbolMacroExposure,
)
from financial_strategy.verification.deterministic import verify_macro_context
from scripts.backfill_macro_history import (
    _fomc_dates,
    _match_rows,
    _match_tradingview_rows,
    _parse_bls_release_actual,
)

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def _event(
    event_date: date,
    *,
    actual: float = 2.1,
    consensus: float = 2.0,
    available_offset: timedelta = timedelta(),
) -> MacroEvidenceRecord:
    event_time = datetime(
        event_date.year, event_date.month, event_date.day, 13, 30, tzinfo=UTC
    )
    consensus_time = event_time - timedelta(hours=1)
    return MacroEvidenceRecord(
        evidence_id=f"cpi_{event_date}",
        scope="macro",
        source_type="test",
        source_name="test",
        source_url="https://example.com/release",
        title="CPI",
        excerpt="CPI release",
        event_time=event_time,
        published_at=event_time,
        first_seen_at=event_time,
        retrieved_at=event_time + timedelta(days=100),
        available_at=event_time + available_offset,
        content_hash="hash",
        snapshot_id="snapshot",
        timestamp_confidence="high",
        event_type="cpi_yoy",
        event_status="released_before_cutoff",
        status="released_before_cutoff",
        actual=actual,
        consensus=consensus,
        consensus_source_name="survey",
        consensus_source_url="https://example.com/consensus",
        consensus_published_at=consensus_time,
        consensus_content_hash="consensus-hash",
        unit="percentage_points",
        affected_risk_factors=["inflation"],
    )


def _feature() -> FeatureRecord:
    return FeatureRecord(
        feature_id="feature",
        symbol="AAA",
        snapshot_id="snapshot",
        formula_version="test",
        input_columns=["open", "close"],
        input_sessions=[date(2024, 1, day) for day in range(1, 22)],
        return_5d=0.0,
        realized_volatility_20d=0.01,
        median_abs_oc_return_20d=0.01,
        previous_session_relative_volume=1.0,
        distance_to_ma20=0.0,
    )


def test_macro_sensitivity_uses_only_prior_releases() -> None:
    history = [
        _event(date(2024, 1, day), actual=2.0 + day / 100)
        for day in range(1, 9)
    ]
    target = _event(date(2024, 2, 1), actual=2.1)
    future = _event(date(2024, 2, 2), actual=20.0)
    returns = {}
    for event in history:
        session = event.event_time.date().isoformat()
        surprise = (event.actual - event.consensus) / 0.1
        returns[session] = {
            "AAA": surprise * 0.02,
            "BBB": 0.0,
            "CCC": 0.0,
        }
    sectors = {"AAA": "technology", "BBB": "technology", "CCC": "technology"}
    exposures = {
        symbol: SymbolMacroExposure(
            symbol=symbol,
            sector="technology",
            exposure_method="sector_heuristic_v1",
            template_version="test",
            exposures={"inflation": 1.0},
        )
        for symbol in sectors
    }
    interactions = build_macro_interactions(
        target_session=date(2024, 2, 1),
        target_events=[target],
        history_events=[*history, future],
        returns=returns,
        sectors=sectors,
        exposures=exposures,
        surprise_scales={"cpi_yoy": 0.1},
    )
    assert interactions["AAA"][0].sensitivity.sample_count == 8
    assert interactions["AAA"][0].sensitivity.fallback_level == "symbol"
    assert interactions["AAA"][0].expected_abs_impact > 0
    assert interactions["AAA"][0].expected_abs_impact > interactions["BBB"][
        0
    ].expected_abs_impact


def test_macro_context_verifier_recomputes_surprise() -> None:
    event = _event(date(2024, 2, 1))
    context = MacroContext(
        task_id="task",
        events=[event],
        evidence_ids=[event.evidence_id],
        target_session=date(2024, 2, 1),
        decision_cutoff=event.available_at + timedelta(minutes=30),
        assessments=[
            {
                "event_type": "cpi_yoy",
                "evidence_ids": [event.evidence_id],
                "target_session_relevance": "high",
                "surprise_direction": "positive",
                "standardized_surprise": 999.0,
                "affected_risk_factors": ["inflation"],
                "target_session_timing": "before_open",
                "confidence": "high",
            }
        ],
    )
    report = verify_macro_context(context, [event], {"cpi_yoy": 0.1})
    assert report.verdict == "REVISE"
    assert {issue.code for issue in report.issues} == {"INVALID_NUMERIC_CLAIM"}


def test_macro_residual_is_independent_of_company_assessment() -> None:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    history = [_event(date(2024, 1, day)) for day in range(1, 9)]
    returns = {
        event.event_time.date().isoformat(): {"AAA": 0.02, "BBB": 0.0}
        for event in history
    }
    exposures = {
        symbol: SymbolMacroExposure(
            symbol=symbol,
            sector="technology",
            exposure_method="sector_heuristic_v1",
            template_version="test",
            exposures={"inflation": 1.0},
        )
        for symbol in ("AAA", "BBB")
    }
    interaction = build_macro_interactions(
        target_session=date(2024, 2, 1),
        target_events=[_event(date(2024, 2, 1))],
        history_events=history,
        returns=returns,
        sectors={"AAA": "technology", "BBB": "technology"},
        exposures=exposures,
        surprise_scales={"cpi_yoy": 0.1},
    )["AAA"]
    card = ResearchCard(
        symbol="AAA",
        task_id="task",
        decision_cutoff=datetime(2024, 2, 1, 14, tzinfo=UTC),
        target_session=date(2024, 2, 1),
        features=_feature(),
        macro_interactions=interaction,
    )
    assert card.catalyst_assessment is None
    assert macro_interaction_residual(card, config.ranking) > 0


def test_backfill_matches_us_fomc_and_annual_cpi_rows() -> None:
    rows = [
        {
            "eventName": "CPI",
            "actual": "0.3%",
            "consensus": "0.2%",
        },
        {
            "eventName": "CPI",
            "actual": "3.4%",
            "consensus": "3.2%",
        },
        {
            "eventName": "Fed Interest Rate Decision",
            "actual": "5.50%",
            "consensus": "5.50%",
        },
    ]
    assert _match_rows(rows, "cpi_yoy")[0]["actual"] == "3.4%"
    assert _match_rows(rows, "fed_rate")[0]["actual"] == "5.50%"


def test_backfill_uses_only_fomc_decision_documents() -> None:
    html = """
    <a href="/monetarypolicy/files/monetary20250730a1.pdf">PDF</a>
    <a href="/newsevents/pressreleases/monetary20250822a.htm">
      Statement on Longer-Run Goals and Monetary Policy Strategy
    </a>
    """

    assert _fomc_dates(html) == {date(2025, 7, 30)}


def test_backfill_matches_tradingview_fallback_rows_by_ticker() -> None:
    rows = [
        {
            "ticker": "ECONOMICS:USIRYY",
            "actual": 3.7,
            "forecast": 3.6,
            "previous": 3.7,
            "date": "2023-10-12T12:30:00.000Z",
        },
        {
            "ticker": "ECONOMICS:USCPI",
            "actual": 307.789,
            "forecast": 307.386,
            "previous": 304.127,
            "date": "2023-10-12T12:30:00.000Z",
        },
    ]

    matched = _match_tradingview_rows(rows, "cpi_yoy")
    assert matched[0]["actual"] == "3.7%"
    assert matched[0]["consensus"] == "3.6%"


def test_backfill_parses_release_time_actuals_from_bls_archive() -> None:
    employment = """
    Total nonfarm payroll employment increased by 187,000 in August,
    and the unemployment rate rose to 3.8 percent.
    """
    cpi = "The all items index increased 3.7 percent before seasonal adjustment."

    assert _parse_bls_release_actual(
        employment, "nonfarm_payrolls"
    ) == 187
    assert _parse_bls_release_actual(employment, "unemployment") == 3.8
    assert _parse_bls_release_actual(cpi, "cpi_yoy") == 3.7
