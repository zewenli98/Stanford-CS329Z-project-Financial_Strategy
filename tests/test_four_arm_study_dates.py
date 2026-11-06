from collections import Counter

import pytest

from scripts.prepare_four_arm_study import (
    SessionProfile,
    assign_time_bins,
    select_study_dates,
    validate_selection,
)


def _profile(session: str, evidence: int, *, macro: int = 0) -> SessionProfile:
    return SessionProfile(
        session=session,
        symbols_with_evidence=evidence,
        important_evidence_count=evidence,
        symbols_with_document=evidence,
        scheduled_macro_count=macro,
        scheduled_macro_types=("cpi_yoy",) if macro else (),
        macro_released_before_cutoff=macro,
    )


def _training_profiles() -> tuple[list[str], dict[str, SessionProfile]]:
    sessions: list[str] = []
    profiles: dict[str, SessionProfile] = {}
    for bin_index in range(5):
        for offset in range(12):
            session = f"2024-{bin_index + 1:02d}-{offset + 1:02d}"
            if offset < 3:
                profile = _profile(session, 0, macro=1)
            else:
                profile = _profile(session, offset - 3)
            sessions.append(session)
            profiles[session] = profile
    return sessions, profiles


def test_time_bins_are_contiguous_and_differ_by_one() -> None:
    labels = assign_time_bins([f"s{index:02d}" for index in range(7)], 3)

    assert labels == [0, 0, 0, 1, 1, 2, 2]


def test_selection_follows_the_registered_design_without_overlap() -> None:
    sessions, profiles = _training_profiles()
    selected = select_study_dates(sessions, profiles)
    smoke = selected["smoke"]
    development = selected["development"]

    validate_selection(sessions, ["2026-07-16"], smoke, development)
    assert [item["stratum"] for item in smoke] == ["low", "medium", "high", "high", "macro"]
    assert Counter(item["stratum"] for item in development) == Counter(
        low=5,
        medium=5,
        high=5,
        macro=5,
    )
    for item in development:
        if item["stratum"] != "high":
            continue
        low = next(
            candidate
            for candidate in development
            if candidate["time_bin"] == item["time_bin"] and candidate["stratum"] == "low"
        )
        assert item["symbols_with_evidence"] > low["symbols_with_evidence"]


def test_validation_window_cannot_be_reused_for_development() -> None:
    sessions, profiles = _training_profiles()
    selected = select_study_dates(sessions, profiles)

    with pytest.raises(ValueError, match="final evaluation window"):
        validate_selection(
            sessions,
            [selected["smoke"][0]["session"]],
            selected["smoke"],
            selected["development"],
        )
