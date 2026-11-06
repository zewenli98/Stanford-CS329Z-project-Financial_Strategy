"""Freeze label-blind smoke and development dates for the four-arm study.

Selection reads only information available at 09:00 America/New_York: historical
market features, company evidence already available, and the scheduled macro
calendar. It does not read target-session high, low, labels, or rank metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
import yaml

from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.ranking.ranker import (
    ABSOLUTE_MARKET_FIELDS,
    MARKET_FEATURE_FIELDS,
    average_percentiles,
)
from financial_strategy.schemas.models import COMPANY_EVENT_TYPES

ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")
DECISION_TIME = time(9, 0)
EVENT_LOOKBACK_SESSIONS = 7
CANDIDATE_COUNT = 20
TEMPORAL_BINS = 5
SCHEDULED_MACRO_TYPES = frozenset({"cpi_yoy", "nonfarm_payrolls", "unemployment", "fed_rate"})
SMOKE_BY_BIN = ("low", "medium", "high", "high", "macro")
DEVELOPMENT_STRATA = ("low", "medium", "high", "macro")
IMPORTANT_EVENT_TYPES = frozenset(COMPANY_EVENT_TYPES)


@dataclass(frozen=True)
class SessionProfile:
    session: str
    symbols_with_evidence: int
    important_evidence_count: int
    symbols_with_document: int
    scheduled_macro_count: int
    scheduled_macro_types: tuple[str, ...]
    macro_released_before_cutoff: int
    market_candidate_symbols: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidencePoint:
    available_at: datetime
    release_session: date
    event_types: frozenset[str]
    has_document: bool


def assign_time_bins(sessions: list[str], bin_count: int) -> list[int]:
    """Split an ordered session list into contiguous bins differing by at most one."""

    if bin_count < 1 or len(sessions) < bin_count:
        raise ValueError("not enough ordered sessions for the requested time bins")
    counts = [len(sessions) // bin_count] * bin_count
    for index in range(len(sessions) % bin_count):
        counts[index] += 1
    labels: list[int] = []
    for bin_index, count in enumerate(counts):
        labels.extend([bin_index] * count)
    return labels


def classify_strata(
    sessions: list[str],
    profiles: dict[str, SessionProfile],
    bins: list[int],
) -> dict[str, str]:
    """Label every session within its own time bin, without using outcomes."""

    grouped: dict[int, list[SessionProfile]] = defaultdict(list)
    for session, bin_index in zip(sessions, bins, strict=True):
        grouped[bin_index].append(profiles[session])
    labels: dict[str, str] = {}
    for bin_index, rows in grouped.items():
        macro = [row for row in rows if row.scheduled_macro_count > 0]
        ordinary = [row for row in rows if row.scheduled_macro_count == 0]
        if not macro or len(ordinary) < 3:
            raise ValueError(f"time bin {bin_index} cannot fill all four event strata")
        ordered = sorted(
            ordinary,
            key=lambda row: (
                row.symbols_with_evidence,
                row.important_evidence_count,
                row.symbols_with_document,
                row.session,
            ),
        )
        low_end = len(ordered) // 3
        middle_end = 2 * len(ordered) // 3
        if low_end == 0 or middle_end == low_end or middle_end == len(ordered):
            raise ValueError(f"time bin {bin_index} has an empty company-event tertile")
        lowest = ordered[0]
        highest = ordered[-1]
        if (
            highest.symbols_with_evidence <= lowest.symbols_with_evidence
            and highest.important_evidence_count <= lowest.important_evidence_count
        ):
            raise ValueError(f"time bin {bin_index} has no pre-cutoff evidence variation")
        for row in macro:
            labels[row.session] = "macro"
        for index, row in enumerate(ordered):
            if index < low_end:
                labels[row.session] = "low"
            elif index < middle_end:
                labels[row.session] = "medium"
            else:
                labels[row.session] = "high"
    return labels


def median_profile(rows: list[SessionProfile]) -> SessionProfile:
    """Return the chronological middle session, not the most extreme event day."""

    if not rows:
        raise ValueError("cannot select a median from an empty date cell")
    ordered = sorted(rows, key=lambda row: row.session)
    return ordered[len(ordered) // 2]


def select_study_dates(
    train_sessions: list[str],
    profiles: dict[str, SessionProfile],
    *,
    bin_count: int = TEMPORAL_BINS,
) -> dict[str, object]:
    """Choose 5 smoke dates and 20 development dates from training sessions only."""

    if len(SMOKE_BY_BIN) != bin_count:
        raise ValueError("smoke assignments must contain one entry per time bin")
    missing = [session for session in train_sessions if session not in profiles]
    if missing:
        raise ValueError(f"missing session profile, first missing {missing[0]}")
    bins = assign_time_bins(train_sessions, bin_count)
    strata = classify_strata(train_sessions, profiles, bins)
    cells: dict[tuple[int, str], list[SessionProfile]] = defaultdict(list)
    for session, bin_index in zip(train_sessions, bins, strict=True):
        cells[(bin_index, strata[session])].append(profiles[session])

    smoke: list[tuple[int, str, SessionProfile]] = []
    for bin_index, stratum in enumerate(SMOKE_BY_BIN):
        chosen = median_profile(cells[(bin_index, stratum)])
        smoke.append((bin_index, stratum, chosen))
        cells[(bin_index, stratum)] = [
            row for row in cells[(bin_index, stratum)] if row.session != chosen.session
        ]

    development: list[tuple[int, str, SessionProfile]] = []
    for bin_index in range(bin_count):
        for stratum in DEVELOPMENT_STRATA:
            development.append((bin_index, stratum, median_profile(cells[(bin_index, stratum)])))

    bin_ranges = []
    for bin_index in range(bin_count):
        members = [
            session
            for session, candidate in zip(train_sessions, bins, strict=True)
            if candidate == bin_index
        ]
        bin_ranges.append(
            {
                "index": bin_index,
                "first_session": members[0],
                "last_session": members[-1],
                "session_count": len(members),
            }
        )
    return {
        "time_bins": bin_ranges,
        "smoke": [_selected_record(item) for item in smoke],
        "development": [_selected_record(item) for item in development],
        "bin_by_session": dict(zip(train_sessions, bins, strict=True)),
        "stratum_by_session": strata,
    }


def validate_selection(
    train_sessions: list[str],
    validation_sessions: list[str],
    smoke: list[dict],
    development: list[dict],
) -> None:
    train = set(train_sessions)
    validation = set(validation_sessions)
    if train & validation:
        raise ValueError("training sessions overlap the final evaluation window")
    smoke_sessions = [item["session"] for item in smoke]
    development_sessions = [item["session"] for item in development]
    if len(smoke_sessions) != 5 or len(set(smoke_sessions)) != 5:
        raise ValueError("smoke selection must contain five distinct sessions")
    if len(development_sessions) != 20 or len(set(development_sessions)) != 20:
        raise ValueError("development selection must contain twenty distinct sessions")
    selected = set(smoke_sessions) | set(development_sessions)
    if set(smoke_sessions) & set(development_sessions):
        raise ValueError("smoke and development sessions overlap")
    if not selected <= train:
        raise ValueError("a selected session is outside the training window")
    if selected & validation:
        raise ValueError("a selected session falls inside the final evaluation window")
    if Counter(item["stratum"] for item in smoke) != Counter(
        {"low": 1, "medium": 1, "high": 2, "macro": 1}
    ):
        raise ValueError("smoke strata do not match the registered design")
    if Counter(item["stratum"] for item in development) != Counter(
        {"low": 5, "medium": 5, "high": 5, "macro": 5}
    ):
        raise ValueError("development strata do not match the registered design")
    cells = {(item["time_bin"], item["stratum"]) for item in development}
    if len(cells) != 20:
        raise ValueError("development cells are not one session per bin and stratum")


def prepare_four_arm_study(
    *,
    output_path: Path | None = None,
    audit_path: Path | None = None,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    output_path = (output_path or ROOT / "configs/four_arm_llm_study.yaml").resolve()
    audit_path = (audit_path or ROOT / "configs/four_arm_llm_study_audit.json").resolve()
    if not overwrite and (output_path.exists() or audit_path.exists()):
        raise FileExistsError(
            "four-arm study plan already exists; refusing to replace a frozen selection"
        )

    splits_path = ROOT / "data/research100_v1/splits.json"
    universe_path = ROOT / "data/research100_v1/universe/symbols.json"
    cache_path = ROOT / "runs/research100_training/feature_cache.parquet"
    report_path = ROOT / "runs/research100_training/train_cv_report.json"
    company_path = ROOT / "data/2023-11-01_2026-10-30/company/evidence.jsonl"
    macro_path = ROOT / "data/2023-11-01_2026-10-30/macro/events.jsonl"
    inputs = {
        "splits": splits_path,
        "universe": universe_path,
        "feature_cache": cache_path,
        "market_training_report": report_path,
        "company_evidence": company_path,
        "macro_events": macro_path,
    }
    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
        if path.name == "labels.parquet":
            raise RuntimeError("label file is not an allowed selection input")

    splits = json.loads(splits_path.read_text())
    train_sessions = list(splits["train_sessions"])
    validation_sessions = list(splits["validation_sessions"])
    if len(train_sessions) != 676 or len(validation_sessions) != 76:
        raise ValueError("expected the registered 676/76 chronological split")
    universe = {item["symbol"] for item in json.loads(universe_path.read_text())["securities"]}
    if len(universe) != 100:
        raise ValueError("research universe must contain 100 symbols")
    report = json.loads(report_path.read_text())
    weights = np.asarray(
        [report["winner_weights"][field] for field in MARKET_FEATURE_FIELDS],
        dtype=float,
    )
    weights = weights / weights.sum()

    calendar_sessions = _calendar_sessions()
    candidates = _market_candidates(cache_path, train_sessions, universe, weights)
    evidence = _company_evidence(company_path, universe, calendar_sessions)
    macro = _scheduled_macro(macro_path, calendar_sessions)
    profiles = {
        session: _profile(session, candidates[session], evidence, macro, calendar_sessions)
        for session in train_sessions
    }
    selected = select_study_dates(train_sessions, profiles)
    smoke = list(selected["smoke"])
    development = list(selected["development"])
    validate_selection(train_sessions, validation_sessions, smoke, development)

    bin_by_session = selected["bin_by_session"]
    stratum_by_session = selected["stratum_by_session"]
    population = Counter(
        (bin_by_session[session], stratum_by_session[session]) for session in train_sessions
    )
    plan = {
        "version": 1,
        "study": "four_arm_llm_comparison",
        "seed": 329,
        "selection_rng": "none",
        "labels_consulted": False,
        "target": "(high-low)/low",
        "decision_cutoff": "09:00 America/New_York",
        "event_lookback_sessions": EVENT_LOOKBACK_SESSIONS,
        "market_candidate_count": CANDIDATE_COUNT,
        "scheduled_macro_types": sorted(SCHEDULED_MACRO_TYPES),
        "daily_series_excluded_from_macro_stratum": ["dgs2", "dgs10", "vixcls"],
        "arms": {
            "market_only": {
                "uses_llm": False,
                "description": "Frozen 20-feature market ranker.",
            },
            "market_plus_llm": {
                "uses_llm": True,
                "description": (
                    "Same frozen 20-feature market ranker. LLM research is recorded "
                    "for process audit and leaves the order unchanged while "
                    "residual_model.enabled is false."
                ),
            },
            "adaptive_baseline": {
                "uses_llm": False,
                "description": "Historical-range baseline without a model call.",
            },
            "adaptive_plus_llm": {
                "uses_llm": True,
                "description": "Historical-range baseline revised by adaptive LLM research.",
            },
        },
        "paired_comparisons": [
            "market_plus_llm_minus_market_only",
            "adaptive_plus_llm_minus_adaptive_baseline",
        ],
        "primary_metric": "ndcg_at_10",
        "budgets_usd": {"smoke_llm": 10, "development_llm": 40, "evaluation_llm": 100},
        "market_ranker": {
            "source": _relative(report_path),
            "winner": report["winner"],
            "weights": {
                field: float(weight)
                for field, weight in zip(MARKET_FEATURE_FIELDS, weights, strict=True)
            },
        },
        "inputs": {
            name: {"path": _relative(path), "sha256": _file_sha256(path)}
            for name, path in inputs.items()
        },
        "selection_rule": {
            "training_sessions": 676,
            "temporal_bins": TEMPORAL_BINS,
            "company_strata": "within-bin tertiles of pre-cutoff top-20 evidence coverage",
            "macro_stratum": (
                "CPI, payrolls, unemployment, or fed funds is scheduled on that session"
            ),
            "within_cell_choice": "chronological median",
            "smoke_by_bin": list(SMOKE_BY_BIN),
            "development": "one chronological median from every remaining bin and stratum",
        },
        "time_bins": selected["time_bins"],
        "stratum_population_by_bin": {
            f"{bin_index}:{stratum}": population[(bin_index, stratum)]
            for bin_index in range(TEMPORAL_BINS)
            for stratum in DEVELOPMENT_STRATA
        },
        "smoke_sessions": smoke,
        "development_sessions": sorted(development, key=lambda item: item["session"]),
        "evaluation_sessions": validation_sessions,
        "evaluation_window": {
            "first_session": validation_sessions[0],
            "last_session": validation_sessions[-1],
            "session_count": len(validation_sessions),
            "previously_inspected_market_only": True,
            "use": "frozen retrospective evaluation after development choices are locked",
        },
        "notes": [
            "Smoke and development dates come only from the 676 training sessions.",
            "Selection does not read target-session high, low, labels, or ranking metrics.",
            "The final 76 sessions stay fixed. Market-only outcomes were already inspected.",
            "Daily DGS and VIX observations do not make a session a macro-event day.",
            "The chosen day is the chronological median of its cell, not its best outcome.",
        ],
    }
    audit = {
        "version": 1,
        "labels_consulted": False,
        "sessions": [
            {
                "session": session,
                "time_bin": bin_by_session[session],
                "stratum": stratum_by_session[session],
                **_profile_fields(profiles[session]),
            }
            for session in train_sessions
        ],
    }
    _write_text(output_path, yaml.safe_dump(plan, sort_keys=False))
    _write_text(audit_path, json.dumps(audit, indent=2) + "\n")
    return output_path, audit_path


def _selected_record(item: tuple[int, str, SessionProfile]) -> dict[str, object]:
    bin_index, stratum, profile = item
    return {
        "session": profile.session,
        "time_bin": bin_index,
        "stratum": stratum,
        **_profile_fields(profile),
    }


def _profile_fields(profile: SessionProfile) -> dict[str, object]:
    return {
        "symbols_with_evidence": profile.symbols_with_evidence,
        "important_evidence_count": profile.important_evidence_count,
        "symbols_with_document": profile.symbols_with_document,
        "scheduled_macro_count": profile.scheduled_macro_count,
        "scheduled_macro_types": list(profile.scheduled_macro_types),
        "macro_released_before_cutoff": profile.macro_released_before_cutoff,
        "market_candidate_symbols": list(profile.market_candidate_symbols),
    }


def _profile(
    session: str,
    candidates: list[str],
    evidence: dict[str, list[EvidencePoint]],
    macro: dict[date, list[tuple[str, bool]]],
    calendar_sessions: list[date],
) -> SessionProfile:
    target = date.fromisoformat(session)
    cutoff = datetime.combine(target, DECISION_TIME, tzinfo=NEW_YORK)
    target_position = calendar_sessions.index(target)
    symbols_with_evidence = 0
    important_evidence_count = 0
    symbols_with_document = 0
    for symbol in candidates:
        visible = _visible_evidence(
            evidence.get(symbol, []),
            cutoff,
            target_position,
            calendar_sessions,
        )
        if not visible:
            continue
        symbols_with_evidence += 1
        important_evidence_count += sum(
            bool(item.event_types & IMPORTANT_EVENT_TYPES) for item in visible
        )
        if any(item.has_document for item in visible):
            symbols_with_document += 1
    scheduled = macro.get(target, [])
    return SessionProfile(
        session=session,
        symbols_with_evidence=symbols_with_evidence,
        important_evidence_count=important_evidence_count,
        symbols_with_document=symbols_with_document,
        scheduled_macro_count=len({event_type for event_type, _before in scheduled}),
        scheduled_macro_types=tuple(sorted({event_type for event_type, _before in scheduled})),
        macro_released_before_cutoff=len(
            {event_type for event_type, before in scheduled if before}
        ),
        market_candidate_symbols=tuple(candidates),
    )


def _visible_evidence(
    points: list[EvidencePoint],
    cutoff: datetime,
    target_position: int,
    calendar_sessions: list[date],
) -> list[EvidencePoint]:
    cutoff_index = bisect_right(points, cutoff, key=lambda point: point.available_at)
    visible = []
    for point in points[:cutoff_index]:
        release_position = bisect_right(calendar_sessions, point.release_session) - 1
        if calendar_sessions[release_position] != point.release_session:
            continue
        sessions_ago = target_position - release_position
        if 0 <= sessions_ago <= EVENT_LOOKBACK_SESSIONS:
            visible.append(point)
    return visible


def _market_candidates(
    cache_path: Path,
    train_sessions: list[str],
    universe: set[str],
    weights: np.ndarray,
) -> dict[str, list[str]]:
    requested = {date.fromisoformat(session) for session in train_sessions}
    frame = pl.read_parquet(cache_path).filter(pl.col("target_session").is_in(requested))
    grouped = frame.partition_by("target_session", as_dict=True, maintain_order=True)
    candidates: dict[str, list[str]] = {}
    for raw_session, group in grouped.items():
        session = raw_session[0] if isinstance(raw_session, tuple) else raw_session
        session_text = session.isoformat()
        symbols = group["symbol"].to_list()
        if set(symbols) != universe:
            raise ValueError(f"{session_text} feature cache does not match the 100-stock universe")
        percentile_maps = {}
        for field in MARKET_FEATURE_FIELDS:
            values = dict(zip(symbols, group[field].to_list(), strict=True))
            if field in ABSOLUTE_MARKET_FIELDS:
                values = {
                    symbol: None if value is None else abs(float(value))
                    for symbol, value in values.items()
                }
            percentile_maps[field] = (
                {symbol: 0.5 for symbol in symbols}
                if any(value is None for value in values.values())
                else average_percentiles(values)
            )
        scored = {
            symbol: float(
                np.dot(
                    [percentile_maps[field][symbol] for field in MARKET_FEATURE_FIELDS],
                    weights,
                )
            )
            for symbol in symbols
        }
        candidates[session_text] = sorted(scored, key=lambda symbol: (-scored[symbol], symbol))[
            :CANDIDATE_COUNT
        ]
    missing = [session for session in train_sessions if session not in candidates]
    if missing:
        raise ValueError(f"feature cache misses training session {missing[0]}")
    return candidates


def _company_evidence(
    path: Path,
    universe: set[str],
    calendar_sessions: list[date],
) -> dict[str, list[EvidencePoint]]:
    grouped: dict[str, list[EvidencePoint]] = defaultdict(list)
    session_set = set(calendar_sessions)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            symbol = row.get("symbol")
            if symbol not in universe or row.get("event_time") is None:
                continue
            event_time = _parse_timestamp(row["event_time"])
            local_day = event_time.astimezone(NEW_YORK).date()
            release = (
                local_day
                if local_day in session_set
                else _previous_session(local_day, calendar_sessions)
            )
            if release is None:
                continue
            event_types = set(row.get("event_types") or [])
            if row.get("event_type"):
                event_types.add(row["event_type"])
            grouped[symbol].append(
                EvidencePoint(
                    available_at=_parse_timestamp(row["available_at"]),
                    release_session=release,
                    event_types=frozenset(event_type for event_type in event_types if event_type),
                    has_document=bool(row.get("content_path") and row.get("document_content_hash")),
                )
            )
    for points in grouped.values():
        points.sort(key=lambda point: point.available_at)
    return grouped


def _scheduled_macro(
    path: Path,
    calendar_sessions: list[date],
) -> dict[date, list[tuple[str, bool]]]:
    session_set = set(calendar_sessions)
    grouped: dict[date, list[tuple[str, bool]]] = defaultdict(list)
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            event_type = row.get("event_type")
            if event_type not in SCHEDULED_MACRO_TYPES or row.get("event_time") is None:
                continue
            event_time = _parse_timestamp(row["event_time"])
            local_day = event_time.astimezone(NEW_YORK).date()
            if local_day not in session_set:
                continue
            cutoff = datetime.combine(local_day, DECISION_TIME, tzinfo=NEW_YORK)
            available_at = _parse_timestamp(row["available_at"])
            grouped[local_day].append((event_type, available_at <= cutoff))
    return grouped


def _calendar_sessions() -> list[date]:
    schedule = TradingCalendar("XNYS").schedule_snapshot(date(2023, 1, 1), date(2026, 10, 30))
    return [date.fromisoformat(str(item["session"])) for item in schedule]


def _previous_session(day: date, sessions: list[date]) -> date | None:
    index = bisect_right(sessions, day) - 1
    if index < 0:
        return None
    return sessions[index]


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must be timezone-aware: {value}")
    return parsed


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        path.chmod(0o644)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--audit", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing frozen plan. The default is to refuse.",
    )
    arguments = parser.parse_args()
    output, audit = prepare_four_arm_study(
        output_path=arguments.output,
        audit_path=arguments.audit,
        overwrite=arguments.force,
    )
    plan = yaml.safe_load(output.read_text())
    print(
        json.dumps(
            {
                "plan": str(output),
                "audit": str(audit),
                "smoke_sessions": [item["session"] for item in plan["smoke_sessions"]],
                "development_sessions": [item["session"] for item in plan["development_sessions"]],
            }
        )
    )


if __name__ == "__main__":
    main()
