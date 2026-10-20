"""Exchange session queries behind a small, testable boundary."""

from __future__ import annotations

from datetime import date, datetime

import exchange_calendars as xcals
import pandas as pd

from financial_strategy.errors import DataError


class TradingCalendar:
    """Use XNYS as the common US regular-session calendar for the MVP universe."""

    def __init__(self, name: str = "XNYS") -> None:
        if name != "XNYS":
            raise DataError("UNSUPPORTED_CALENDAR", f"MVP supports XNYS, not {name}")
        self.name = name
        self._calendar = xcals.get_calendar(name)

    def is_session(self, session: date) -> bool:
        return self._calendar.is_session(pd.Timestamp(session))

    def session_open(self, session: date) -> datetime:
        self._require_session(session)
        return self._calendar.session_open(pd.Timestamp(session)).to_pydatetime()

    def session_close(self, session: date) -> datetime:
        self._require_session(session)
        return self._calendar.session_close(pd.Timestamp(session)).to_pydatetime()

    def previous_session(self, session: date) -> date:
        self._require_session(session)
        return self._calendar.previous_session(pd.Timestamp(session)).date()

    def completed_sessions_before(self, cutoff: datetime, count: int) -> list[date]:
        """Return the newest `count` sessions whose close is strictly before cutoff."""

        if cutoff.tzinfo is None:
            raise DataError("NAIVE_CUTOFF", "cutoff must be timezone-aware")
        minute = pd.Timestamp(cutoff).tz_convert("UTC")
        session = self._calendar.minute_to_past_session(minute, count=1)
        # A cutoff during or before a session must not make that incomplete session available.
        if self._calendar.session_close(session) >= minute:
            session = self._calendar.previous_session(session)
        # exchange_calendars interprets a negative window as the exact number of sessions.
        sessions = self._calendar.sessions_window(session, -count)
        return [item.date() for item in sessions]

    def schedule_snapshot(self, start: date, end: date) -> list[dict[str, str | bool]]:
        sessions = self._calendar.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        result: list[dict[str, str | bool]] = []
        for session in sessions:
            opened = self._calendar.session_open(session)
            closed = self._calendar.session_close(session)
            result.append(
                {
                    "session": str(session.date()),
                    "market_open": opened.isoformat(),
                    "market_close": closed.isoformat(),
                    "is_half_day": closed - opened < pd.Timedelta(hours=6),
                }
            )
        return result

    def _require_session(self, session: date) -> None:
        if not self.is_session(session):
            raise DataError("INVALID_TARGET_SESSION", f"{session} is not an XNYS session")
