"""Append-only verified research memory, isolated by experimental arm and day.

Expiry changes retrieval, never historical audit records. A card becomes visible
only after its recorded time, even if the underlying source existed earlier.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from financial_strategy.adaptive.models import InfoCard


class AcceptedCardMemory:
    def __init__(
        self,
        root: str | Path,
        run_id: str,
        arm: str,
        cutoff: datetime,
        ttl_days: int = 7,
    ) -> None:
        if cutoff.tzinfo is None:
            raise ValueError("memory cutoff must be timezone-aware")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", arm) or arm in {".", ".."}:
            raise ValueError("memory arm must be a safe path component")
        if ttl_days <= 0:
            raise ValueError("memory lifetime must be positive")
        self.root = Path(root)
        self.run_id = run_id
        self.arm = arm
        self.cutoff = cutoff
        self.ttl_days = ttl_days
        self.arm_root = self.root / arm
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self.arm_root / self.cutoff.date().isoformat() / "cards.jsonl"

    def append(self, cards: list[InfoCard]) -> list[InfoCard]:
        """Append already accepted cards; reject future/expired or duplicate entries."""
        accepted: list[InfoCard] = []
        with self._lock:
            known = {
                (record["run_id"], record["card"]["card_id"], record["card"].get("version", 1))
                for record in self._records()
            }
            for raw in cards:
                card = InfoCard.model_validate(raw).model_copy(deep=True)
                if card.status != "accepted":
                    raise ValueError("only accepted cards may enter verified memory")
                if card.available_at > self.cutoff:
                    raise ValueError("future evidence cannot enter memory")
                latest_expiry = self.cutoff + timedelta(days=self.ttl_days)
                if card.expires_at is None or card.expires_at > latest_expiry:
                    card.expires_at = latest_expiry
                if card.expires_at <= self.cutoff:
                    raise ValueError("expired cards cannot enter active memory")
                identity = (self.run_id, card.card_id, card.version)
                if identity in known:
                    continue
                known.add(identity)
                accepted.append(card)
            if not accepted:
                return []
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                json.dumps(
                    {
                        "run_id": self.run_id,
                        "recorded_at": self.cutoff.isoformat(),
                        "card": card.model_dump(mode="json"),
                    },
                    sort_keys=True,
                )
                + "\n"
                for card in accepted
            ]
            # Each append uses one write under an instance lock; worker completion
            # does not mutate memory. The runner serializes sorted batch commits.
            fd = os.open(self.path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                data = "".join(lines).encode("utf-8")
                written = 0
                while written < len(data):
                    written += os.write(fd, data[written:])
                os.fsync(fd)
            finally:
                os.close(fd)
        return accepted

    def _records(self) -> list[dict[str, Any]]:
        if not self.arm_root.exists():
            return []
        records = []
        for path in sorted(self.arm_root.glob("????-??-??/cards.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    recorded_at = datetime.fromisoformat(record["recorded_at"])
                    if recorded_at.tzinfo is None:
                        raise ValueError("stored memory timestamps must be timezone-aware")
                    records.append(record)
        return records

    def active(self, as_of: datetime | None = None) -> list[InfoCard]:
        """Return eligible versions without reading labels or future memory entries."""
        as_of = as_of or self.cutoff
        if as_of.tzinfo is None:
            raise ValueError("memory read time must be timezone-aware")
        if as_of > self.cutoff:
            raise ValueError("memory reads cannot exceed the run cutoff")
        active: dict[str, tuple[datetime, InfoCard]] = {}
        for record in self._records():
            recorded_at = datetime.fromisoformat(record["recorded_at"])
            if not as_of - timedelta(days=self.ttl_days) <= recorded_at <= as_of:
                continue
            card = InfoCard.model_validate(record["card"])
            if (
                card.status != "accepted"
                or card.available_at > as_of
                or card.expires_at is None
                or card.expires_at <= as_of
            ):
                continue
            previous = active.get(card.card_id)
            if previous is None or (card.version, recorded_at) > (previous[1].version, previous[0]):
                active[card.card_id] = (recorded_at, card)
        superseded = {old_id for _, card in active.values() for old_id in card.supersedes}
        return [card for card_id, (_, card) in sorted(active.items()) if card_id not in superseded]

    def supersede(self, replacement: InfoCard, previous_ids: list[str]) -> list[InfoCard]:
        """Record a new accepted version without modifying earlier journal entries."""
        replacement = replacement.model_copy(deep=True)
        replacement.supersedes = sorted(set([*replacement.supersedes, *previous_ids]))
        return self.append([replacement])
