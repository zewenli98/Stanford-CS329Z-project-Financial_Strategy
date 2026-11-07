"""Atomic request-hash cache used only by live LLM mode."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


class AtomicJsonCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.tmp = root / ".tmp"

    def read(self, key: str) -> dict | None:
        path = self.root / f"{key}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            path.unlink(missing_ok=True)
            return None

    def write(self, key: str, value: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.tmp.mkdir(parents=True, exist_ok=True)
        final = self.root / f"{key}.json"
        if final.exists():
            return
        fd, temporary_name = tempfile.mkstemp(prefix=f"{key}.", dir=self.tmp)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(value, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            # Hard-link creation is atomic and never overwrites a concurrent winner.
            try:
                os.link(temporary, final)
            except FileExistsError:
                pass
            except PermissionError:
                if os.name != "nt":
                    raise
                # Windows rename is atomic and refuses to replace an existing
                # destination, unlike POSIX rename. Restricted Windows sessions
                # may disallow hard links even inside the workspace.
                try:
                    os.rename(temporary, final)
                except FileExistsError:
                    pass
        finally:
            temporary.unlink(missing_ok=True)
