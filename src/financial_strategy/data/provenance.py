"""Check frozen sources without rewriting Windows text checkouts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from financial_strategy.errors import DataError


def verify_snapshot_provenance(root: str | Path) -> dict:
    """Accept exact bytes or UTF-8 text differing solely by Git CRLF conversion.

    Acquisition SHA and size must both match after that single transformation.
    Binary artifacts remain byte-exact. Run manifests hash actual on-disk bytes.
    """
    root = Path(root).resolve()
    manifest = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
    checked = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise DataError("SOURCE_HASH_MISMATCH", name)
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        exact = digest == expected["sha256"] and len(payload) == expected["size_bytes"]
        mode = "exact_bytes"
        if not exact and path.suffix.lower() in {".txt", ".json", ".jsonl"}:
            payload.decode("utf-8", errors="strict")
            normalized = payload.replace(b"\r\n", b"\n")
            exact = (
                hashlib.sha256(normalized).hexdigest() == expected["sha256"]
                and len(normalized) == expected["size_bytes"]
            )
            mode = "git_crlf_only"
        if not exact:
            raise DataError("SOURCE_HASH_MISMATCH", name)
        checked.append({"path": name, "verification": mode, "on_disk_sha256": digest})
    return {
        "snapshot_id": manifest["snapshot_id"],
        "files": checked,
        "git_crlf_file_count": sum(row["verification"] == "git_crlf_only" for row in checked),
    }
