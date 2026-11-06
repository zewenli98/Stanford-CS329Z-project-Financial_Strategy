import hashlib
import json

import pytest

from financial_strategy.data.provenance import verify_snapshot_provenance
from financial_strategy.errors import DataError


def _manifest(root, name, expected):
    (root / "provenance.json").write_text(
        json.dumps(
            {
                "snapshot_id": "fixture",
                "files": {
                    name: {
                        "sha256": hashlib.sha256(expected).hexdigest(),
                        "size_bytes": len(expected),
                    }
                },
            }
        )
    )


def test_git_line_endings_are_recognized_without_modifying_sources(tmp_path):
    original = b"first\nsecond\n"
    checkout = original.replace(b"\n", b"\r\n")
    source = tmp_path / "source.txt"
    source.write_bytes(checkout)
    _manifest(tmp_path, source.name, original)
    report = verify_snapshot_provenance(tmp_path)
    assert report["git_crlf_file_count"] == 1
    assert source.read_bytes() == checkout
    source.write_bytes(checkout.replace(b"second", b"tampered"))
    with pytest.raises(DataError, match="source.txt"):
        verify_snapshot_provenance(tmp_path)


def test_binary_hashes_are_strict_even_for_crlf_changes(tmp_path):
    (tmp_path / "data.parquet").write_bytes(b"a\r\nb")
    _manifest(tmp_path, "data.parquet", b"a\nb")
    with pytest.raises(DataError):
        verify_snapshot_provenance(tmp_path)


def test_windows_cache_fallback_preserves_first_writer(tmp_path, monkeypatch):
    import os

    from financial_strategy.llm.cache import AtomicJsonCache

    if os.name != "nt":
        pytest.skip("Windows-only non-overwriting rename semantics")

    def no_links(*args):
        raise PermissionError("sandbox disallows links")

    monkeypatch.setattr("financial_strategy.llm.cache.os.link", no_links)
    cache = AtomicJsonCache(tmp_path / "cache")
    cache.write("key", {"response": {"value": 1}})
    cache.write("key", {"response": {"value": 2}})
    assert cache.read("key") == {"response": {"value": 1}}
