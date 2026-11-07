from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from scripts.build_expanded_data import load_config, select_universe
from scripts.enrich_evidence import enrich_snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_expansion_config_freezes_large_research_universe() -> None:
    config = load_config(ROOT / "configs/data_expansion.yaml")
    assert config["target_universe_size"] == 200
    assert config["top_k"] == 10
    assert config["deep_research_candidates"] == 50
    assert config["reproducibility"]["allow_overwrite"] is True
    assert config["snapshot_id"] == "2023-11-01_2026-10-30"
    assert config["output_path"] == "data/2023-11-01_2026-10-30"
    assert config["company_source"]["name"] == "sec_edgar_full_documents"


def test_universe_selection_uses_warmup_liquidity_and_records_exclusions() -> None:
    config = load_config(ROOT / "configs/data_expansion.yaml")
    config = {
        **config,
        "target_universe_size": 200,
        "minimum_session_completeness": 0.98,
    }
    sessions = [date(2023, 7, 3) + timedelta(days=index) for index in range(60)]
    securities = [
        {
            "symbol": f"S{index:03}",
            "company_name": f"Security {index}",
            "cik": str(index).zfill(10),
            "permanent_entity_id": f"sec_cik_{index:010}",
            "sector": "technology",
        }
        for index in range(205)
    ]
    frames = {}
    for index, security in enumerate(securities):
        rows = [
            {
                "session": session,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": float(1_000 + index),
                "stock_splits": 0.0,
            }
            for session in sessions
        ]
        frames[security["symbol"]] = pl.DataFrame(rows)
    selected, excluded = select_universe(config, securities, frames, sessions)
    assert len(selected) == 200
    assert selected[0]["symbol"] == "S204"
    assert selected[-1]["symbol"] == "S005"
    assert {item["symbol"] for item in excluded} == {
        "S000",
        "S001",
        "S002",
        "S003",
        "S004",
    }
    assert {item["reason"] for item in excluded} == {"BELOW_LIQUIDITY_CUTOFF"}


def test_evidence_enrichment_can_replace_its_source_in_place(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    for directory in ("company", "macro", "market", "universe"):
        (snapshot / directory).mkdir(parents=True, exist_ok=True)
    (snapshot / "company/evidence.jsonl").write_text("")
    (snapshot / "macro/events.jsonl").write_text("")
    (snapshot / "universe/symbols.json").write_text(
        json.dumps({"snapshot_id": "base", "securities": []})
    )
    (snapshot / "splits.json").write_text(
        json.dumps({"snapshot_id": "base", "content_hash": "old"})
    )
    (snapshot / "manifest.json").write_text(
        json.dumps({"file_sha256": {}, "source_limitations": []})
    )
    pl.DataFrame({"snapshot_id": ["base"], "close": [100.0]}).write_parquet(
        snapshot / "market/daily_bars.parquet"
    )
    candidate_cards = tmp_path / "candidate_cards.json"
    candidate_cards.write_text("[]")

    result = enrich_snapshot(
        source=snapshot,
        output=snapshot,
        candidate_cards=candidate_cards,
        macro_source=None,
        maximum_documents=0,
    )

    assert result == snapshot
    assert snapshot.is_dir()
    assert json.loads((snapshot / "manifest.json").read_text())["snapshot_id"] == (
        "snapshot"
    )
    assert pl.read_parquet(snapshot / "market/daily_bars.parquet")[
        "snapshot_id"
    ].to_list() == ["snapshot"]


def test_expanded_snapshot_has_complete_200_symbol_panel_and_valid_hashes() -> None:
    dataset = ROOT / "data/2023-11-01_2026-10-30"
    universe = json.loads((dataset / "universe/symbols.json").read_text())
    quality = json.loads((dataset / "quality_report.json").read_text())
    manifest = json.loads((dataset / "manifest.json").read_text())
    assert len(universe["securities"]) == 200
    assert quality["evaluation_session_count"] == 752
    assert quality["evaluation_bar_count"] == 150_400
    assert quality["evaluation_completeness"] == 1.0
    assert quality["duplicate_bar_keys"] == 0
    assert quality["zero_volume_rows"] == 0
    for relative, expected in manifest["file_sha256"].items():
        path = dataset / relative
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected and path.suffix.lower() in {".txt", ".json", ".jsonl"}:
            # Git may expand acquisition LF text to CRLF on Windows.
            payload.decode("utf-8", errors="strict")
            actual = hashlib.sha256(payload.replace(b"\r\n", b"\n")).hexdigest()
        assert actual == expected


def test_expanded_split_is_chronological_and_labels_are_complete() -> None:
    dataset = ROOT / "data/2023-11-01_2026-10-30"
    split = json.loads((dataset / "splits.json").read_text())
    train = [date.fromisoformat(value) for value in split["train_sessions"]]
    validation = [date.fromisoformat(value) for value in split["validation_sessions"]]
    labels = pl.read_parquet(dataset / "evaluation/labels.parquet")
    assert len(train) == 676
    assert len(validation) == 76
    assert max(train) < min(validation)
    assert set(train).isdisjoint(validation)
    assert labels.height == 752 * 200
    assert labels.group_by("session").len()["len"].min() == 200

