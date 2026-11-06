from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest
import yaml

from financial_strategy.config import TuningConfig, load_mvp_config
from financial_strategy.errors import AuditError, ConfigError
from financial_strategy.tuning import prepare_artifacts, run_tuning

ROOT = Path(__file__).resolve().parents[1]


def _write_split(path: Path) -> None:
    payload = {
        "snapshot_id": "test_split",
        "strategy": "chronological_holdout",
        "train_sessions": ["2026-10-01", "2026-10-02"],
        "validation_sessions": ["2026-10-05"],
        "train_fraction": 2 / 3,
        "validation_fraction": 1 / 3,
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _screening_config(tmp_path: Path) -> Path:
    config = load_mvp_config(ROOT / "configs/screening_2026-10-14.yaml")
    data = config.model_dump(mode="json")
    for key in (
        "universe",
        "market_bars",
        "corporate_actions",
        "company_evidence",
        "macro_events",
        "macro_documents",
        "sector_exposures",
    ):
        data["data"][key] = str(ROOT / data["data"][key])
    data["agents"]["mode"] = "scripted"
    data["agents"]["provider"] = "openai"
    data["agents"]["model"] = "scripted"
    data["agents"]["cache_path"] = str(tmp_path / "cache")
    data["screening"]["minimum_candidates"] = 20
    data["output"]["runs_path"] = str(tmp_path / "runs")
    path = tmp_path / "screening.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=True))
    return path


def _tuning_config(
    tmp_path: Path,
    screening: Path,
    labels: Path,
    suffix: str,
) -> TuningConfig:
    return TuningConfig.model_validate(
        {
            "screening_config": screening,
            "splits": tmp_path / "splits.json",
            "labels": labels,
            "runs_path": tmp_path / "runs",
            "artifact_index": tmp_path / "index.json",
            "output_path": tmp_path / f"result_{suffix}.json",
            "tuned_config_path": tmp_path / f"tuned_{suffix}.yaml",
            "large_move_threshold": 0.02,
            "search": {
                "seed": 329,
                "trials": 4,
                "minimum_scores": [0.2, 0.35],
                "candidate_counts": [10, 20],
                "primary_metric": "ndcg_at_k",
                "minimum_session_coverage": 1.0,
                "maximum_failed_run_rate": 0.0,
                "bootstrap_samples": 100,
                "bootstrap_block_sessions": 1,
            },
        }
    )


def test_tuning_is_reproducible_and_validation_is_consumed_once(
    tmp_path: Path,
) -> None:
    _write_split(tmp_path / "splits.json")
    screening = _screening_config(tmp_path)
    missing_labels = tmp_path / "not_created_yet.parquet"
    setup = _tuning_config(tmp_path, screening, missing_labels, "setup")

    # Artifact preparation is prediction-only and must not require labels.
    index = prepare_artifacts(setup)
    assert len(index["sessions"]) == 3
    assert not missing_labels.exists()

    source_labels = pl.read_parquet(
        ROOT / "tests/fixtures/integration/evaluation/labels.parquet"
    ).filter(
        pl.col("session").is_in(
            [date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 5)]
        )
    )
    labels_a = tmp_path / "labels_a.parquet"
    source_labels.write_parquet(labels_a)
    labels_b = tmp_path / "labels_b.parquet"
    source_labels.with_columns(
        pl.when(pl.col("session").cast(pl.String) == "2026-10-05")
        .then(pl.col("open") * 1.5)
        .otherwise(pl.col("close"))
        .alias("close")
    ).write_parquet(labels_b)

    # A valid run cannot be relabeled as another session even when its own
    # manifest remains cryptographically valid.
    bad_index = json.loads(json.dumps(index))
    first_two = sorted(bad_index["sessions"])[:2]
    bad_index["sessions"][first_two[0]], bad_index["sessions"][first_two[1]] = (
        bad_index["sessions"][first_two[1]],
        bad_index["sessions"][first_two[0]],
    )
    bad_index_path = tmp_path / "bad_index.json"
    bad_index_path.write_text(json.dumps(bad_index))
    bad_binding_config = _tuning_config(tmp_path, screening, labels_a, "bad_binding")
    bad_binding_config = bad_binding_config.model_copy(
        update={"artifact_index": bad_index_path}
    )
    with pytest.raises(ConfigError) as binding_error:
        run_tuning(bad_binding_config)
    assert binding_error.value.code == "ARTIFACT_SESSION_MISMATCH"

    duplicate_labels = tmp_path / "duplicate_labels.parquet"
    pl.concat([source_labels, source_labels.head(1)]).write_parquet(duplicate_labels)
    duplicate_config = _tuning_config(
        tmp_path,
        screening,
        duplicate_labels,
        "duplicate",
    )
    with pytest.raises(ConfigError) as duplicate_error:
        run_tuning(duplicate_config)
    assert duplicate_error.value.code == "DUPLICATE_LABEL"

    first_config = _tuning_config(tmp_path, screening, labels_a, "a")
    second_config = _tuning_config(tmp_path, screening, labels_b, "b")
    _first_path, first = run_tuning(first_config)
    candidate = load_mvp_config(first_config.tuned_config_path.with_suffix(".candidate.yaml"))
    assert sum(candidate.ranking.final_weights.model_dump().values()) == pytest.approx(1.0)
    assert sum(candidate.ranking.volatility_weights.model_dump().values()) == pytest.approx(1.0)
    assert sum(candidate.ranking.anomaly_weights.model_dump().values()) == pytest.approx(1.0)
    assert first["validation"]["session_count"] == 1
    assert "original_full_system" in first["validation"]["variants"]
    assert "tuned_full_system" in first["validation"]["variants"]
    assert first_config.tuned_config_path.exists() is first["validation"][
        "accept_tuned_config"
    ]

    with pytest.raises(ConfigError, match="already exists"):
        run_tuning(first_config)
    with pytest.raises(ConfigError) as consumption_error:
        run_tuning(second_config)
    assert consumption_error.value.code == "VALIDATION_ALREADY_CONSUMED"

    # Tuning must reject any artifact changed after its immutable manifest was
    # published, even when every session is still marked completed in the index.
    tampered_session = sorted(index["sessions"])[0]
    tampered_run = index["sessions"][tampered_session]["run_id"]
    cards_path = tmp_path / "runs" / tampered_run / "research_cards.json"
    cards_path.write_text(cards_path.read_text() + "\n")
    tamper_config = _tuning_config(tmp_path, screening, labels_a, "tampered")
    with pytest.raises(AuditError) as exc_info:
        run_tuning(tamper_config)
    assert exc_info.value.code == "ARTIFACT_HASH_MISMATCH"


def test_split_rejects_overlap(tmp_path: Path) -> None:
    payload = {
        "snapshot_id": "bad",
        "strategy": "chronological_holdout",
        "train_sessions": ["2026-10-01"],
        "validation_sessions": ["2026-10-01"],
    }
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    config = _tuning_config(
        tmp_path,
        ROOT / "configs/screening_2026-10-14.yaml",
        ROOT / "tests/fixtures/integration/evaluation/labels.parquet",
        "bad",
    )
    config = config.model_copy(update={"splits": path})
    with pytest.raises(ConfigError, match="disjoint"):
        prepare_artifacts(config)
