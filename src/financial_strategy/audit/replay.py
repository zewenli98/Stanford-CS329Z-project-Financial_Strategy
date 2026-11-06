"""Deterministically rebuild ranking from one immutable successful run."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from financial_strategy.agents import AgentSuite, scripted_responder
from financial_strategy.audit.writer import verify_manifest
from financial_strategy.config import load_mvp_config
from financial_strategy.data.gateway import PointInTimeSnapshot
from financial_strategy.errors import AuditError
from financial_strategy.llm import make_client
from financial_strategy.ranking.ranker import market_components
from financial_strategy.schemas import (
    CorporateActionRecord,
    EvidenceRecord,
    FeatureRecord,
    MacroEvidenceRecord,
    ScreeningTask,
    SymbolMacroExposure,
)
from financial_strategy.workflow import build_screening_graph


def replay_run(run_dir: Path) -> list[dict]:
    from financial_strategy.adaptive.config import is_adaptive_config
    if is_adaptive_config(run_dir / "config.yaml"):
        from financial_strategy.adaptive.replay import replay_adaptive
        return replay_adaptive(run_dir)
    manifest = verify_manifest(run_dir)
    if manifest["status"] != "completed":
        raise AuditError("RUN_NOT_REPLAYABLE", str(run_dir))
    config = load_mvp_config(run_dir / "config.yaml")
    task = ScreeningTask.model_validate(json.loads((run_dir / "task.json").read_text()))
    features_frame = pl.read_parquet(run_dir / "features.parquet")
    features = {
        row["symbol"]: FeatureRecord.model_validate(row)
        for row in features_frame.to_dicts()
    }
    volatility, anomaly = market_components(features, config.ranking)
    metadata = json.loads((run_dir / "snapshot/metadata.json").read_text())
    exposure_values = json.loads((run_dir / "snapshot/exposures.json").read_text())
    snapshot = PointInTimeSnapshot(
        snapshot_id=metadata["snapshot_id"],
        bars=pl.read_parquet(run_dir / "snapshot/market.parquet"),
        company_evidence=tuple(
            EvidenceRecord.model_validate(item)
            for item in _read_jsonl(run_dir / "snapshot/company_evidence.jsonl")
        ),
        macro_evidence=tuple(
            MacroEvidenceRecord.model_validate(item)
            for item in _read_jsonl(run_dir / "snapshot/macro_evidence.jsonl")
        ),
        corporate_actions=tuple(
            CorporateActionRecord.model_validate(item)
            for item in _read_jsonl(run_dir / "snapshot/corporate_actions.jsonl")
        ),
        exposures=tuple(SymbolMacroExposure.model_validate(item) for item in exposure_values),
        calendar_schedule=tuple(
            json.loads((run_dir / "snapshot/calendar.json").read_text())
        ),
        calendar_version=metadata["calendar_version"],
    )
    screening = json.loads((run_dir / "screening.json").read_text())
    replay_agents = config.agents.model_copy(
        update={
            "mode": "recorded",
            "recordings_path": run_dir / "llm_responses",
        }
    )
    replay_config = config.model_copy(update={"agents": replay_agents})
    client = make_client(replay_agents, scripted_responder, replay_config.retry)
    graph = build_screening_graph(replay_config, AgentSuite(replay_config, client))
    result = graph.invoke(
        {
            "task": task,
            "snapshot": snapshot,
            "features": features,
            "candidates": screening["candidates"],
            "volatility": volatility,
            "anomaly": anomaly,
            "cards": [],
            "initial_cards": [],
            "attempts": [],
            "retrieval_traces": [],
        },
        {"max_concurrency": replay_agents.max_parallel_symbols},
    )
    rebuilt = result["watchlist"]
    stored = json.loads((run_dir / "watchlist.json").read_text())
    rebuilt_json = [item.model_dump(mode="json") for item in rebuilt]
    if rebuilt_json != stored:
        raise AuditError("REPLAY_MISMATCH", str(run_dir))
    return rebuilt_json


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
