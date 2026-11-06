"""Application service that joins deterministic preparation, LangGraph, and audit output."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import polars as pl

from financial_strategy.agents import AgentSuite, scripted_responder
from financial_strategy.audit import RunWriter
from financial_strategy.config import content_hash, load_mvp_config
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.data.gateway import PointInTimeGateway
from financial_strategy.data.providers import (
    LocalEvidenceProvider,
    LocalMarketDataProvider,
    LocalUniverseProvider,
)
from financial_strategy.features.engine import compute_features
from financial_strategy.llm import make_client
from financial_strategy.ranking.ranker import market_components
from financial_strategy.ranking.screen import coarse_screen
from financial_strategy.task.validator import validate_and_freeze_task
from financial_strategy.workflow import build_screening_graph


def _make_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def run_screening(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    task_overrides: dict | None = None,
) -> tuple[Path, list[dict]]:
    from financial_strategy.adaptive.config import is_adaptive_config
    if is_adaptive_config(config_path):
        from financial_strategy.adaptive.service import run_adaptive
        return run_adaptive(config_path, run_id=run_id, task_overrides=task_overrides)
    config = load_mvp_config(config_path)
    if task_overrides:
        config = config.model_copy(
            update={"task": config.task.model_copy(update=task_overrides)}
        )
    run_id = run_id or _make_run_id()
    writer = RunWriter(Path(config.output["runs_path"]), run_id)
    published = False
    try:
        writer.write_yaml("config.yaml", config)
        calendar = TradingCalendar(config.calendar["name"])
        universe_provider = LocalUniverseProvider(
            config.data.universe, config.data.sector_exposures
        )
        universe = universe_provider.universe()
        task, exposures, warnings = validate_and_freeze_task(
            config,
            calendar,
            universe,
            universe_provider.sector_exposures(),
        )
        writer.write_json("task.json", task)

        gateway = PointInTimeGateway(
            LocalMarketDataProvider(
                config.data.market_bars,
                config.data.corporate_actions,
            ),
            LocalEvidenceProvider(
                config.data.company_evidence,
                config.data.macro_events,
            ),
            calendar,
            config.data.input_adjustment_policy,
        )
        snapshot = gateway.freeze_inputs(task, exposures)
        expected_sessions = calendar.completed_sessions_before(
            task.decision_cutoff, config.features["required_complete_sessions"]
        )
        features, rejected = compute_features(
            snapshot.bars,
            expected_sessions,
            int(config.features["required_complete_sessions"]),
        )
        volatility, anomaly = market_components(features, config.ranking)
        candidates, coarse_scores = coarse_screen(
            volatility,
            anomaly,
            task.top_k,
            config.screening,
        )

        # Freeze deterministic inputs before the graph starts. If an Agent fails, the
        # failed run still preserves every input that had already been materialized.
        writer.write_json("snapshot/universe.json", universe)
        sector_templates = universe_provider.sector_exposures()
        writer.write_json("snapshot/sector_exposures.json", sector_templates)
        writer.write_parquet("snapshot/market.parquet", snapshot.bars)
        writer.write_jsonl("snapshot/company_evidence.jsonl", list(snapshot.company_evidence))
        writer.write_jsonl("snapshot/macro_evidence.jsonl", list(snapshot.macro_evidence))
        writer.write_jsonl("snapshot/corporate_actions.jsonl", list(snapshot.corporate_actions))
        writer.write_json("snapshot/exposures.json", list(snapshot.exposures))
        writer.write_json("snapshot/calendar.json", list(snapshot.calendar_schedule))
        snapshot_metadata = {
            "snapshot_id": snapshot.snapshot_id,
            "calendar_name": config.calendar["name"],
            "calendar_version": snapshot.calendar_version,
            "input_adjustment_policy": config.data.input_adjustment_policy,
            "snapshot_adjustment_policy": config.data.snapshot_adjustment_policy,
            "adjusted_through": task.decision_cutoff,
            "universe_hash": content_hash(universe),
            "sector_templates_hash": content_hash(sector_templates),
            "corporate_actions_hash": content_hash(list(snapshot.corporate_actions)),
            "calendar_schedule_hash": content_hash(list(snapshot.calendar_schedule)),
        }
        writer.write_json("snapshot/metadata.json", snapshot_metadata)
        writer.write_json(
            "snapshot/manifest.json",
            {
                "snapshot_id": snapshot.snapshot_id,
                "task_id": task.task_id,
                "content_hash": content_hash(
                    {
                        "metadata": snapshot_metadata,
                        "company_evidence": list(snapshot.company_evidence),
                        "macro_evidence": list(snapshot.macro_evidence),
                        "exposures": list(snapshot.exposures),
                    }
                ),
            },
        )
        feature_frame = pl.DataFrame(
            [feature.model_dump(mode="json") for feature in features.values()]
        )
        writer.write_parquet("features.parquet", feature_frame)
        writer.write_json(
            "screening.json",
            {
                "candidates": candidates,
                "coarse_scores": coarse_scores,
                "rejected": rejected,
                "warnings": warnings,
            },
        )

        client = make_client(config.agents, scripted_responder, config.retry)
        suite = AgentSuite(config=config, client=client)
        graph = build_screening_graph(config, suite)
        result = graph.invoke(
            {
                "task": task,
                "snapshot": snapshot,
                "features": features,
                "candidates": candidates,
                "volatility": volatility,
                "anomaly": anomaly,
                "cards": [],
                "initial_cards": [],
                "attempts": [],
                "retrieval_traces": [],
            },
            {"max_concurrency": config.agents.max_parallel_symbols},
        )

        writer.write_json("macro_context.json", result["macro_context"])
        writer.write_json("initial_research_cards.json", result["initial_cards"])
        writer.write_json("research_cards.json", result["cards"])
        writer.write_json("verification_reports.json", [
            {
                "symbol": card.symbol,
                "deterministic": card.deterministic_verification,
                "semantic": card.semantic_verification,
            }
            for card in result["cards"]
        ])
        writer.write_json("watchlist.json", result["watchlist"])
        writer.write_jsonl("retrieval_traces.jsonl", result["retrieval_traces"])

        attempts = result["attempts"]
        writer.write_jsonl(
            "model_attempts.jsonl",
            getattr(client, "attempt_log", []),
        )
        for attempt in attempts:
            writer.write_json(
                f"llm_responses/{attempt.request_hash}.json",
                {
                    "request_hash": attempt.request_hash,
                    "role": attempt.role,
                    "mode": config.agents.mode,
                    "provider": config.agents.provider if config.agents.mode == "live" else "local",
                    "model": attempt.model,
                    "prompt_version": attempt.prompt_version,
                    "schema_name": attempt.schema_name,
                    "response_source": attempt.source,
                    "attempt_type": attempt.attempt_type,
                    "response": attempt.raw,
                    "validation_status": "valid",
                    "response_content_hash": content_hash(attempt.raw),
                },
            )
        path = writer.publish("completed")
        published = True
        return path, [item.model_dump(mode="json") for item in result["watchlist"]]
    except Exception as exc:
        if not published:
            path = writer.publish(
                "failed",
                {
                    "type": type(exc).__name__,
                    "code": getattr(exc, "code", "UNEXPECTED_ERROR"),
                    "message": str(exc),
                },
            )
            published = True
        raise RuntimeError(f"Screening failed; audit saved at {path}: {exc}") from exc
