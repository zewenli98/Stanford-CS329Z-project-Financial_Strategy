"""Generate label-blind live/recorded LLM v2 cards for event-bearing candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from dotenv import load_dotenv

from financial_strategy.agents.roles import (
    CompanyAgentOutput,
    MacroAgentOutput,
    SynthesisOutput,
    align_synthesis_output,
    normalize_company_output,
    normalize_macro_assessments,
    normalize_semantic_verification,
    scripted_responder,
    semantic_evidence_payloads,
    semantic_scope,
)
from financial_strategy.config import load_mvp_config
from financial_strategy.data.providers import LocalEvidenceProvider
from financial_strategy.llm import make_client
from financial_strategy.llm.cache import AtomicJsonCache
from financial_strategy.llm.client import request_hash
from financial_strategy.macro import build_macro_interactions
from financial_strategy.ranking.ranker import llm_residual_features
from financial_strategy.retrieval.bm25 import search
from financial_strategy.schemas import (
    Eligibility,
    FeatureRecord,
    MacroContext,
    Prediction,
    ResearchCard,
    ScreeningTask,
    SymbolMacroExposure,
    Verdict,
    VerificationIssue,
    VerificationReport,
)
from financial_strategy.verification.deterministic import verify_card

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)


def generate_artifacts(
    *,
    mode: str,
    maximum_cards: int,
    maximum_model_calls: int,
    output_directory: Path | None = None,
    maximum_sessions: int | None = None,
    maximum_input_tokens: int = 4_000_000,
    maximum_output_tokens: int = 500_000,
    maximum_cost_usd: float = 100.0,
    parallel_workers: int = 1,
    base_config_path: Path | None = None,
    candidate_cards_path: Path | None = None,
    feature_cache_path: Path | None = None,
    reuse_directory: Path | None = None,
) -> tuple[Path, dict]:
    base_config_path = (
        base_config_path or ROOT / "configs/screening_expanded_candidate.yaml"
    ).resolve()
    candidate_cards_path = (
        candidate_cards_path
        or ROOT / "runs/expanded_agent/published_research_cards.json"
    ).resolve()
    feature_cache_path = (
        feature_cache_path
        or ROOT / "runs/expanded_training/feature_cache.parquet"
    ).resolve()
    base = load_mvp_config(base_config_path)
    retrieval = base.retrieval.model_copy(update={"top_n": 4})
    dataset = ROOT / "data/2023-11-01_2026-10-30"
    output = output_directory or ROOT / "runs/llm_residual_session40"
    output.mkdir(parents=True, exist_ok=True)
    if reuse_directory is not None:
        for name in ("recordings", "cache"):
            source = reuse_directory.resolve() / name
            if source.exists():
                shutil.copytree(source, output / name, dirs_exist_ok=True)
    artifact_path = output / f"artifacts_{mode}.json"
    manifest_path = output / f"manifest_{mode}.json"
    roles = {
        name: role.model_copy(
            update={
                "prompt_version": (
                    "synthesis_v5" if name == "synthesis" else f"{name}_v4"
                ),
                # The v2 company schema includes quote-level citations and can
                # legitimately exceed the old 2K completion cap. The aggregate
                # output-token and dollar budgets below remain the hard limits.
                "max_tokens": 8_192 if name == "company_evidence" else 4_096,
            }
        )
        for name, role in base.agents.roles.items()
    }
    agents = base.agents.model_copy(
        update={
            "mode": mode,
            "model": "openai/gpt-5.6-luna" if mode != "scripted" else "scripted",
            "roles": roles,
            "recordings_path": output / "recordings",
            "cache_path": output / "cache",
            "max_total_model_calls": maximum_model_calls,
            # This batch spans many historical tasks, so a ticker may appear
            # in several sessions. Per-symbol limits belong to one screening
            # task; the aggregate batch is governed by the total-call budget.
            "max_model_calls_per_symbol": maximum_model_calls,
            # Session-grouped experiments can reference many distinct filings;
            # every batch still has explicit call, token, and dollar caps.
            "max_total_input_tokens": maximum_input_tokens,
            "max_total_output_tokens": maximum_output_tokens,
            "max_cost_usd": maximum_cost_usd,
            "input_cost_per_million": 0.2,
            "output_cost_per_million": 1.2,
        }
    )
    client = make_client(agents, scripted_responder, base.retry)
    failure_recordings = AtomicJsonCache(agents.recordings_path)

    def failure_record(
        role: str,
        payload: dict,
        schema: type,
        error: Exception,
    ) -> dict:
        role_config = agents.roles[role]
        record = {
            "role": role,
            "request_hash": request_hash(
                role,
                payload,
                schema,
                agents.model_for(role),
                role_config.prompt_version,
                "initial",
                role_config.max_tokens,
            ),
            "source": "failed",
            "model": agents.model_for(role),
            "prompt_version": role_config.prompt_version,
            "schema_name": schema.__name__,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_seconds": 0.0,
            "error_code": type(error).__name__,
        }
        failure_recordings.write(
            record["request_hash"],
            {
                "failure": {
                    "error_code": record["error_code"],
                    "fail_closed": True,
                },
                "model": record["model"],
                "prompt_version": record["prompt_version"],
                "schema_name": record["schema_name"],
            },
        )
        return record

    provider = LocalEvidenceProvider(
        dataset / "company/evidence.jsonl",
        dataset / "macro/events.jsonl",
    )
    evidence_metadata = [
        json.loads(line)
        for line in (dataset / "company/evidence.jsonl").read_text().splitlines()
        if line
    ]
    metadata_by_id = {
        item["evidence_id"]: item
        for item in evidence_metadata
    }
    all_ids = set(metadata_by_id)
    related_ids = {
        evidence_id: sorted(
            candidate_id
            for candidate_id in all_ids
            if candidate_id == evidence_id
            or candidate_id.startswith(f"{evidence_id}_ex99_")
        )
        for evidence_id in all_ids
    }
    candidate_cards = json.loads(
        candidate_cards_path.read_text()
    )
    eligible_cards = [
        item
        for item in candidate_cards
        if any(
            metadata_by_id.get(evidence_id, {}).get("content_path") is not None
            and metadata_by_id.get(evidence_id, {}).get("document_content_hash")
            is not None
            for evidence_id in item["evidence_ids"]
        )
    ]
    selected = (
        _select_grouped_sessions(eligible_cards, maximum_sessions)
        if maximum_sessions is not None
        else _evenly_spaced(eligible_cards, maximum_cards)
    )
    selected_sessions = sorted({item["target_session"] for item in selected})
    desired_ids = {
        related_id
        for item in selected
        for evidence_id in item["evidence_ids"]
        for related_id in related_ids.get(evidence_id, [evidence_id])
    }
    evidence = provider.company_evidence_by_ids(desired_ids)
    macro_evidence = provider.macro_evidence()
    by_id = {item.evidence_id: item for item in evidence}
    feature_frame = pl.read_parquet(feature_cache_path)
    feature_rows = {
        (row["target_session"].isoformat(), row["symbol"]): row
        for row in feature_frame.to_dicts()
    }

    macro_cache: dict[str, list] = {}
    macro_assessment_cache: dict[str, list] = {}
    invocation_records = []
    tasks_by_session: dict[str, ScreeningTask] = {}
    for session_text in selected_sessions:
        session = date.fromisoformat(session_text)
        cutoff = datetime.combine(
            session,
            time(9, 0),
            ZoneInfo("America/New_York"),
        )
        task = ScreeningTask(
            task_id=f"llm_v2_{session.isoformat()}",
            universe_snapshot_id=dataset.name,
            decision_cutoff=cutoff,
            target_session=session,
            top_k=10,
            price_lookback_sessions=60,
            news_lookback_days=7,
            max_revision_rounds=0,
        )
        tasks_by_session[session_text] = task
        relevant_macro = [
            event
            for event in macro_evidence
            if event.available_at <= cutoff
            and (
                event.event_time is not None
                and event.event_time.date() == session
                or event.available_at >= cutoff - timedelta(days=1)
            )
        ]
        if not relevant_macro:
            macro_cache[session_text] = []
            macro_assessment_cache[session_text] = []
            continue
        hits = search(
            relevant_macro,
            "inflation employment interest rates growth oil dollar",
            retrieval,
            scope="macro",
            target_session=session,
        )
        macro_payload = {
            "task_id": task.task_id,
            "decision_cutoff": cutoff,
            "evidence": [
                {
                    **hit.evidence.model_dump(mode="json"),
                    "retrieved_chunk": {
                        "chunk_id": hit.chunk_id,
                        "text": hit.text,
                    },
                }
                for hit in hits
            ],
        }
        macro_invocation = client.generate(
            "macro_evidence",
            macro_payload,
            MacroAgentOutput,
        )
        invocation_records.append(_invocation_record(macro_invocation))
        macro_output = MacroAgentOutput.model_validate(macro_invocation.response)
        selected_macro_ids = set(macro_output.evidence_ids)
        macro_cache[session_text] = [
            event
            for event in relevant_macro
            if event.evidence_id in selected_macro_ids
        ]
        macro_assessment_cache[session_text] = normalize_macro_assessments(
            macro_output.assessments,
            macro_cache[session_text],
            task,
            base.ranking.macro.surprise_scales,
        )

    macro_interactions = _build_macro_interactions(
        dataset=dataset,
        sessions=selected_sessions,
        target_events=macro_cache,
        history_events=macro_evidence,
        surprise_scales=base.ranking.macro.surprise_scales,
    )

    card_contexts = []
    extraction_jobs: dict[str, tuple] = {}
    for item in selected:
        task = tasks_by_session[item["target_session"]]
        expanded_evidence_ids = sorted(
            {
                related_id
                for evidence_id in item["evidence_ids"]
                for related_id in related_ids.get(evidence_id, [evidence_id])
            }
        )
        card_contexts.append((item, task, expanded_evidence_ids))
        for evidence_id in expanded_evidence_ids:
            source = by_id.get(evidence_id)
            if source is None or source.document_content_hash is None:
                continue
            document_key = source.document_content_hash or source.content_hash
            extraction_jobs.setdefault(
                document_key,
                (source, item["symbol"], task),
            )

    def extract_document(job: tuple[str, tuple]) -> tuple:
        document_key, (source, symbol, task) = job
        hits = search(
            [source],
            "reported revenue earnings guidance consensus outlook material change risk",
            retrieval,
            scope="company",
            symbol=symbol,
            target_session=task.target_session,
        )
        payload = {
            "symbol": symbol,
            "evidence": [
                {
                    **hit.evidence.model_dump(mode="json"),
                    "retrieved_chunk": {
                        "chunk_id": hit.chunk_id,
                        "text": hit.text,
                    },
                }
                for hit in hits
            ],
            "revision_feedback": [],
        }
        try:
            invocation = client.generate(
                "company_evidence",
                payload,
                CompanyAgentOutput,
            )
        except ConnectionError as error:
            # One malformed provider response must not discard an otherwise
            # immutable multi-session batch. Fail this document closed and
            # preserve the error in the invocation audit.
            return (
                document_key,
                CompanyAgentOutput(events=[], claims=[], metrics=[]),
                failure_record(
                    "company_evidence",
                    payload,
                    CompanyAgentOutput,
                    error,
                ),
            )
        output = normalize_company_output(
            CompanyAgentOutput.model_validate(invocation.response),
            [source],
            task.decision_cutoff,
        )
        return document_key, output, _invocation_record(invocation)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        extracted = list(
            executor.map(extract_document, sorted(extraction_jobs.items()))
        )
    extraction_cache = {
        document_key: output for document_key, output, _ in extracted
    }
    invocation_records.extend(record for _, _, record in extracted)

    def process_card(context: tuple) -> tuple[dict, list[dict]]:
        item, task, expanded_evidence_ids = context
        session_macro = macro_cache[item["target_session"]]
        company_outputs = []
        for evidence_id in expanded_evidence_ids:
            source = by_id.get(evidence_id)
            if source is None or source.document_content_hash is None:
                continue
            document_key = source.document_content_hash or source.content_hash
            company_outputs.append(extraction_cache[document_key])
        company = normalize_company_output(
            _merge_company_outputs(company_outputs),
            [
                by_id[evidence_id]
                for evidence_id in expanded_evidence_ids
                if evidence_id in by_id
            ],
            task.decision_cutoff,
        )
        feature = _feature_record(
            feature_rows[(item["target_session"], item["symbol"])],
            dataset.name,
        )
        synthesis_payload = {
            "symbol": item["symbol"],
            "task_id": task.task_id,
            "market_score": item["market_score"],
            "feature": feature.model_dump(mode="json"),
            "company_claims": [
                claim.model_dump(mode="json") for claim in company.claims
            ],
            "company_metrics": [
                metric.model_dump(mode="json") for metric in company.metrics
            ],
            "events": [event.model_dump(mode="json") for event in company.events],
            "revision_feedback": [],
        }
        synthesis_error: ConnectionError | None = None
        try:
            synthesis_invocation = client.generate(
                "synthesis",
                synthesis_payload,
                SynthesisOutput,
            )
            synthesis_output = SynthesisOutput.model_validate(
                synthesis_invocation.response
            )
            synthesis_record = _invocation_record(synthesis_invocation)
        except ConnectionError as error:
            synthesis_error = error
            synthesis_output = SynthesisOutput()
            synthesis_record = failure_record(
                "synthesis",
                synthesis_payload,
                SynthesisOutput,
                error,
            )
        macro_context = MacroContext(
            task_id=task.task_id,
            events=session_macro,
            evidence_ids=[event.evidence_id for event in session_macro],
            assessments=macro_assessment_cache[item["target_session"]],
            target_session=task.target_session,
            decision_cutoff=task.decision_cutoff,
        )
        synthesis = align_synthesis_output(
            synthesis_output,
            company,
            macro_context,
        )
        card = ResearchCard(
            symbol=item["symbol"],
            task_id=task.task_id,
            decision_cutoff=task.decision_cutoff,
            target_session=task.target_session,
            features=feature,
            events=company.events,
            claims=company.claims,
            catalyst_assessment=synthesis.catalyst_assessment,
            macro_interactions=macro_interactions.get(
                item["target_session"], {}
            ).get(item["symbol"], []),
            prediction=Prediction(
                market_score=item["market_score"],
                rationale=synthesis.rationale,
                counterarguments=synthesis.counterarguments,
                uncertainty=synthesis.uncertainty,
            ),
        )
        relevant_evidence = [
            by_id[evidence_id]
            for evidence_id in expanded_evidence_ids
            if evidence_id in by_id
        ]
        relevant_evidence.extend(session_macro)
        deterministic = verify_card(card, relevant_evidence)
        card.deterministic_verification = deterministic
        card_invocations = [synthesis_record]
        if synthesis_error is not None:
            card.semantic_verification = VerificationReport(
                verdict=Verdict.ABSTAIN,
                issues=[
                    VerificationIssue(
                        code="SCHEMA_ERROR",
                        severity="blocking",
                        field="synthesis",
                        required_action="retry_model",
                    )
                ],
            )
        elif deterministic.verdict == Verdict.PASS:
            verifier_payload = {
                "symbol": card.symbol,
                "semantic_scope": semantic_scope(card),
                "trusted_computed_context": {
                    "feature_id": card.features.feature_id,
                    "market_score": card.prediction.market_score,
                },
                "evidence": semantic_evidence_payloads(
                    card,
                    relevant_evidence,
                    retrieval.top_n,
                ),
                "attempt": 0,
            }
            try:
                verifier_invocation = client.generate(
                    "verifier",
                    verifier_payload,
                    type(deterministic),
                )
                card_invocations.append(_invocation_record(verifier_invocation))
                card.semantic_verification = normalize_semantic_verification(
                    type(deterministic).model_validate(
                        verifier_invocation.response
                    )
                )
            except ConnectionError as error:
                card_invocations.append(
                    failure_record(
                        "verifier",
                        verifier_payload,
                        type(deterministic),
                        error,
                    )
                )
                card.semantic_verification = VerificationReport(
                    verdict=Verdict.ABSTAIN,
                    issues=[
                        VerificationIssue(
                            code="SCHEMA_ERROR",
                            severity="blocking",
                            field="semantic_verifier",
                            required_action="retry_model",
                        )
                    ],
                )
        card.eligibility = Eligibility(
            eligible=(
                card.deterministic_verification.verdict == Verdict.PASS
                and card.semantic_verification is not None
                and card.semantic_verification.verdict == Verdict.PASS
            )
        )
        return (
            {
                "target_session": item["target_session"],
                "symbol": item["symbol"],
                "market_score": item["market_score"],
                "company_evidence_ids": expanded_evidence_ids,
                "macro_evidence_ids": [
                    event.evidence_id for event in session_macro
                ],
                "macro_assessments": [
                    assessment.model_dump(mode="json")
                    for assessment in macro_context.assessments
                ],
                "card": card.model_dump(mode="json"),
            },
            card_invocations,
        )

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        processed = list(executor.map(process_card, card_contexts))
    artifacts = [artifact for artifact, _ in processed]
    invocation_records.extend(
        invocation
        for _, invocations in processed
        for invocation in invocations
    )
    _atomic_json(artifact_path, artifacts)
    invocation_path = output / f"invocations_{mode}.json"
    _atomic_json(invocation_path, invocation_records)
    usage_complete = not any(
        item["source"] == "cache"
        and item["input_tokens"] == 0
        and item["output_tokens"] == 0
        for item in invocation_records
    )
    report = {
        "version": 4,
        "mode": mode,
        "labels_loaded": False,
        "dataset_binding": {
            "snapshot_id": dataset.name,
            "manifest_sha256": _file_hash(dataset / "manifest.json"),
            "split_hash": json.loads(
                (dataset / "splits.json").read_text()
            )["content_hash"],
        },
        "base_config_path": str(base_config_path),
        "candidate_cards_path": str(candidate_cards_path),
        "feature_cache_path": str(feature_cache_path),
        "prompt_versions": {
            name: role.prompt_version for name, role in agents.roles.items()
        },
        "schema_hashes": {
            schema.__name__: hashlib.sha256(
                json.dumps(
                    schema.model_json_schema(),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
            for schema in (
                CompanyAgentOutput,
                MacroAgentOutput,
                SynthesisOutput,
                VerificationReport,
                ResearchCard,
            )
        },
        "requested_card_limit": maximum_cards,
        "requested_session_limit": maximum_sessions,
        "selected_sessions": selected_sessions,
        "session_count": len(selected_sessions),
        "cards_per_session": {
            session: sum(item["target_session"] == session for item in selected)
            for session in selected_sessions
        },
        "card_count": len(artifacts),
        "unique_document_extractions": len(extraction_cache),
        "verified_card_count": sum(
            item["card"]["eligibility"]["eligible"] for item in artifacts
        ),
        "encodable_card_count": sum(
            bool(llm_residual_features(ResearchCard.model_validate(item["card"])))
            for item in artifacts
        ),
        "eligible_encodable_card_count": sum(
            item["card"]["eligibility"]["eligible"]
            and bool(
                llm_residual_features(
                    ResearchCard.model_validate(item["card"])
                )
            )
            for item in artifacts
        ),
        "macro_interaction_card_count": sum(
            bool(item["card"].get("macro_interactions"))
            for item in artifacts
        ),
        "invocation_count": len(invocation_records),
        "failed_invocation_count": sum(
            item["source"] == "failed" for item in invocation_records
        ),
        "failure_rate": (
            sum(item["source"] == "failed" for item in invocation_records)
            / len(invocation_records)
            if invocation_records
            else 0.0
        ),
        "usage": {
            "complete": usage_complete,
            "input_tokens": sum(item["input_tokens"] for item in invocation_records),
            "output_tokens": sum(item["output_tokens"] for item in invocation_records),
            "estimated_cost_usd": (
                (
                    sum(item["input_tokens"] for item in invocation_records)
                    * agents.input_cost_per_million
                    + sum(item["output_tokens"] for item in invocation_records)
                    * agents.output_cost_per_million
                )
                / 1_000_000
                if usage_complete
                and (
                    agents.input_cost_per_million > 0
                    or agents.output_cost_per_million > 0
                )
                else None
            ),
        },
        "artifact_sha256": _file_hash(artifact_path),
        "invocations_sha256": _file_hash(invocation_path),
    }
    report_path = output / f"generation_report_{mode}.json"
    _atomic_json(report_path, report)
    _atomic_json(
        manifest_path,
        {
            "version": 4,
            "files": {
                artifact_path.name: _file_hash(artifact_path),
                invocation_path.name: _file_hash(invocation_path),
                report_path.name: _file_hash(report_path),
                **{
                    f"recordings/{request_hash}.json": _file_hash(
                        output / "recordings" / f"{request_hash}.json"
                    )
                    for request_hash in sorted(
                        {item["request_hash"] for item in invocation_records}
                    )
                    if (
                        output / "recordings" / f"{request_hash}.json"
                    ).exists()
                },
            },
        },
    )
    return artifact_path, report


def _build_macro_interactions(
    *,
    dataset: Path,
    sessions: list[str],
    target_events: dict[str, list],
    history_events: list,
    surprise_scales: dict[str, float],
) -> dict[str, dict[str, list]]:
    """Build point-in-time macro interactions without opening target labels."""

    universe = json.loads((dataset / "universe/symbols.json").read_text())
    sectors = {
        item["symbol"]: item["sector"] for item in universe["securities"]
    }
    exposure_config = json.loads(
        (ROOT / "configs/sector_exposures.json").read_text()
    )
    exposures = {
        symbol: SymbolMacroExposure(
            symbol=symbol,
            sector=sector,
            exposure_method="sector_heuristic_v1",
            template_version=exposure_config["version"],
            exposures=exposure_config["sectors"][sector],
        )
        for symbol, sector in sectors.items()
    }
    signed_returns: dict[str, dict[str, float]] = {}
    for row in pl.read_parquet(
        dataset / "market/daily_bars.parquet"
    ).iter_rows(named=True):
        signed_returns.setdefault(row["session"].isoformat(), {})[
            row["symbol"]
        ] = (row["close"] - row["open"]) / row["open"]

    return {
        session: build_macro_interactions(
            target_session=date.fromisoformat(session),
            target_events=target_events.get(session, []),
            history_events=history_events,
            returns=signed_returns,
            sectors=sectors,
            exposures=exposures,
            surprise_scales=surprise_scales,
        )
        for session in sessions
    }


def _merge_company_outputs(outputs: list[CompanyAgentOutput]) -> CompanyAgentOutput:
    return CompanyAgentOutput(
        events=[event for output in outputs for event in output.events],
        claims=[claim for output in outputs for claim in output.claims],
        metrics=[metric for output in outputs for metric in output.metrics],
    )


def _evenly_spaced[ValueT](values: list[ValueT], limit: int) -> list[ValueT]:
    """Choose a label-blind chronological pilot spanning the full train period."""

    if limit >= len(values):
        return values
    if limit <= 0:
        return []
    positions = np.linspace(0, len(values) - 1, num=limit, dtype=int)
    return [values[int(position)] for position in positions]


def _select_grouped_sessions(values: list[dict], limit: int) -> list[dict]:
    """Select sessions without breaking their within-session candidate cross-section."""

    sessions = sorted({item["target_session"] for item in values})
    selected_sessions = set(_evenly_spaced(sessions, limit))
    return [item for item in values if item["target_session"] in selected_sessions]


def _feature_record(row: dict, snapshot_id: str) -> FeatureRecord:
    values = {
        name: value
        for name, value in row.items()
        if name not in {"target_session", "symbol"}
    }
    return FeatureRecord(
        feature_id=f"feature_{row['symbol']}_{row['target_session'].isoformat()}",
        symbol=row["symbol"],
        snapshot_id=snapshot_id,
        formula_version="v2",
        input_columns=["open", "high", "low", "close", "volume"],
        input_sessions=[row["target_session"]],
        **values,
    )


def _invocation_record(invocation: object) -> dict:
    return {
        "role": invocation.role,
        "request_hash": invocation.request_hash,
        "source": invocation.source,
        "model": invocation.model,
        "prompt_version": invocation.prompt_version,
        "schema_name": invocation.schema_name,
        "input_tokens": invocation.input_tokens,
        "output_tokens": invocation.output_tokens,
        "latency_seconds": invocation.latency_seconds,
    }


def _atomic_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("scripted", "recorded", "live"),
        default="recorded",
    )
    parser.add_argument("--maximum-cards", type=int, default=200)
    parser.add_argument(
        "--maximum-sessions",
        type=int,
        default=None,
        help="Select complete event-bearing candidate groups for this many sessions.",
    )
    parser.add_argument("--maximum-model-calls", type=int, default=1_000)
    parser.add_argument("--maximum-input-tokens", type=int, default=4_000_000)
    parser.add_argument("--maximum-output-tokens", type=int, default=500_000)
    parser.add_argument("--maximum-cost-usd", type=float, default=100.0)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--base-config", type=Path, default=None)
    parser.add_argument("--candidate-cards", type=Path, default=None)
    parser.add_argument("--feature-cache", type=Path, default=None)
    parser.add_argument("--reuse-directory", type=Path, default=None)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "runs/llm_residual_session40",
    )
    arguments = parser.parse_args()
    path, report = generate_artifacts(
        mode=arguments.mode,
        maximum_cards=arguments.maximum_cards,
        maximum_model_calls=arguments.maximum_model_calls,
        output_directory=arguments.output_directory,
        maximum_sessions=arguments.maximum_sessions,
        maximum_input_tokens=arguments.maximum_input_tokens,
        maximum_output_tokens=arguments.maximum_output_tokens,
        maximum_cost_usd=arguments.maximum_cost_usd,
        parallel_workers=arguments.parallel_workers,
        base_config_path=arguments.base_config,
        candidate_cards_path=arguments.candidate_cards,
        feature_cache_path=arguments.feature_cache,
        reuse_directory=arguments.reuse_directory,
    )
    print(json.dumps({"artifact": str(path), **report}))


if __name__ == "__main__":
    main()
