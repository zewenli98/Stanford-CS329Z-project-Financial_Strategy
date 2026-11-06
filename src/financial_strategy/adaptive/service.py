"""LangGraph orchestration for bounded adaptive range research."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TypedDict
from uuid import uuid4
from zoneinfo import ZoneInfo

import polars as pl
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from financial_strategy.adaptive.agents import ResearchAgentSuite, deterministic_responder
from financial_strategy.adaptive.allocation import allocate_batch
from financial_strategy.adaptive.budget import AdaptiveClient, BudgetLedger
from financial_strategy.adaptive.config import AdaptiveConfig, load_adaptive_config
from financial_strategy.adaptive.data import compute_range_features
from financial_strategy.adaptive.memory import AcceptedCardMemory
from financial_strategy.adaptive.models import InfoCard, RangeForecast
from financial_strategy.adaptive.research_v2 import (
    history_baselines,
    packet_questions,
    range_uncertainty,
    select_candidates,
)
from financial_strategy.adaptive.tools import OpaqueToolView
from financial_strategy.audit import RunWriter
from financial_strategy.config import canonical_json
from financial_strategy.data.calendar import TradingCalendar
from financial_strategy.data.gateway import PointInTimeGateway
from financial_strategy.data.provenance import verify_snapshot_provenance
from financial_strategy.data.providers import (
    LocalEvidenceProvider,
    LocalMarketDataProvider,
    LocalUniverseProvider,
)
from financial_strategy.errors import StrategyError
from financial_strategy.llm.client import ScriptedClient
from financial_strategy.task.validator import validate_and_freeze_task


class ResearchState(TypedDict):
    forecasts: list[RangeForecast]
    batch: int
    stop_reason: str | None


class RunMemory:
    """A run-local view; concurrent workers never mutate shared memory."""

    def __init__(self, cards, cutoff, ttl_days, backend=None):
        self.cards = list(cards)
        self.cutoff, self.ttl_days, self.backend = cutoff, ttl_days, backend

    def active(self, as_of=None):
        cutoff = as_of or self.cutoff
        superseded = {old for card in self.cards for old in card.supersedes}
        return [
            card
            for card in self.cards
            if card.status == "accepted"
            and card.available_at <= cutoff
            and card.card_id not in superseded
            and (card.expires_at is None or card.expires_at > cutoff)
        ]

    def append(self, cards):
        stored = []
        for card in cards:
            card = card.model_copy(deep=True)
            expiry = self.cutoff + timedelta(days=self.ttl_days)
            card.expires_at = min(card.expires_at, expiry) if card.expires_at else expiry
            stored.append(card)
        if self.backend:
            self.backend.append(stored)
        self.cards.extend(stored)
        return stored


def _question_key(question):
    content = {
        "text": " ".join(question.text.lower().split()),
        "symbols": sorted(question.symbols),
        "evidence": sorted(question.evidence_ids),
    }
    return hashlib.sha256(canonical_json(content)).hexdigest()


def _card_key(card):
    return (
        card.event_id,
        tuple(sorted(card.symbols)),
        tuple(sorted((s.evidence_id, s.start, s.end, s.quote) for s in card.source_spans)),
    )


def _freeze_documents(writer, snapshot, config):
    for scope, records, root in (
        ("company", snapshot.company_evidence, config.data.company_evidence.parent),
        ("macro", snapshot.macro_evidence, config.data.macro_events.parent),
    ):
        root = root.resolve()
        for record in records:
            if not record.content_path:
                continue
            source = (root / record.content_path).resolve()
            if not source.is_relative_to(root):
                raise StrategyError("UNSAFE_DOCUMENT_PATH", "source leaves configured root")
            if source.is_file():
                target = writer._path(f"snapshot/{scope}/{record.content_path}")
                target.write_bytes(source.read_bytes())
                writer._fsync_file(target)


def _execute(
    config,
    task,
    snapshot,
    features,
    universe,
    writer,
    *,
    initial_cards=None,
    memory=None,
    replay_base=None,
):
    ledger = BudgetLedger(config.adaptive.daily_budget_usd)
    if config.adaptive.policy == "baseline" and replay_base is None:
        replay_base = ScriptedClient(config.agents, deterministic_responder)
    client = AdaptiveClient(config, deterministic_responder, ledger, base=replay_base)
    partial_artifacts = {}

    def failure_audit():
        writer.write_jsonl("model_calls.jsonl", client.meter.records)
        writer.write_jsonl("model_attempts.jsonl", client.attempt_log)
        writer.write_jsonl("budget_ledger.jsonl", ledger.events)
        writer.write_json("budget.json", ledger.summary())
        for name, values in partial_artifacts.items():
            if name.endswith(".jsonl"):
                writer.write_jsonl(name, values)
            else:
                writer.write_json(name, values)

    writer._adaptive_failure_audit = failure_audit
    # Forecast the whole universe, but keep costly research context bounded.
    all_features = features
    baselines = history_baselines(features, snapshot.bars, config.adaptive.baseline_method)
    limit = config.adaptive.research_candidate_limit
    if limit is not None:
        candidates = select_candidates(
            baselines,
            snapshot.company_evidence,
            task.decision_cutoff,
            limit,
            config.adaptive.candidate_method,
            range_uncertainty(snapshot.bars),
        )
        features = {symbol: features[symbol] for symbol in sorted(candidates)}
    writer.write_json(
        "research_screening.json",
        {
            "eligible_count": len(all_features),
            "candidate_count": len(features),
            "candidate_limit": limit,
            "method": "descending_history_only_mean_range_20d_then_symbol",
            "candidate_symbols": sorted(features),
            "remaining_forecasts": "retain_history_only_baseline",
            "baseline_method": config.adaptive.baseline_method,
            "candidate_method": config.adaptive.candidate_method,
        },
    )
    writer.write_json("baseline_values.json", baselines)
    counters = {"tools": 0}
    counter_lock = threading.Lock()

    def reserve_tool(_name):
        with counter_lock:
            if counters["tools"] >= config.agents.max_total_tool_calls:
                raise StrategyError("TASK_TOOL_BUDGET_EXHAUSTED", "local tool limit reached")
            counters["tools"] += 1

    backend = memory
    if backend is None and replay_base is None:
        root = config.adaptive.memory_path or writer.temporary / "memory"
        arm = (
            f"{config.adaptive.policy}_budget_{config.adaptive.daily_budget_usd:g}"
            f"_replicate_{config.adaptive.replicate_id}"
        )
        backend = AcceptedCardMemory(
            root, writer.run_id, arm, task.decision_cutoff, config.adaptive.memory_days
        )
    prior = (
        initial_cards
        if initial_cards is not None
        else (backend.active(task.decision_cutoff) if backend else [])
    )
    run_memory = RunMemory([], task.decision_cutoff, config.adaptive.memory_days, backend)
    view = OpaqueToolView(
        snapshot,
        task,
        universe,
        features,
        company_root=writer.temporary / "snapshot/company",
        macro_root=writer.temporary / "snapshot/macro",
        reserve_tool=reserve_tool,
        memory=run_memory,
        protocol_v2=config.adaptive.research_protocol != "legacy",
    )
    prior = [InfoCard.model_validate(card) for card in prior]
    prior = [
        card
        for card in prior
        if set(card.symbols) <= set(view.reverse_aliases)
        and all(span.evidence_id in view.evidence_map for span in card.source_spans)
    ]
    run_memory.cards = prior
    writer.write_json("memory_initial.json", prior)
    writer.write_json("private_mapping.json", view.private_mapping())
    writer.write_json("model_context.json", view.model_context())
    # Broad deterministic coverage consumes tool calls, not model dollars.
    for alias in view.reverse_aliases:
        view.dispatch(
            {
                "tool": "search",
                "arguments": {"symbol": alias, "scope": "company", "query": "", "top_n": 2},
            }
        )
    view.dispatch({"tool": "search", "arguments": {"scope": "macro", "query": "", "top_n": 8}})
    suite = ResearchAgentSuite(client, view, run_memory, config.adaptive)
    forecasts = [
        RangeForecast(
            symbol=alias,
            baseline_range=baselines[symbol],
            predicted_range=baselines[symbol],
        )
        for symbol, alias in view.aliases.items()
    ]
    decisions, questions_log, research_log, revisions, rejected = [], [], [], [], []
    partial_artifacts.update(
        {
            "allocation_decisions.jsonl": decisions,
            "research_questions.jsonl": questions_log,
            "research_results.jsonl": research_log,
            "forecast_revisions.jsonl": revisions,
            "research_cards.json": run_memory.cards,
            "rejected_cards.json": rejected,
        }
    )
    inspected, closed, seen_cards = {}, set(), {_card_key(card) for card in prior}
    pending, reservations = {}, {}
    policy = config.adaptive.policy
    final_hold = None

    def update(current, cards, *, final=False):
        nonlocal final_hold
        if not cards:
            return current
        try:
            if final:
                with client.using(final_hold):
                    revised = suite.update_forecasts(current, cards)
            else:
                revised = suite.update_forecasts(current, cards)
        except Exception as exc:
            revisions.append(
                {"error": getattr(exc, "code", "INVALID_FORECAST"), "retained_previous": True}
            )
            return current
        revisions.append(
            {
                "before": [f.model_dump(mode="json") for f in current],
                "after": [f.model_dump(mode="json") for f in revised],
                "new_card_ids": [card.card_id for card in cards],
            }
        )
        return revised

    def plan_node(state):
        if state["batch"] >= config.adaptive.max_batches:
            return {**state, "stop_reason": "batch_limit"}
        try:
            proposed = (
                packet_questions(view, state["forecasts"], inspected, client.assignment_bound())
                if config.adaptive.research_protocol == "packet"
                else suite.plan(
                    state["forecasts"],
                    reserved_cost_usd=client.assignment_bound(),
                    history=research_log,
                    remaining_budget_usd=ledger.remaining,
                )
            )
        except (StrategyError, ValidationError) as exc:
            return {**state, "stop_reason": getattr(exc, "code", "INVALID_PLAN")}
        pending.clear()
        for question in proposed:
            if _question_key(question) not in closed:
                pending[question.question_id] = question
                questions_log.append(
                    {"batch": state["batch"], "question": question.model_dump(mode="json")}
                )
        return state

    def allocate_node(state):
        if state["stop_reason"]:
            return state
        means = {f.symbol: f.predicted_range for f in state["forecasts"]}
        batch = allocate_batch(
            list(pending.values()),
            means,
            top_k=task.top_k,
            batch_size=config.adaptive.max_workers,
            policy=policy,
            epsilon=config.adaptive.exploration_probability,
            inspection_counts=inspected,
            seed=config.adaptive.seed + state["batch"] * 1009,
            budget_usd=ledger.remaining,
        )
        if config.adaptive.research_protocol == "packet":
            # Deterministic freshness ordering; no planner or random discovery calls.
            from types import SimpleNamespace

            chosen = list(pending.values())[: config.adaptive.max_workers]
            batch = [
                SimpleNamespace(
                    question_id=q.question_id,
                    model_dump=lambda mode, q=q: {
                        "question_id": q.question_id,
                        "policy": "packet_freshness",
                        "priority": q.resolution_probability,
                    },
                )
                for q in chosen
            ]
        reservations.clear()
        for decision in batch:
            decisions.append({"batch": state["batch"], **decision.model_dump(mode="json")})
            if decision.question_id is not None:
                question = pending[decision.question_id]
                try:
                    reservations[question.question_id] = ledger.reserve(
                        question.reserved_cost_usd, label=question.question_id
                    )
                except StrategyError:
                    break
        if not reservations:
            reason = (
                "budget_exhausted"
                if pending and ledger.remaining < client.assignment_bound()
                else batch[-1].stop_reason
                if batch
                else "no_eligible_questions"
            )
            return {**state, "stop_reason": reason}
        return state

    def research_node(state):
        if state["stop_reason"]:
            return state

        def worker(question_id, reservation):
            question = pending[question_id]
            try:
                with client.using(reservation):
                    packet = (
                        view.packet_observations(question.symbols[0])
                        if config.adaptive.research_protocol == "packet"
                        else None
                    )
                    return suite.research(question, state["forecasts"], packet=packet)
            except Exception as exc:
                # A single bounded worker failure does not discard the market baseline.
                from financial_strategy.adaptive.agents import ResearchResult

                return ResearchResult(
                    question_id=question_id, stop_reason=getattr(exc, "code", type(exc).__name__)
                )
            finally:
                ledger.release(reservation)

        with ThreadPoolExecutor(max_workers=config.adaptive.max_workers) as pool:
            futures = [pool.submit(worker, key, hold) for key, hold in reservations.items()]
            results = sorted((future.result() for future in futures), key=lambda x: x.question_id)
        new_cards = []
        for result in results:
            question = pending[result.question_id]
            closed.add(_question_key(question))
            for symbol in question.symbols:
                inspected[symbol] = inspected.get(symbol, 0) + 1
            accepted = []
            for card in result.accepted_cards:
                if _card_key(card) in seen_cards:
                    card = card.model_copy(
                        update={"status": "rejected", "verification_issues": ["DUPLICATE_EVENT"]}
                    )
                    rejected.append(card)
                else:
                    seen_cards.add(_card_key(card))
                    accepted.append(card)
            new_cards.extend(accepted)
            rejected.extend(result.rejected_cards)
            research_log.append(
                {
                    "question_id": question.question_id,
                    "text": question.text,
                    "stop_reason": result.stop_reason,
                    "accepted_card_ids": [c.card_id for c in accepted],
                    "tool_traces": result.tool_traces,
                }
            )
        stored = run_memory.append(new_cards)
        current = update(state["forecasts"], stored)
        # Remember pending finalization if a non-final update could not be afforded.
        state = {**state, "forecasts": current, "batch": state["batch"] + 1}
        return state

    if forecasts and policy != "baseline":
        try:
            final_hold = ledger.reserve(client.bound("synthesis"), label="finalization")
            if policy == "history_only":
                with client.using(final_hold):
                    forecasts = suite.forecast_history(forecasts)
            else:
                if prior:
                    forecasts = update(forecasts, prior)
                builder = StateGraph(ResearchState)
                builder.add_node("plan", plan_node)
                builder.add_node("allocate", allocate_node)
                builder.add_node("research", research_node)
                builder.add_edge(START, "plan")
                builder.add_edge("plan", "allocate")
                builder.add_conditional_edges(
                    "allocate", lambda s: END if s["stop_reason"] else "research"
                )
                builder.add_edge("research", "plan")
                state = builder.compile().invoke(
                    {"forecasts": forecasts, "batch": 0, "stop_reason": None},
                    {"recursion_limit": config.adaptive.max_batches * 4 + 10},
                )
                forecasts = state["forecasts"]
                counters["stop_reason"] = state["stop_reason"]
                # Every accepted but unapplied card gets one reserved final attempt.
                applied = {cid for item in forecasts for cid in item.supporting_card_ids}
                remaining_cards = [
                    card for card in run_memory.active() if card.card_id not in applied
                ]
                forecasts = update(forecasts, remaining_cards, final=True)
        except StrategyError as exc:
            counters["stop_reason"] = exc.code
        finally:
            if final_hold:
                ledger.release(final_hold)
    counters.setdefault(
        "stop_reason",
        "history_only"
        if policy == "history_only"
        else "baseline"
        if policy == "baseline"
        else "empty_universe",
    )
    output = []
    for forecast in forecasts:
        row = forecast.model_dump(mode="json")
        row["symbol"] = view.resolve_private(forecast.symbol)
        output.append(row)
    for symbol in sorted(set(all_features) - set(features)):
        baseline = baselines[symbol]
        output.append(
            RangeForecast(
                symbol=symbol, baseline_range=baseline, predicted_range=baseline
            ).model_dump(mode="json")
        )
    output.sort(key=lambda row: (-row["predicted_range"], row["symbol"]))
    if config.adaptive.forecast_blend != 1:
        for row in output:
            row["predicted_range"] = row["baseline_range"] + config.adaptive.forecast_blend * (
                row["predicted_range"] - row["baseline_range"]
            )
        output.sort(key=lambda row: (-row["predicted_range"], row["symbol"]))
    watchlist = [{"rank": index, **row} for index, row in enumerate(output[: task.top_k], 1)]
    writer.write_json("range_predictions.json", output)
    writer.write_json("watchlist.json", watchlist)
    writer.write_json("research_cards.json", run_memory.cards)
    writer.write_json("rejected_cards.json", rejected)
    writer.write_jsonl("research_questions.jsonl", questions_log)
    writer.write_jsonl("allocation_decisions.jsonl", decisions)
    writer.write_jsonl("research_results.jsonl", research_log)
    writer.write_jsonl("forecast_revisions.jsonl", revisions)
    writer.write_json("forecast_issues.json", suite.forecast_issues)
    writer.write_jsonl("tool_traces.jsonl", sorted(view.trace, key=lambda x: canonical_json(x)))
    writer.write_jsonl("model_attempts.jsonl", client.attempt_log)
    writer.write_jsonl("model_calls.jsonl", client.meter.records)
    writer.write_jsonl("budget_ledger.jsonl", ledger.events)
    writer.write_json("budget.json", ledger.summary())
    writer.write_json("execution.json", counters)
    writer.write_json(
        "pricing.json",
        {
            "prices": config.adaptive.prices,
            "pricing_as_of": config.adaptive.pricing_as_of,
            "offline_reference": config.adaptive.offline_price,
            "offline_usage_is_estimated": config.agents.mode != "live",
        },
    )
    return watchlist


def run_adaptive(
    config_or_path,
    *,
    run_id=None,
    task_overrides=None,
    memory=None,
    _replay_inputs=None,
    _replay_base=None,
):
    config = (
        config_or_path
        if isinstance(config_or_path, AdaptiveConfig)
        else load_adaptive_config(config_or_path)
    )
    if task_overrides:
        data = config.model_dump(mode="python")
        data["task"].update(task_overrides)
        config = AdaptiveConfig.model_validate(data)
    # Exact requests can share within a replicate, never across stochastic replicates.
    config = config.model_copy(
        update={
            "agents": config.agents.model_copy(
                update={
                    "cache_path": config.agents.cache_path
                    / f"replicate_{config.adaptive.replicate_id}"
                }
            )
        }
    )
    run_id = run_id or f"adaptive_{uuid4().hex[:16]}"
    writer = RunWriter(Path(config.output["runs_path"]), run_id)
    try:
        writer.write_yaml("config.yaml", config)
        if _replay_inputs is None:
            source_root = config.data.universe.parent.parent
            if (source_root / "provenance.json").is_file():
                writer.write_json("source_provenance.json", verify_snapshot_provenance(source_root))
            calendar = TradingCalendar(config.calendar["name"])
            provider = LocalUniverseProvider(config.data.universe, config.data.sector_exposures)
            universe = provider.universe()
            task, exposures, warnings = validate_and_freeze_task(
                config, calendar, universe, provider.sector_exposures()
            )
            gateway = PointInTimeGateway(
                LocalMarketDataProvider(config.data.market_bars, config.data.corporate_actions),
                LocalEvidenceProvider(config.data.company_evidence, config.data.macro_events),
                calendar,
                config.data.input_adjustment_policy,
            )
            snapshot = gateway.freeze_inputs(task, exposures)
            symbols = {item.symbol for item in exposures}
            entities = {s["symbol"]: s["permanent_entity_id"] for s in universe["securities"]}
            bars = snapshot.bars.filter(pl.col("symbol").is_in(sorted(symbols)))
            if any(
                row["permanent_entity_id"] != entities[row["symbol"]]
                for row in bars.select("symbol", "permanent_entity_id").to_dicts()
            ):
                raise StrategyError("ENTITY_MISMATCH", "market entity is outside frozen universe")
            snapshot = replace(
                snapshot,
                bars=bars,
                company_evidence=tuple(
                    item for item in snapshot.company_evidence if item.symbol in symbols
                ),
            )
            if config.adaptive.evidence_previous_day_only:
                boundary = datetime.combine(
                    task.target_session, time(), ZoneInfo("America/New_York")
                )
                snapshot = replace(
                    snapshot,
                    company_evidence=tuple(
                        r
                        for r in snapshot.company_evidence
                        if r.available_at < boundary and r.published_at < boundary
                    ),
                    macro_evidence=tuple(
                        r
                        for r in snapshot.macro_evidence
                        if r.available_at < boundary and r.published_at < boundary
                    ),
                )
                writer.write_json(
                    "evidence_policy.json",
                    {
                        "boundary": boundary,
                        "rule": "published_and_available_before_target_midnight",
                    },
                )
            expected = calendar.completed_sessions_before(task.decision_cutoff, 21)
            features, rejected = compute_range_features(
                snapshot.bars,
                expected,
                symbol_sectors={item.symbol: item.sector for item in exposures},
            )
            _freeze_documents(writer, snapshot, config)
            initial_cards = None
        else:
            task, snapshot, features, universe, initial_cards, doc_root = _replay_inputs
            warnings, rejected = [], {}
            import shutil

            for scope in ("company", "macro"):
                source = doc_root / scope
                if source.is_dir():
                    shutil.copytree(
                        source, writer.temporary / "snapshot" / scope, dirs_exist_ok=True
                    )
        writer.write_json("task.json", task)
        writer.write_json("snapshot/universe.json", universe)
        writer.write_parquet("snapshot/market.parquet", snapshot.bars)
        writer.write_jsonl("snapshot/company_evidence.jsonl", list(snapshot.company_evidence))
        writer.write_jsonl("snapshot/macro_evidence.jsonl", list(snapshot.macro_evidence))
        writer.write_json("snapshot/exposures.json", list(snapshot.exposures))
        writer.write_json("snapshot/calendar.json", list(snapshot.calendar_schedule))
        writer.write_json(
            "snapshot/metadata.json",
            {
                "snapshot_id": snapshot.snapshot_id,
                "calendar_version": snapshot.calendar_version,
                "decision_timezone": getattr(task.decision_cutoff.tzinfo, "key", None),
            },
        )
        writer.write_json("range_features.json", {key: value for key, value in features.items()})
        writer.write_json("screening.json", {"warnings": warnings, "rejected": rejected})
        watchlist = _execute(
            config,
            task,
            snapshot,
            features,
            universe,
            writer,
            initial_cards=initial_cards,
            memory=memory,
            replay_base=_replay_base,
        )
        return writer.publish("completed"), watchlist
    except Exception as exc:
        if hasattr(writer, "_adaptive_failure_audit"):
            writer._adaptive_failure_audit()
        path = writer.publish(
            "failed",
            {
                "type": type(exc).__name__,
                "code": getattr(exc, "code", "ADAPTIVE_FAILED"),
                "message": str(exc),
            },
        )
        raise RuntimeError(f"Adaptive screening failed; audit saved at {path}: {exc}") from exc
