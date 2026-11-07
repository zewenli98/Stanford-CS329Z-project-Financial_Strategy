"""Bounded planner, local ReAct researcher, verifier, and mean forecaster.

Roles reuse the repository's four client slots. Every prompt declares its actual
operation explicitly; cards separate supported observations from uncertain impact
inferences. A factual verification pass is not a guarantee about future prices.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from financial_strategy.adaptive.models import (
    ForecastOutput,
    InfoCard,
    PlannerOutput,
    RangeForecast,
    ResearchQuestion,
    SingleAgentOutput,
    WorkerAction,
)
from financial_strategy.adaptive.tools import OpaqueToolView, ResearchToolError, _plain
from financial_strategy.schemas import Verdict, VerificationReport


@dataclass
class ResearchResult:
    question_id: str
    accepted_cards: list[InfoCard] = field(default_factory=list)
    rejected_cards: list[InfoCard] = field(default_factory=list)
    forecasts: list[RangeForecast] = field(default_factory=list)
    tool_traces: list[dict] = field(default_factory=list)
    attempts: list[Any] = field(default_factory=list)
    stop_reason: str = "worker_limit"


def _numbers(text: str) -> set[str]:
    from decimal import Decimal

    text = re.sub(r"\b(?:company|source|event|card|question)_[a-zA-Z0-9_]+\b", "", text)
    # A dash between positive endpoints is a range separator, not a unary minus.
    # Preserve genuine negatives, including a negative second endpoint and exponents.
    text = re.sub(
        r"(\d[\d,]*(?:\.\d+)?\s*(?:%|[kmbt]|million|billion|trillion)?)"
        r"\s*[-\u2013\u2014]\s*(?=[$\u20ac\u00a3]?\d)",
        r"\1 to ",
        text,
        flags=re.IGNORECASE,
    )
    matches = re.findall(r"(?<![\w])[-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    return {str(Decimal(value.replace(",", "")).normalize()) for value in matches}


_MONTHS = {
    name: i
    for i, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1
    )
}
_DATE_MENTION = re.compile(
    r"\b(?P<iso>\d{4}-\d{2}-\d{2})\b|"
    r"\b(?P<month>Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s+(?P<day>\d{1,2})(?:st|nd|rd|th)?(?:,?\s+(?P<year>\d{4}))?\b",
    re.IGNORECASE,
)


def _date_mentions(text: str, default_year: int):
    for match in _DATE_MENTION.finditer(text):
        try:
            value = (
                date.fromisoformat(match["iso"])
                if match["iso"]
                else date(
                    int(match["year"] or default_year),
                    _MONTHS[match["month"][:3].lower()],
                    int(match["day"]),
                )
            )
        except ValueError:
            value = None
        yield match, value


def _fact_numbers(text: str, supported_dates: set[date], default_year: int):
    """Validate dates as dates; never whitelist their components as financial facts."""
    chunks = []
    previous = 0
    invalid_date = False
    for match, value in _date_mentions(text, default_year):
        chunks.append(text[previous : match.start()])
        if value not in supported_dates:
            invalid_date = True
            chunks.append(match.group())
        previous = match.end()
    chunks.append(text[previous:])
    return _numbers(" ".join(chunks)), invalid_date


def verify_card_deterministic(
    card: InfoCard,
    question: ResearchQuestion,
    view: OpaqueToolView,
    read_spans: list[dict] | None = None,
) -> list[str]:
    """Validate references, exact visible spans, timestamps, entity, and fact numbers."""
    issues: list[str] = []
    if card.question_id != question.question_id:
        issues.append("WRONG_QUESTION")
    if set(card.symbols) - set(question.symbols) or set(card.symbols) - set(view.reverse_aliases):
        issues.append("WRONG_ENTITY")
    if card.available_at > view.cutoff:
        issues.append("TIME_BOUNDARY_VIOLATION")
    if card.expires_at is not None and card.expires_at <= view.cutoff:
        issues.append("EXPIRED_CARD")
    source_text: list[str] = []
    times: list[datetime] = []
    event_ids: set[str] = set()
    event_times: set[datetime] = set()
    event_types: set[str] = set()
    for span in card.source_spans:
        record = view.evidence_map.get(span.evidence_id)
        if record is None:
            issues.append("MISSING_EVIDENCE")
            continue
        times.append(record.available_at)
        metadata = view._records[span.evidence_id]
        event_ids.add(metadata["event_id"])
        if record.event_time:
            event_times.add(record.event_time)
        if record.event_type:
            event_types.add(record.event_type)
        event_types.update(record.event_types)
        if record.available_at > view.cutoff:
            issues.append("TIME_BOUNDARY_VIOLATION")
        if record.scope == "company" and view.aliases.get(record.symbol) not in card.symbols:
            issues.append("WRONG_ENTITY")
        try:
            exact = view.source_span(span.evidence_id, span.start, span.end)
        except ResearchToolError:
            issues.append("INVALID_SOURCE_SPAN")
            continue
        if span.quote != exact["quote"]:
            issues.append("SOURCE_SPAN_MISMATCH")
        else:
            source_text.append(span.quote)
        if read_spans is not None and not any(
            read["evidence_id"] == span.evidence_id
            and read["start"] <= span.start < span.end <= read["end"]
            for read in read_spans
        ):
            issues.append("UNREAD_SOURCE")
    if times and card.available_at < max(times):
        issues.append("BACKDATED_AVAILABILITY")
    if event_ids and card.event_id not in event_ids:
        issues.append("UNSUPPORTED_EVENT_ID")
    if event_types and card.event_type not in event_types:
        issues.append("UNSUPPORTED_EVENT_TYPE")
    if card.event_time is not None and event_times and card.event_time not in event_times:
        issues.append("UNSUPPORTED_EVENT_TIME")
    if getattr(view, "protocol_v2", False):
        # Metadata supplied by the archive is an admissible source for filing dates/forms.
        source_text.extend(
            f"{view.evidence_map[s.evidence_id].published_at.isoformat()} "
            f"{view.evidence_map[s.evidence_id].available_at.isoformat()} "
            f"{view.evidence_map[s.evidence_id].published_at.strftime('%Y %m %d %H %M %S')} "
            f"{view.evidence_map[s.evidence_id].filing_form or ''}"
            for s in card.source_spans
            if s.evidence_id in view.evidence_map
        )
    default_year = view.task.target_session.year
    supported_dates = {view.task.target_session, view.cutoff.date()}
    for span in card.source_spans:
        record = view.evidence_map.get(span.evidence_id)
        if record is not None:
            supported_dates.update([record.published_at.date(), record.available_at.date()])
            if record.event_time is not None:
                supported_dates.add(record.event_time.date())
    for text in source_text:
        supported_dates.update(
            value for _, value in _date_mentions(text, default_year) if value is not None
        )
    supported_numbers, _ = _fact_numbers("\n".join(source_text), supported_dates, default_year)
    for observation in card.observations:
        numbers, invalid_date = _fact_numbers(observation, supported_dates, default_year)
        if invalid_date:
            issues.append("UNSUPPORTED_DATE_FACT")
        if numbers - supported_numbers:
            issues.append("UNSUPPORTED_NUMERICAL_FACT")
    return sorted(set(issues))


class ResearchAgentSuite:
    def __init__(
        self,
        client: Any,
        view: OpaqueToolView,
        memory: Any = None,
        limits: Any = None,
        *,
        max_cards_per_assignment: int = 1,
    ) -> None:
        self.client = client
        self.view = view
        self.memory = memory
        self.limits = limits
        self.max_cards_per_assignment = max_cards_per_assignment
        self.plan_attempts: list[Any] = []
        self.forecast_attempts: list[Any] = []
        self.forecast_issues: list[dict] = []
        self.forecast_adjustments: list[dict] = []

    def _limit(self, key: str, default: int) -> int:
        if isinstance(self.limits, dict):
            return int(self.limits.get(key, default))
        return int(getattr(self.limits, key, default))

    def _protocol(self):
        if isinstance(self.limits, dict):
            return self.limits.get("research_protocol", "legacy")
        return getattr(self.limits, "research_protocol", "legacy")

    def _generate(
        self,
        role: str,
        payload: dict,
        schema: Any,
        attempts: list[Any],
        *,
        attempt_type: str = "initial",
    ) -> Any:
        # Task context is authoritative input, not a fact that must be in a filing.
        payload = {
            **payload,
            "target_session": self.view.task.target_session.isoformat(),
            "decision_cutoff": self.view.cutoff.isoformat(),
        }
        # These instructions are authored by the harness; redact evidence and model
        # content without rewriting trusted control text (e.g. "Target the next session").
        visible = self.view.sanitize_model(
            {k: v for k, v in payload.items() if k != "instructions"}
        )
        if "instructions" in payload:
            visible["instructions"] = payload["instructions"]
        result = self.client.generate(role, visible, schema, attempt_type=attempt_type)
        attempts.append(result)
        return schema.model_validate(result.response)

    def plan(
        self,
        forecasts: list[RangeForecast] | dict[str, RangeForecast],
        *,
        reserved_cost_usd: float = 0.05,
        history: list[dict] | None = None,
        remaining_budget_usd: float | None = None,
    ) -> list[ResearchQuestion]:
        values = list(forecasts.values()) if isinstance(forecasts, dict) else forecasts
        payload = {
            "op": "adaptive_plan",
            "instructions": (
                "Plan bounded local evidence research for next regular-session (high-low)/low "
                "ranking, at the supplied premarket cutoff. Use only opaque company/source/event "
                "IDs in the context. Ask only about a dated lead that is already available: "
                "earnings, guidance, merger, buyback, financing, litigation, management change, "
                "cybersecurity, or product news. Do not create a question for a company with no "
                "available filing, and do not spend a question confirming the absence of news. "
                "Prefer a lead whose current range forecast is uncertain. Each question needs two "
                "or three mutually exclusive answer scenarios with probabilities summing to one "
                "and log_shifts to current mean range forecasts. These are subjective planning "
                "assumptions, not calibrated probabilities or verified future outcomes. State how "
                "an answer could change the regular-session range. Never claim archive exhaustion "
                "proves no real event exists. Use the supplied reserved_cost_usd for each question."
            ),
            "context": self.view.model_context(),
            "forecasts": [_plain(item) for item in values],
            "lead_index": self.view.lead_index(),
            "reserved_cost_usd": reserved_cost_usd,
            # Detailed tool observations remain in the audit. Repeating full
            # documents here consumes context without giving the planner new facts.
            "research_history": [
                {
                    "question_id": item.get("question_id", "unknown"),
                    "text": item.get("text", "")[:300],
                    "stop_reason": item.get("stop_reason"),
                    "accepted_card_ids": item.get("accepted_card_ids", []),
                }
                for item in (history or [])
            ],
            "remaining_budget_usd": remaining_budget_usd,
            "accepted_memory_cards": [
                {
                    "card_id": card.card_id,
                    "symbols": card.symbols,
                    "event_id": card.event_id,
                    "event_type": card.event_type,
                    "observations": [text[:240] for text in card.observations[:2]],
                    "range_mechanism": card.range_mechanism[:400],
                    "source_ids": [span.evidence_id for span in card.source_spans],
                    "available_at": card.available_at.isoformat(),
                }
                for card in (
                    self.memory.active(self.view.cutoff) if self.memory is not None else []
                )
                if all(span.evidence_id in self.view.evidence_map for span in card.source_spans)
            ],
        }
        if self._limit("compact_research_prompts", 0):
            payload["instructions"] = (
                "Plan at most SIX specific, evidence-grounded research questions for the coming "
                "regular-session (high-low)/low ranking. The harness automatically adds broad "
                "discovery questions for every company: DO NOT generate discovery questions. "
                "Each question must have two or three mutually exclusive outcomes with "
                "probabilities summing to one. log_shifts keys must be a subset of that exact "
                "question's symbols list. Every symbols/evidence_ids/event_ids/allowed_tools "
                "list must have UNIQUE entries. Several filings can share the same event_id: "
                "include that event_id only once. Use only supplied opaque IDs and the supplied "
                "reserved_cost_usd. Do not repeat closed questions in research_history. "
                "Return an empty questions list if there are no new specific leads. "
                "Scenario probabilities and impacts are subjective planning assumptions, "
                "not verified future outcomes. Explain uncertainty remaining after open."
            )
        output = self._generate("macro_evidence", payload, PlannerOutput, self.plan_attempts)
        questions = []
        discovery_by_symbol = {}
        for question in output.questions:
            if set(question.symbols) - set(self.view.reverse_aliases):
                continue
            if set(question.evidence_ids) - set(self.view.evidence_map):
                continue
            if set(question.event_ids) - {item["event_id"] for item in self.view.lead_index()}:
                continue
            question = question.model_copy(deep=True)
            question.reserved_cost_usd = reserved_cost_usd
            if not question.allowed_tools:
                question.allowed_tools = sorted(self.view.TOOL_NAMES)
            if question.discovery:
                if len(question.symbols) == 1:
                    discovery_by_symbol.setdefault(question.symbols[0], question)
            elif len(questions) < (6 if self._limit("compact_research_prompts", 0) else 20):
                questions.append(question)
        # Initial coverage is a harness invariant, not a request the planner may
        # silently omit when its output-token limit is reached.
        used_ids = {question.question_id for question in questions}
        for alias in sorted(self.view.reverse_aliases):
            discovery = discovery_by_symbol.get(alias)
            if discovery is None or discovery.question_id in used_ids:
                discovery = ResearchQuestion(
                    question_id=f"discovery_{alias}",
                    text="Inspect the permitted archive for overlooked range-relevant evidence.",
                    symbols=[alias],
                    evidence_ids=[],
                    event_ids=[],
                    outcomes=[],
                    resolution_probability=0.3,
                    discovery=True,
                    allowed_tools=sorted(self.view.TOOL_NAMES),
                    reserved_cost_usd=reserved_cost_usd,
                )
            while discovery.question_id in used_ids:
                discovery.question_id += "_broad"
            used_ids.add(discovery.question_id)
            questions.append(discovery)
        return questions

    def forecast_history(
        self, forecasts: list[RangeForecast] | dict[str, RangeForecast]
    ) -> list[RangeForecast]:
        """Explicit history-only control: model forecasts using market features only."""
        current = list(forecasts.values()) if isinstance(forecasts, dict) else list(forecasts)
        payload = {
            "op": "adaptive_history",
            "instructions": (
                "History-only experimental control. Predict the mean coming regular-session "
                "(high-low)/low in ratio units using only supplied historical market features. "
                "Return finite nonnegative means and preserve each baseline_range. No company "
                "events, memory cards, future data, or invented citations may be used."
            ),
            "forecasts": [_plain(item) for item in current],
            "market_features": [self.view.feature_payload(item.symbol) for item in current],
        }
        output = self._generate("synthesis", payload, ForecastOutput, self.forecast_attempts)
        updated = {item.symbol: item for item in current}
        for forecast in output.forecasts:
            old = updated.get(forecast.symbol)
            if old is None or forecast.baseline_range != old.baseline_range:
                self.forecast_issues.append({"symbol": forecast.symbol, "code": "WRONG_BASELINE"})
                continue
            if set(forecast.supporting_card_ids) - set(old.supporting_card_ids):
                self.forecast_issues.append(
                    {"symbol": forecast.symbol, "code": "INVENTED_CITATION"}
                )
                continue
            forecast = forecast.model_copy(deep=True)
            forecast.revision = old.revision + 1
            forecast.fallback = forecast.predicted_range == old.baseline_range
            updated[forecast.symbol] = forecast
        return [updated[item.symbol] for item in current]

    def _semantic_issues(
        self, card: InfoCard, attempts: list[Any], question=None, forecasts=None
    ) -> list[str]:
        payload = {
            "op": "adaptive_verify",
            "symbol": card.symbols[0],
            "instructions": (
                "Check the observations against exact supplied source spans. Reject unsupported "
                "facts, wrong attribution, fabricated numerical facts, and impact inferences "
                "presented as established facts. Check completeness for the stated research "
                "question and whether the range mechanism concerns the coming regular session. "
                "Uncertain, explicitly labeled impact inferences may be plausible without being "
                "verified predictions. Do not certify future outcomes. Return PASS only when "
                "supported; otherwise REVISE with concrete issues or ABSTAIN."
            ),
            "card": card.model_dump(mode="json"),
            "evidence": [span.model_dump(mode="json") for span in card.source_spans],
            "decision_cutoff": self.view.cutoff.isoformat(),
            "attempt": card.version - 1,
        }
        if self._protocol() != "legacy":
            payload["question"] = _plain(question)
            payload["target_session"] = self.view.task.target_session.isoformat()
            payload["market_baselines"] = [
                _plain(f) for f in forecasts or [] if f.symbol in card.symbols
            ]
            payload["instructions"] += (
                " Distinguish source facts from explicitly uncertain market hypotheses. "
                "A future effective date announced before cutoff is allowed. "
                "Baseline comparisons may refer to supplied market_baselines. "
                "ABSTAIN on filing-existence claims without a substantive catalyst. "
                "Check counterevidence, event age, and whether information may already be priced."
            )
        report = self._generate("verifier", payload, VerificationReport, attempts)
        if report.verdict == Verdict.PASS and not any(
            issue.severity == "blocking" for issue in report.issues
        ):
            return []
        return [
            f"{issue.code}: {issue.field}: {issue.required_action}" for issue in report.issues
        ] or [f"SEMANTIC_{report.verdict.value}"]

    def research(
        self,
        question: ResearchQuestion,
        forecasts: list[RangeForecast] | dict[str, RangeForecast],
        *,
        packet=None,
    ) -> ResearchResult:
        result = ResearchResult(question_id=question.question_id)
        values = list(forecasts.values()) if isinstance(forecasts, dict) else forecasts
        max_models = self._limit("max_worker_model_calls", 3)
        max_tools = self._limit("max_worker_tool_calls", 4)
        observations: list[dict] = list(packet or [])
        result.tool_traces.extend(observations)
        read_spans: list[dict] = [
            p["result"]["span"] for p in observations if "span" in p["result"]
        ]
        previous_cards: list[InfoCard] = []
        repair_issues: list[str] = []
        repaired = False
        tool_count = 0
        allowed_tools = set(question.allowed_tools or self.view.TOOL_NAMES)
        for step in range(max_models):
            payload = {
                "op": "adaptive_research",
                "symbol": question.symbols[0],
                "instructions": (
                    "Resolve the assigned question using only bounded local tools. Take one "
                    "search/read/features/timeline/cards action at a time, or finish/abstain. "
                    "Read sources before citing them. Finish with at most one InfoCard using "
                    "exact source spans from read results (half-open character offsets). Copy "
                    "event_id and event_type from source metadata. observations contain only "
                    "supported facts; inferences and range_mechanism explain uncertain impacts "
                    "on the coming regular-session high-low range. available_at cannot precede "
                    "the latest cited source availability. Never promote lack of archive evidence "
                    "to proof no significant real-world event exists."
                ),
                "question": question.model_dump(mode="json"),
                "decision_cutoff": self.view.cutoff.isoformat(),
                "forecasts": [_plain(item) for item in values if item.symbol in question.symbols],
                "observations": observations,
                "remaining_model_calls": max_models - step,
                "remaining_tool_calls": max_tools - tool_count,
                "previous_cards": [_plain(card) for card in previous_cards],
                "verification_issues": repair_issues,
            }
            v2 = self._protocol() != "legacy"
            if v2:
                payload["target_session"] = self.view.task.target_session.isoformat()
                payload["tool_contracts"] = {
                    "search": "query,symbol,scope(company|macro|null),top_n(1..20)",
                    "read": "evidence_id,start,end; at most4000 characters",
                    "cards": "symbol only",
                    "features": "symbol only",
                    "timeline": "symbol only",
                }
                payload["instructions"] += (
                    " Copy numerical facts in their original units and wording; put any conversion "
                    "in inferences. Do not make a card confirming only "
                    "filing existence or headers. "
                    "Use the target date; distinguish old news from an unresolved catalyst, "
                    "and include explicit unchanged-outlook or other counterevidence."
                )
            finish_due = step >= max_models - 1 - (1 if v2 and not repaired else 0)
            if packet is not None:
                payload["instructions"] += (
                    " The supplied packet is an initial set of read passages, not the complete "
                    "archive. If it contains only metadata or omits a needed passage, use the "
                    "remaining permitted read actions before deciding whether to abstain."
                )
            if (packet is not None or self._limit("compact_research_prompts", 0)) and finish_due:
                payload["final_action_required"] = True
                payload["instructions"] += (
                    " This is your FINAL action. Finish with one supported card using an "
                    "already read exact source span, or abstain with a reason. "
                    "No further tool action is permitted."
                )
            action = self._generate(
                "company_evidence",
                payload,
                WorkerAction,
                result.attempts,
                attempt_type="targeted_repair" if repaired else "initial",
            )
            if action.tool == "abstain":
                result.stop_reason = "archive_insufficient"
                return result
            if action.tool != "finish":
                if action.tool not in allowed_tools or tool_count >= max_tools:
                    result.stop_reason = "tool_limit"
                    return result
                tool_count += 1
                try:
                    output = self.view.dispatch(action)
                except ResearchToolError as exc:
                    output = {"error_code": exc.code}
                trace = {
                    "step": step,
                    "tool": action.tool,
                    "arguments": action.arguments,
                    "result": output,
                }
                result.tool_traces.append(trace)
                observations.append(trace)
                if "span" in output:
                    read_spans.append(output["span"])
                continue
            if not action.cards:
                result.stop_reason = "no_card"
                return result
            if len(action.cards) > self.max_cards_per_assignment:
                repair_issues = ["CARD_LIMIT_EXCEEDED: finish with at most one info card"]
                previous_cards = action.cards
                for card in action.cards:
                    rejected = card.model_copy(deep=True)
                    rejected.status = "rejected"
                    rejected.verification_issues = repair_issues
                    result.rejected_cards.append(rejected)
            else:
                card = action.cards[0].model_copy(deep=True)
                card.status = "pending"
                card.version = 2 if repaired else 1
                issues = verify_card_deterministic(card, question, self.view, read_spans)
                if not issues:
                    issues = self._semantic_issues(card, result.attempts, question, values)
                if not issues:
                    card.status = "accepted"
                    card.verification_issues = []
                    result.accepted_cards.append(card)
                    result.stop_reason = "accepted"
                    return result
                card.status = "rejected"
                card.verification_issues = issues
                result.rejected_cards.append(card)
                previous_cards = [card]
                repair_issues = issues
            if repaired:
                result.stop_reason = "repair_rejected"
                return result
            repaired = True
        result.stop_reason = "repair_limit" if repair_issues else "worker_limit"
        return result

    def single_agent(
        self,
        question: ResearchQuestion,
        forecasts: list[RangeForecast] | dict[str, RangeForecast],
        *,
        packet: list[dict],
    ) -> ResearchResult:
        """Run the fair one-shot LLM control without independent semantic verification."""
        result = ResearchResult(question_id=question.question_id)
        values = list(forecasts.values()) if isinstance(forecasts, dict) else forecasts
        relevant = [item for item in values if item.symbol in question.symbols]
        read_spans = [
            item["result"]["span"]
            for item in packet
            if isinstance(item.get("result"), dict) and "span" in item["result"]
        ]
        payload = {
            "op": "adaptive_single_agent",
            "symbol": question.symbols[0],
            "instructions": (
                "Act as a single financial-research agent. Using only the supplied point-in-time "
                "source packets, decide whether there is one material catalyst for the coming "
                "regular session. If supported, return one evidence card with an exact supplied "
                "source span and one updated range forecast for the same company. Otherwise return "
                "empty cards and forecasts. Preserve baseline_range, cite the returned card in "
                "supporting_card_ids, and keep predicted_range within the configured relative "
                "bound. Separate source facts from uncertain range-impact inferences. Do not "
                "invent facts, "
                "IDs, sources, future outcomes, or external lookups. Negative news may increase "
                "absolute range; forecast volatility rather than price direction."
            ),
            "question": question.model_dump(mode="json"),
            "decision_cutoff": self.view.cutoff.isoformat(),
            "target_session": self.view.task.target_session.isoformat(),
            "forecasts": [_plain(item) for item in relevant],
            "source_packets": packet,
            "max_forecast_relative_change": self._relative_change_limit(),
        }
        try:
            output = self._generate(
                "synthesis", payload, SingleAgentOutput, result.attempts
            )
        except Exception:
            result.stop_reason = "invalid_single_agent_output"
            return result
        if not output.cards:
            result.stop_reason = "abstained"
            return result
        card = output.cards[0].model_copy(deep=True)
        card.status = "pending"
        issues = verify_card_deterministic(card, question, self.view, read_spans)
        if issues:
            card.status = "rejected"
            card.verification_issues = issues
            result.rejected_cards.append(card)
            result.stop_reason = "deterministic_rejected"
            return result
        if len(output.forecasts) != 1:
            card.status = "rejected"
            card.verification_issues = ["MISSING_FORECAST"]
            result.rejected_cards.append(card)
            result.stop_reason = "invalid_single_agent_forecast"
            return result
        forecast = output.forecasts[0]
        old = next((item for item in relevant if item.symbol == forecast.symbol), None)
        if (
            old is None
            or forecast.baseline_range != old.baseline_range
            or card.card_id not in forecast.supporting_card_ids
            or set(forecast.supporting_card_ids) - ({card.card_id} | set(old.supporting_card_ids))
        ):
            card.status = "rejected"
            card.verification_issues = ["INVALID_FORECAST_LINK"]
            result.rejected_cards.append(card)
            result.stop_reason = "invalid_single_agent_forecast"
            return result
        card.status = "accepted"
        card.verification_issues = []
        result.accepted_cards.append(card)
        result.forecasts.append(forecast)
        result.stop_reason = "accepted"
        return result

    def apply_single_agent_forecasts(
        self,
        forecasts: list[RangeForecast] | dict[str, RangeForecast],
        proposed: list[RangeForecast],
        cards: list[InfoCard],
    ) -> list[RangeForecast]:
        current = list(forecasts.values()) if isinstance(forecasts, dict) else list(forecasts)
        by_symbol = {item.symbol: item for item in current}
        accepted = {card.card_id: card for card in cards if card.status == "accepted"}
        for forecast in proposed:
            old = by_symbol.get(forecast.symbol)
            cited = set(forecast.supporting_card_ids)
            new_ids = cited - (set(old.supporting_card_ids) if old else set())
            if (
                old is None
                or forecast.baseline_range != old.baseline_range
                or not new_ids
                or new_ids - set(accepted)
                or any(forecast.symbol not in accepted[card_id].symbols for card_id in new_ids)
            ):
                self.forecast_issues.append(
                    {"symbol": forecast.symbol, "code": "INVALID_SINGLE_AGENT_FORECAST"}
                )
                continue
            forecast = forecast.model_copy(deep=True)
            proposed_range = forecast.predicted_range
            limit = self._relative_change_limit()
            lower = old.baseline_range * (1 - limit)
            upper = old.baseline_range * (1 + limit)
            forecast.predicted_range = min(upper, max(lower, proposed_range))
            forecast.supporting_card_ids = sorted(set(old.supporting_card_ids) | cited)
            forecast.revision = old.revision + 1
            forecast.fallback = False
            self.forecast_adjustments.append(
                {
                    "symbol": forecast.symbol,
                    "baseline_range": old.baseline_range,
                    "proposed_range": proposed_range,
                    "bounded_range": forecast.predicted_range,
                    "relative_change": (
                        forecast.predicted_range / old.baseline_range - 1
                        if old.baseline_range > 0
                        else 0.0
                    ),
                    "was_clamped": proposed_range != forecast.predicted_range,
                    "supporting_card_ids": forecast.supporting_card_ids,
                    "source": "single_agent",
                }
            )
            by_symbol[forecast.symbol] = forecast
        return [by_symbol[item.symbol] for item in current]

    def update_forecasts(
        self,
        forecasts: list[RangeForecast] | dict[str, RangeForecast],
        cards: list[InfoCard],
    ) -> list[RangeForecast]:
        current = list(forecasts.values()) if isinstance(forecasts, dict) else list(forecasts)
        by_symbol = {item.symbol: item for item in current}
        new_cards = [
            card
            for card in cards
            if card.status == "accepted"
            and card.available_at <= self.view.cutoff
            and all(span.evidence_id in self.view.evidence_map for span in card.source_spans)
            and any(
                symbol in by_symbol and card.card_id not in by_symbol[symbol].supporting_card_ids
                for symbol in card.symbols
            )
        ]
        if not new_cards:
            return current
        accepted_by_id = {card.card_id: card for card in new_cards}
        affected = sorted(
            {
                symbol
                for card in new_cards
                for symbol in card.symbols
                if symbol in by_symbol and card.card_id not in by_symbol[symbol].supporting_card_ids
            }
        )
        if not affected:
            return current
        payload = {
            "op": "adaptive_forecast",
            "instructions": (
                "Update expected regular-session (high-low)/low in ratio units, using only "
                "new accepted info cards. Output finite nonnegative mean forecasts for affected "
                "companies. Preserve baseline_range. Cite every new card that justifies an "
                "update in supporting_card_ids; distinguish uncertain causal interpretation from "
                "observed facts. Leave a forecast unchanged when evidence adds no material signal. "
                "Any update must remain within the configured relative bound around "
                "baseline_range. Negative news can increase absolute range; assess volatility, "
                "not price direction. "
                "No price target, probability calibration, or trade instruction is requested."
            ),
            "forecasts": [_plain(by_symbol[symbol]) for symbol in affected],
            "accepted_cards": [_plain(card) for card in new_cards],
            "max_forecast_relative_change": self._relative_change_limit(),
        }
        output = self._generate("synthesis", payload, ForecastOutput, self.forecast_attempts)
        for forecast in output.forecasts:
            old = by_symbol.get(forecast.symbol)
            if old is None or forecast.symbol not in affected:
                self.forecast_issues.append({"symbol": forecast.symbol, "code": "WRONG_ENTITY"})
                continue
            cited = set(forecast.supporting_card_ids)
            prior_ids = set(old.supporting_card_ids)
            new_ids = cited & set(accepted_by_id) - prior_ids
            relevant = all(
                forecast.symbol in accepted_by_id[card_id].symbols for card_id in new_ids
            )
            if (
                cited - prior_ids - set(accepted_by_id)
                or not relevant
                or forecast.baseline_range != old.baseline_range
                or (forecast.predicted_range != old.predicted_range and not new_ids)
            ):
                self.forecast_issues.append(
                    {"symbol": forecast.symbol, "code": "UNSUPPORTED_FORECAST_UPDATE"}
                )
                continue
            forecast = forecast.model_copy(deep=True)
            proposed = forecast.predicted_range
            limit = self._relative_change_limit()
            lower = old.baseline_range * (1 - limit)
            upper = old.baseline_range * (1 + limit)
            forecast.predicted_range = min(upper, max(lower, proposed))
            self.forecast_adjustments.append(
                {
                    "symbol": forecast.symbol,
                    "baseline_range": old.baseline_range,
                    "proposed_range": proposed,
                    "bounded_range": forecast.predicted_range,
                    "relative_change": (
                        forecast.predicted_range / old.baseline_range - 1
                        if old.baseline_range > 0
                        else 0.0
                    ),
                    "was_clamped": proposed != forecast.predicted_range,
                    "supporting_card_ids": sorted(prior_ids | cited),
                }
            )
            forecast.supporting_card_ids = sorted(prior_ids | cited)
            forecast.revision = old.revision + 1
            forecast.fallback = old.fallback if not new_ids else False
            by_symbol[forecast.symbol] = forecast
        return [by_symbol[item.symbol] for item in current]

    def _relative_change_limit(self) -> float:
        if isinstance(self.limits, dict):
            return float(self.limits.get("max_forecast_relative_change", 0.2))
        return float(getattr(self.limits, "max_forecast_relative_change", 0.2))


def deterministic_responder(role: str, payload: dict) -> dict:
    """Offline integration behavior, deliberately not claimed as financial intelligence."""
    op = payload.get("op")
    if op == "adaptive_plan":
        questions = []
        for index, lead in enumerate(payload["lead_index"]):
            symbols = (
                [lead["symbol"]]
                if lead["symbol"]
                else [item["symbol"] for item in payload["forecasts"]]
            )
            questions.append(
                {
                    "question_id": f"question_{index:04d}",
                    "text": (
                        "Does this source establish newly released information "
                        "or repeat an earlier update?"
                    ),
                    "symbols": symbols,
                    "evidence_ids": [lead["evidence_id"]],
                    "event_ids": [lead["event_id"]],
                    "outcomes": [
                        {
                            "label": "new_information",
                            "probability": 0.5,
                            "log_shifts": dict.fromkeys(symbols, 0.15),
                        },
                        {
                            "label": "repeated_or_resolved_before_open",
                            "probability": 0.5,
                            "log_shifts": dict.fromkeys(symbols, -0.15),
                        },
                    ],
                    "resolution_probability": 0.6,
                    "discovery": False,
                    "allowed_tools": ["search", "read", "features", "timeline", "cards"],
                    "reserved_cost_usd": payload["reserved_cost_usd"],
                }
            )
        for forecast in payload["forecasts"]:
            questions.append(
                {
                    "question_id": "discovery_" + forecast["symbol"],
                    "text": "Inspect the permitted archive for overlooked range-relevant evidence.",
                    "symbols": [forecast["symbol"]],
                    "evidence_ids": [],
                    "event_ids": [],
                    "outcomes": [],
                    "resolution_probability": 0.3,
                    "discovery": True,
                    "allowed_tools": ["search", "read", "features", "timeline", "cards"],
                    "reserved_cost_usd": payload["reserved_cost_usd"],
                }
            )
        return {"questions": questions}
    if op == "adaptive_research":
        question = payload["question"]
        reads = [item["result"] for item in payload["observations"] if "span" in item["result"]]
        if not reads:
            if question["evidence_ids"]:
                return {
                    "tool": "read",
                    "arguments": {"evidence_id": question["evidence_ids"][0], "start": 0},
                }
            searches = [
                item["result"] for item in payload["observations"] if item["tool"] == "search"
            ]
            if not searches:
                return {
                    "tool": "search",
                    "arguments": {"symbol": question["symbols"][0], "query": "", "top_n": 2},
                }
            hits = searches[-1].get("hits", [])
            if not hits:
                return {"tool": "abstain", "reason": "No additional evidence in permitted archive."}
            return {
                "tool": "read",
                "arguments": {
                    "evidence_id": hits[0]["evidence_id"],
                    "start": 0,
                    "end": min(500, hits[0]["document_length"]),
                },
            }
        source = reads[-1]
        card_id = (
            "card_"
            + hashlib.sha256(
                f"{question['question_id']}:{source['event_id']}".encode()
            ).hexdigest()[:16]
        )
        return {
            "tool": "finish",
            "cards": [
                {
                    "card_id": card_id,
                    "question_id": question["question_id"],
                    "symbols": question["symbols"],
                    "event_id": source["event_id"],
                    "event_type": source["event_type"] or "unclassified",
                    "observations": [source["span"]["quote"]],
                    "inferences": [
                        "This event may leave uncertainty about regular-session repricing."
                    ],
                    "range_mechanism": (
                        "Unresolved repricing may widen the regular-session range; "
                        "effect uncertain."
                    ),
                    "source_spans": [source["span"]],
                    "available_at": source["available_at"],
                    "event_time": source.get("event_time"),
                    "counterevidence": [
                        "The information may already be reflected in the opening price."
                    ],
                    "missing_information": ["Future intraday response is unknown."],
                }
            ],
        }
    if op == "adaptive_verify":
        return {"verdict": "PASS", "issues": [], "attempt": payload.get("attempt", 0)}
    if op == "adaptive_history":
        return {
            "forecasts": [
                {**old, "rationale": ["Offline history control preserves the market baseline."]}
                for old in payload["forecasts"]
            ]
        }
    if op == "adaptive_forecast":
        forecasts = []
        for old in payload["forecasts"]:
            relevant = [
                card for card in payload["accepted_cards"] if old["symbol"] in card["symbols"]
            ]
            forecasts.append(
                {
                    **old,
                    "predicted_range": old["predicted_range"],
                    "supporting_card_ids": sorted(
                        set(old["supporting_card_ids"]) | {card["card_id"] for card in relevant}
                    ),
                    "rationale": [
                        "Offline mode preserves the baseline; it tests orchestration only."
                    ],
                    "fallback": not bool(relevant),
                    "revision": old["revision"] + 1,
                }
            )
        return {"forecasts": forecasts}
    raise ValueError(f"unsupported adaptive operation for role {role}")
