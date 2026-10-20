"""Structured agent roles. Retrieval remains deterministic and point-in-time bounded."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from pydantic import Field

from financial_strategy.config import MVPConfig
from financial_strategy.errors import StrategyError
from financial_strategy.llm import InvocationResult, LLMClient
from financial_strategy.retrieval.bm25 import search
from financial_strategy.schemas import (
    Claim,
    ClaimType,
    EventRecord,
    EvidenceRecord,
    FeatureRecord,
    MacroContext,
    MacroEvidenceRecord,
    ResearchCard,
    ScreeningTask,
    VerificationReport,
)
from financial_strategy.schemas.models import StrictModel


class MacroAgentOutput(StrictModel):
    evidence_ids: list[str]


class CompanyAgentOutput(StrictModel):
    events: list[EventRecord]
    claims: list[Claim]


class SynthesisOutput(StrictModel):
    claims: list[Claim]
    rationale: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    uncertainty: list[str] = Field(default_factory=list)


def scripted_responder(role: str, payload: dict) -> dict:
    """Deterministic offline behavior used by CI."""

    if role == "macro_evidence":
        return {"evidence_ids": [item["evidence_id"] for item in payload["evidence"]]}
    if role == "company_evidence":
        evidence = payload["evidence"]
        events = []
        claims = []
        for index, item in enumerate(evidence):
            event_type = item.get("event_type") or "routine_product_news"
            status = item.get("event_status") or "timing_unknown"
            events.append(
                {
                    "event_id": f"event_{item['evidence_id']}",
                    "event_type": event_type,
                    "status": status,
                    "scheduled_at": (
                        item.get("event_time")
                        if status.startswith("scheduled_")
                        else None
                    ),
                    "released_at": (
                        item["published_at"]
                        if status == "released_before_cutoff"
                        else None
                    ),
                    "confidence": item["timestamp_confidence"],
                    "evidence_ids": [item["evidence_id"]],
                    "sessions_ago": 1,
                }
            )
            claims.append(
                {
                    "claim_id": f"claim_company_{index}",
                    "text": item["excerpt"],
                    "claim_type": "observed_fact",
                    "evidence_ids": [item["evidence_id"]],
                }
            )
        return {"events": events, "claims": claims}
    if role == "synthesis":
        claims = list(payload["company_claims"])
        if payload["macro_evidence_ids"]:
            claims.append(
                {
                    "claim_id": "claim_macro_context",
                    "text": (
                        "The frozen macro context may interact with the symbol's "
                        "fixed exposures."
                    ),
                    "claim_type": "inference",
                    "evidence_ids": payload["macro_evidence_ids"],
                }
            )
        return {
            "claims": claims,
            "rationale": [
                "Deterministic market features and verified events may support elevated "
                "absolute open-to-close movement."
            ],
            "counterarguments": [
                "Historical volatility and event presence do not guarantee a large move."
            ],
            "uncertainty": ["The score is a heuristic rank, not a calibrated probability."],
        }
    if role == "verifier":
        return {"verdict": "PASS", "issues": [], "attempt": payload["attempt"]}
    raise KeyError(f"Unknown scripted role: {role}")


@dataclass
class AgentSuite:
    config: MVPConfig
    client: LLMClient
    _budget_lock: Lock = field(default_factory=Lock)
    _total_tool_calls: int = 0
    _macro_tool_calls: int = 0
    _company_tool_calls: dict[str, int] = field(default_factory=dict)

    def _reserve_tool_call(self, role: str, symbol: str | None = None) -> None:
        with self._budget_lock:
            if self._total_tool_calls >= self.config.agents.max_total_tool_calls:
                raise StrategyError(
                    "TASK_TOOL_BUDGET_EXHAUSTED",
                    "task retrieval budget exhausted",
                )
            if role == "macro":
                if self._macro_tool_calls >= self.config.agents.macro_max_tool_calls:
                    raise StrategyError(
                        "MACRO_BUDGET_EXHAUSTED",
                        "macro retrieval budget exhausted",
                    )
                self._macro_tool_calls += 1
            elif symbol is not None:
                used = self._company_tool_calls.get(symbol, 0)
                if used >= self.config.agents.company_max_tool_calls_per_symbol:
                    raise StrategyError(
                        "SYMBOL_BUDGET_EXHAUSTED",
                        f"{symbol} retrieval budget exhausted",
                    )
                self._company_tool_calls[symbol] = used + 1
            self._total_tool_calls += 1

    def macro(
        self,
        task: ScreeningTask,
        evidence: list[MacroEvidenceRecord],
        snapshot_id: str,
    ) -> tuple[MacroContext, InvocationResult, list[dict]]:
        self._reserve_tool_call("macro")
        hits = search(
            list(evidence),
            "inflation interest rates oil prices dollar economic release",
            self.config.retrieval,
            scope="macro",
        )
        payload = {
            "task_id": task.task_id,
            "decision_cutoff": task.decision_cutoff,
            "evidence": [hit.evidence.model_dump(mode="json") for hit in hits],
        }
        invocation = self.client.generate("macro_evidence", payload, MacroAgentOutput)
        output = MacroAgentOutput.model_validate(invocation.response)
        selected = [item for item in evidence if item.evidence_id in set(output.evidence_ids)]
        context = MacroContext(
            task_id=task.task_id,
            events=selected,
            evidence_ids=[item.evidence_id for item in selected],
        )
        trace = [
            {
                "scope": "macro",
                "snapshot_id": snapshot_id,
                "query": "inflation interest rates oil prices dollar economic release",
                "algorithm": self.config.retrieval.algorithm,
                "tokenizer_version": self.config.retrieval.tokenizer_version,
                "k1": self.config.retrieval.k1,
                "b": self.config.retrieval.b,
                "top_n": self.config.retrieval.top_n,
                "candidate_count": len(evidence),
                "returned_evidence_ids": [
                    hit.evidence.evidence_id for hit in hits
                ],
                "scores": [hit.score for hit in hits],
            }
        ]
        return context, invocation, trace

    def company(
        self,
        task: ScreeningTask,
        symbol: str,
        evidence: list,
        snapshot_id: str,
    ) -> tuple[CompanyAgentOutput, InvocationResult, list[dict]]:
        self._reserve_tool_call("company", symbol)
        query = "earnings guidance merger regulation financing litigation management product"
        hits = search(
            evidence,
            query,
            self.config.retrieval,
            scope="company",
            symbol=symbol,
        )
        payload = {
            "symbol": symbol,
            "task_id": task.task_id,
            "evidence": [hit.evidence.model_dump(mode="json") for hit in hits],
        }
        invocation = self.client.generate("company_evidence", payload, CompanyAgentOutput)
        output = CompanyAgentOutput.model_validate(invocation.response)
        trace = [
            {
                "scope": "company",
                "symbol": symbol,
                "snapshot_id": snapshot_id,
                "query": query,
                "algorithm": self.config.retrieval.algorithm,
                "tokenizer_version": self.config.retrieval.tokenizer_version,
                "k1": self.config.retrieval.k1,
                "b": self.config.retrieval.b,
                "top_n": self.config.retrieval.top_n,
                "candidate_count": sum(
                    item.scope == "company" and item.symbol == symbol
                    for item in evidence
                ),
                "returned_evidence_ids": [
                    hit.evidence.evidence_id for hit in hits
                ],
                "scores": [hit.score for hit in hits],
            }
        ]
        return output, invocation, trace

    def synthesize(
        self,
        task: ScreeningTask,
        feature: FeatureRecord,
        company: CompanyAgentOutput,
        macro: MacroContext,
    ) -> tuple[SynthesisOutput, InvocationResult]:
        payload = {
            "symbol": feature.symbol,
            "task_id": task.task_id,
            "feature": feature.model_dump(mode="json"),
            "company_claims": [item.model_dump(mode="json") for item in company.claims],
            "events": [item.model_dump(mode="json") for item in company.events],
            "macro_evidence_ids": macro.evidence_ids,
        }
        invocation = self.client.generate("synthesis", payload, SynthesisOutput)
        return SynthesisOutput.model_validate(invocation.response), invocation

    def semantic_verify(
        self,
        card: ResearchCard,
        attempt: int,
        evidence: list[EvidenceRecord],
    ) -> tuple[VerificationReport, InvocationResult]:
        referenced_ids = {
            evidence_id
            for claim in card.claims
            for evidence_id in claim.evidence_ids
        } | {
            evidence_id
            for event in card.events
            for evidence_id in event.evidence_ids
        }
        payload = {
            "symbol": card.symbol,
            "card": card.model_dump(
                mode="json",
                exclude={
                    "deterministic_verification",
                    "semantic_verification",
                    "eligibility",
                },
            ),
            "evidence": [
                item.model_dump(mode="json")
                for item in evidence
                if item.evidence_id in referenced_ids
            ],
            "attempt": attempt,
        }
        invocation = self.client.generate("verifier", payload, VerificationReport)
        return VerificationReport.model_validate(invocation.response), invocation


def empty_company_output() -> CompanyAgentOutput:
    return CompanyAgentOutput(
        events=[],
        claims=[
            Claim(
                claim_id="claim_no_event",
                text="No company event was found in the configured search scope.",
                claim_type=ClaimType.INFERENCE,
                evidence_ids=[],
            )
        ],
    )
