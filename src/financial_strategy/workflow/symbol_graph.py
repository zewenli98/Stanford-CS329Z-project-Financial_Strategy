"""Per-symbol research graph with targeted revision routing."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from financial_strategy.agents.roles import AgentSuite
from financial_strategy.schemas import (
    Eligibility,
    Prediction,
    ResearchCard,
    Verdict,
    VerificationIssue,
    VerificationReport,
)
from financial_strategy.verification.deterministic import verify_card
from financial_strategy.workflow.state import SymbolResearchState


def build_symbol_graph(agents: AgentSuite):
    def company_node(state: SymbolResearchState) -> dict:
        output, invocations, traces = agents.company(
            state["task"],
            state["symbol"],
            list(state["snapshot"].company_evidence),
            state["snapshot"].snapshot_id,
        )
        return {
            "company_output": output,
            "attempts": [*state.get("attempts", []), *invocations],
            "retrieval_traces": [*state.get("retrieval_traces", []), *traces],
        }

    def synthesis_node(state: SymbolResearchState) -> dict:
        output, invocation = agents.synthesize(
            state["task"],
            state["feature"],
            state["company_output"],
            state["macro_context"],
        )
        card = ResearchCard(
            symbol=state["symbol"],
            task_id=state["task"].task_id,
            decision_cutoff=state["task"].decision_cutoff,
            target_session=state["task"].target_session,
            features=state["feature"],
            events=state["company_output"].events,
            claims=output.claims,
            prediction=Prediction(
                rationale=output.rationale,
                counterarguments=output.counterarguments,
                uncertainty=output.uncertainty,
            ),
            revision_round=state.get("revision_round", 0),
        )
        return {"card": card, "attempts": [*state.get("attempts", []), invocation]}

    def deterministic_node(state: SymbolResearchState) -> dict:
        evidence = [
            *state["snapshot"].company_evidence,
            *state["snapshot"].macro_evidence,
        ]
        report = verify_card(state["card"], list(evidence), state.get("revision_round", 0))
        card = state["card"].model_copy(deep=True)
        card.deterministic_verification = report
        update: dict = {"card": card, "deterministic_report": report}
        if report.verdict == Verdict.PASS and "initial_card" not in state:
            update["initial_card"] = card.model_copy(deep=True)
        if report.verdict == Verdict.REVISE:
            update["revision_round"] = state.get("revision_round", 0) + 1
        return update

    def route_deterministic(state: SymbolResearchState) -> str:
        report = state["deterministic_report"]
        if report.verdict == Verdict.PASS:
            return "semantic"
        if report.verdict == Verdict.ABSTAIN:
            return "abstain"
        if state.get("revision_round", 0) > state["task"].max_revision_rounds:
            return "abstain"
        codes = {issue.code for issue in report.issues}
        if "TIME_BOUNDARY_VIOLATION" in codes:
            return "abstain"
        if codes & {"MISSING_EVIDENCE", "UNSUPPORTED_EVENT_STATUS"}:
            return "company"
        return "synthesis"

    def semantic_node(state: SymbolResearchState) -> dict:
        evidence = [
            *state["snapshot"].company_evidence,
            *state["snapshot"].macro_evidence,
        ]
        report, invocation = agents.semantic_verify(
            state["card"],
            state.get("revision_round", 0),
            list(evidence),
        )
        card = state["card"].model_copy(deep=True)
        card.semantic_verification = report
        if report.verdict == Verdict.PASS:
            card.eligibility = Eligibility(eligible=True)
        update: dict = {
            "card": card,
            "semantic_report": report,
            "attempts": [*state.get("attempts", []), invocation],
        }
        if report.verdict == Verdict.REVISE:
            update["revision_round"] = state.get("revision_round", 0) + 1
        return update

    def route_semantic(state: SymbolResearchState) -> str:
        report = state["semantic_report"]
        if report.verdict == Verdict.PASS:
            return END
        if report.verdict == Verdict.ABSTAIN:
            return "abstain"
        if state.get("revision_round", 0) > state["task"].max_revision_rounds:
            return "abstain"
        codes = {issue.code for issue in report.issues}
        if "TIME_BOUNDARY_VIOLATION" in codes:
            return "abstain"
        if codes & {"MISSING_EVIDENCE", "UNSUPPORTED_EVENT_STATUS"}:
            return "company"
        if codes & {"BAD_INFERENCE", "WRONG_CITATION"}:
            return "synthesis"
        return "abstain"

    def abstain_node(state: SymbolResearchState) -> dict:
        card = state["card"].model_copy(deep=True)
        reasons = [
            issue.code
            for report in (
                state.get("deterministic_report"),
                state.get("semantic_report"),
            )
            if report is not None
            for issue in report.issues
        ]
        if not reasons:
            reasons = ["REVISION_LIMIT_EXCEEDED"]
        card.eligibility = Eligibility(eligible=False, abstain_reasons=sorted(set(reasons)))
        if card.semantic_verification is None:
            card.semantic_verification = VerificationReport(
                verdict=Verdict.ABSTAIN,
                issues=[
                    VerificationIssue(
                        code=reasons[0],
                        severity="blocking",
                        field="research_card",
                        required_action="abstain",
                    )
                ],
                attempt=state.get("revision_round", 0),
            )
        return {"card": card}

    builder = StateGraph(SymbolResearchState)
    builder.add_node("company", company_node)
    builder.add_node("synthesis", synthesis_node)
    builder.add_node("deterministic", deterministic_node)
    builder.add_node("semantic", semantic_node)
    builder.add_node("abstain", abstain_node)
    builder.add_edge(START, "company")
    builder.add_edge("company", "synthesis")
    builder.add_edge("synthesis", "deterministic")
    builder.add_conditional_edges("deterministic", route_deterministic)
    builder.add_conditional_edges("semantic", route_semantic)
    builder.add_edge("abstain", END)
    return builder.compile()
