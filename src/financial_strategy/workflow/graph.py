"""Top-level orchestrator with one macro task and dynamic per-symbol workers."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from financial_strategy.agents.roles import AgentSuite
from financial_strategy.config import MVPConfig
from financial_strategy.errors import StrategyError
from financial_strategy.ranking.ranker import rank_cards
from financial_strategy.schemas import (
    Eligibility,
    Prediction,
    ResearchCard,
    Verdict,
    VerificationIssue,
    VerificationReport,
)
from financial_strategy.workflow.state import ScreeningState
from financial_strategy.workflow.symbol_graph import build_symbol_graph


def build_screening_graph(config: MVPConfig, agents: AgentSuite):
    symbol_graph = build_symbol_graph(agents)

    def macro_node(state: ScreeningState) -> dict:
        context, invocation, traces = agents.macro(
            state["task"],
            list(state["snapshot"].macro_evidence),
            state["snapshot"].snapshot_id,
        )
        return {
            "macro_context": context,
            "attempts": [invocation],
            "retrieval_traces": traces,
        }

    def dispatch(state: ScreeningState) -> list[Send] | str:
        if not state["candidates"]:
            return "rank"
        return [
            Send(
                "research_symbol",
                {
                    "task": state["task"],
                    "symbol": symbol,
                    "feature": state["features"][symbol],
                    "snapshot": state["snapshot"],
                    "macro_context": state["macro_context"],
                    "revision_round": 0,
                    "attempts": [],
                    "retrieval_traces": [],
                },
            )
            for symbol in state["candidates"]
        ]

    def research_symbol(state: dict) -> dict:
        try:
            result = symbol_graph.invoke(state)
        except StrategyError as exc:
            if exc.code.startswith("TASK_"):
                raise
            report = VerificationReport(
                verdict=Verdict.ABSTAIN,
                issues=[
                    VerificationIssue(
                        code=exc.code,
                        severity="blocking",
                        field="agent",
                        required_action="abstain",
                    )
                ],
            )
            card = ResearchCard(
                symbol=state["symbol"],
                task_id=state["task"].task_id,
                decision_cutoff=state["task"].decision_cutoff,
                target_session=state["task"].target_session,
                features=state["feature"],
                prediction=Prediction(
                    uncertainty=["Agent execution failed within the bounded budget."]
                ),
                deterministic_verification=report,
                semantic_verification=report,
                eligibility=Eligibility(eligible=False, abstain_reasons=[exc.code]),
            )
            return {
                "cards": [card],
                "initial_cards": [card.model_copy(deep=True)],
                "attempts": [],
                "retrieval_traces": [],
            }
        initial = result.get("initial_card", result["card"]).model_copy(deep=True)
        return {
            "cards": [result["card"]],
            "initial_cards": [initial],
            "attempts": result.get("attempts", []),
            "retrieval_traces": result.get("retrieval_traces", []),
        }

    def rank_node(state: ScreeningState) -> dict:
        exposures = {item.symbol: item for item in state["snapshot"].exposures}
        watchlist = rank_cards(
            state["cards"],
            state["volatility"],
            state["anomaly"],
            state["macro_context"],
            exposures,
            config.ranking,
            state["task"].top_k,
        )
        return {"watchlist": watchlist}

    builder = StateGraph(ScreeningState)
    builder.add_node("macro", macro_node)
    builder.add_node("research_symbol", research_symbol)
    builder.add_node("rank", rank_node)
    builder.add_edge(START, "macro")
    builder.add_conditional_edges("macro", dispatch, ["research_symbol", "rank"])
    builder.add_edge("research_symbol", "rank")
    builder.add_edge("rank", END)
    return builder.compile()
