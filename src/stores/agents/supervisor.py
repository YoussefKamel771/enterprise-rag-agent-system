"""
stores/agents/supervisor.py

One cheap LLM call classifies the question, then routes via `Command(goto=...)`.
The classification update and the routing decision are produced together,
since the destination is a direct function of the classification.

Backward-compatible re-exports: eval/evaluators.py and eval/run_supervisor.py
import ROUTE_TABLE and build_classification_chain from this module.
"""

import logging
from typing import Iterable, Optional

from langgraph.graph import END
from langgraph.types import Command

from .prompts import CLASSIFICATION_PROMPT, SUPERVISOR_SYSTEM_PROMPT  # noqa: F401 (re-export)
from .routing import DIRECT_RETRIEVER, ROUTE_TABLE  # noqa: F401 (re-export)
from .schemas import SupervisorClassification
from .state import RAGState

logger = logging.getLogger("uvicorn")


def build_classification_chain(classifier_llm):
    return CLASSIFICATION_PROMPT | classifier_llm.with_structured_output(
        SupervisorClassification, method="function_calling")


def build_supervisor_node(
    classifier_llm,
    default_max_iterations: int = 3,
    allowed_destinations: Optional[Iterable[str]] = None,
):
    """
    allowed_destinations: nodes that actually exist in the graph being built.
        Phase 1 only has `direct_retriever`; the classifier may still say
        "completeness", so we clamp the *jump* to what exists while keeping
        `active_agent` as the INTENDED agent. That keeps the eval metric
        `routing_correct` meaningful in every phase and shows in the run what
        the supervisor wanted. None = no clamping.

    The return annotation is deliberately a bare `Command` (not
    Command[Literal[...]]): LangGraph derives edges from that Literal and
    rejects destinations that are not registered, which would break phase 1.
    """
    chain = build_classification_chain(classifier_llm)
    allowed = frozenset(allowed_destinations) if allowed_destinations is not None else None

    async def supervisor_node(state: RAGState) -> Command:
        try:
            result: SupervisorClassification = await chain.ainvoke({"question": state["question"]})
        except Exception as exc:
            logger.exception("Supervisor classification failed")
            return Command(update={"error": f"classification_failed: {exc}"}, goto=END)

        intended = ROUTE_TABLE.get(result.question_type, DIRECT_RETRIEVER)
        destination = intended
        if allowed is not None and intended not in allowed:
            logger.info("[agent] supervisor wanted %s (not in this phase) -> %s", intended, DIRECT_RETRIEVER)
            destination = DIRECT_RETRIEVER

        return Command(
            update={
                "classification": result.model_dump(),
                "question_type": result.question_type,
                "active_agent": intended,
                "retrieval_attempts": 0,
                "iteration_count": state.get("iteration_count", 0),
                "max_iterations": state.get("max_iterations", default_max_iterations),
            },
            goto=destination,
        )

    return supervisor_node
