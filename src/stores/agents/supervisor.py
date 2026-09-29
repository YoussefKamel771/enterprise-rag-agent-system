"""
stores/agents/supervisor.py

One cheap/fast LLM call classifies the question, then routes via
`Command(goto=...)` -- the update (classification result) and the routing
decision are produced together, since the destination is a direct function
of the classification.
"""

import logging
from typing import Literal, Sequence

from langgraph.graph import END
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate

from .schemas import SupervisorClassification
from .state import AgentName, RAGState

logger = logging.getLogger("uvicorn")

SUPERVISOR_SYSTEM_PROMPT = """\
You are the routing supervisor for an enterprise document Q&A system.
Classify the user's question into exactly one category, and estimate a few
routing signals. Do not attempt to answer the question.

Categories:
- basic: simple question, single clear ground-truth document likely answers it.
- semantic: same as basic but phrased indirectly / low keyword overlap with likely source text.
- intra_document: answer requires combining multiple sections of ONE document.
- project_related: answer requires aggregating knowledge across several RELATED documents.
- constrained: multiple documents could be relevant, but explicit qualifiers (date, team,
  product, status) narrow it to one.
- conflicting_info: question topic is one where documents are likely to disagree or contain
  updated/superseded facts.
- completeness: question asks for an exhaustive list/set -- answer requires gathering
  potentially many documents.
- high_level: broad question with no single ground-truth document; needs synthesis across
  many sources.
- miscellaneous: informal, off-topic, or targets loosely-organized documents.
- info_not_found: clearly out of scope for an internal company knowledge base.

Be decisive. Prefer "basic" when uncertain between basic/semantic.
"""

_classification_prompt = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_SYSTEM_PROMPT),
    ("human", "{question}"),
])

# Centralized question_type -> agent-node mapping. Adding a category later is
# a one-line change here, not a rewritten conditional.
ROUTE_TABLE: dict[str, AgentName] = {
    "basic": "direct_retriever",
    "semantic": "direct_retriever",
    "miscellaneous": "direct_retriever",
    "intra_document": "decomposer",
    "project_related": "decomposer",
    "constrained": "constraint_filter",
    "conflicting_info": "conflict_resolver",
    "completeness": "completeness_sweep",
    "high_level": "broad_synthesis",
    "info_not_found": "direct_retriever",
}

def build_supervisor_node(
    classifier_llm,
    allowed_destinations: Sequence[str] | None = None,
    default_max_iterations: int = 3,
):
    # Phase-1 default: only the nodes that actually exist
    allowed = set(allowed_destinations or ["direct_retriever"])

    structured_llm = classifier_llm.with_structured_output(SupervisorClassification)
    chain = _classification_prompt | structured_llm

    async def supervisor_node(state: RAGState) -> Command:
        try:
            result: SupervisorClassification = await chain.ainvoke(
                {"question": state["question"]}
            )
        except Exception as exc:
            logger.exception("Supervisor classification failed")
            return Command(update={"error": f"classification_failed: {exc}"}, goto=END)

        destination = ROUTE_TABLE.get(result.question_type, "direct_retriever")

        # Clamp to what actually exists in the current graph
        if destination not in allowed:
            logger.warning(
                "Supervisor wanted %s but it is not allowed in this phase → falling back to direct_retriever",
                destination,
            )
            destination = "direct_retriever"

        return Command(
            update={
                "classification": result.model_dump(),
                "question_type": result.question_type,
                "active_agent": destination,
                "retrieval_attempts": 0,
                "iteration_count": state.get("iteration_count", 0),
                "max_iterations": state.get("max_iterations", default_max_iterations),
            },
            goto=destination,
        )

    return supervisor_node

# def build_supervisor_node(classifier_llm, default_max_iterations: int = 3):
#     structured_llm = classifier_llm.with_structured_output(SupervisorClassification)
#     chain = _classification_prompt | structured_llm

#     async def supervisor_node(state: RAGState) -> Command[Literal[
#         "direct_retriever", "decomposer", "constraint_filter",
#         "conflict_resolver", "completeness_sweep", "broad_synthesis", "__end__"
#     ]]:
#         try:
#             result: SupervisorClassification = await chain.ainvoke({"question": state["question"]})
#         except Exception as exc:
#             logger.exception("Supervisor classification failed")
#             return Command(update={"error": f"classification_failed: {exc}"}, goto=END)

#         destination = ROUTE_TABLE.get(result.question_type, "direct_retriever")

#         return Command(
#             update={
#                 "classification": result.model_dump(),
#                 "question_type": result.question_type,
#                 "active_agent": destination,
#                 "retrieval_attempts": 0,
#                 "iteration_count": state.get("iteration_count", 0),
#                 "max_iterations": state.get("max_iterations", default_max_iterations),
#             },
#             goto=destination,
#         )

#     return supervisor_node