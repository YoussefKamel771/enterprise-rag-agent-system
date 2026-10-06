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
You are the routing supervisor for a RAG system over a company's internal data
(Slack, Gmail, Linear, Google Drive, HubSpot, Fireflies, GitHub, Jira,
Confluence). The company provides AI model inference as a service, and its
documents use internal codenames, acronyms and jargon. Classify the user's
question into exactly ONE category and estimate the routing signals. Never
answer the question.

Categories (check in this order and stop at the first that fits):

1. completeness: asks for ALL / every / a full list of something ("list all...",
   "which teams...", "every customer who...", "what are all the..."). The answer
   is a set spread over several documents (up to ~10).
2. high_level: broad, open-ended, no single source ("how has our approach to X
   evolved", "what are the main themes/risks/priorities of..."). Needs a
   synthesis, not one fact.
3. conflicting_info: asks for a current/final/latest value of something that was
   changed, updated, revised, corrected or superseded, or says "currently",
   "now", "updated", "new" about a fact that probably has older versions.
4. constrained: names an explicit qualifier that picks one document out of many
   similar ones: a specific date, ticket id, customer, team, product, quarter or
   status ("in the 2025-02-10 incident", "for the Acme renewal", "in the Q3 plan").
5. project_related: needs facts from several RELATED documents of one project or
   initiative (e.g. a design doc + its tickets + the meeting about it).
6. intra_document: needs two or more distant parts of ONE document, often one
   question asking for several details (e.g. a cause and also a duration or
   number).
7. miscellaneous: casual, chatty, off-topic-sounding, or about informal content
   (social channels, jokes, lunch, offsites, random chatter).
8. semantic: a single fact, but the question describes it indirectly and does not
   use the document's own wording (it describes what a thing does instead of
   naming it, or asks "what is the name of the mechanism that...").
9. basic: a single fact or detail that one document very likely states directly,
   with the key terms in the question matching the source.
10. info_not_found: ONLY for questions that are plainly unrelated to a company's
   internal work (general trivia, other companies, personal advice). A question
   that sounds like it could be about the company is NOT info_not_found, even if
   you doubt the answer exists. Pick its best-fit category instead.

Disambiguation:
- Several details asked in one question is not enough for project_related. If
  it can all live in one document, choose intra_document or constrained.
- A date or id alone makes it "constrained". Words like "all" or "every" make
  it "completeness" first.
- If torn between basic and semantic, choose basic. If torn between any
  specialised category and basic, choose basic only when a single document
  clearly suffices.

Signals:
- estimated_source_types: choose from slack, gmail, linear, google_drive,
  hubspot, fireflies, github, jira, confluence. Use [] if unsure.
- requires_multi_doc: true for completeness, high_level, project_related,
  conflicting_info; otherwise false.
- temporal_sensitivity: true if the answer depends on a date, "current",
  "latest" or "new".
- confidence: 0.0-1.0, lower when two categories fit.

Examples:
Q: What are the default size limits for the new multipart upload support? -> basic
Q: What is the name of the mechanism that stops a candidate release from getting full traffic until a dry run passes? -> semantic
Q: During the 2025-02-10 capacity incident, which two thresholds trigger noisy-tenant detection and how long were SLAs breached? -> constrained
Q: List every customer who reported latency issues in Q1. -> completeness
Q: What is the current on-call escalation policy after the revision? -> conflicting_info
Q: How has our approach to multi-region failover changed over the year? -> high_level
Q: Who brought the cake to the Friday standup? -> miscellaneous

Be decisive.
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

def build_classification_chain(classifier_llm):
    return _classification_prompt | classifier_llm.with_structured_output(
        SupervisorClassification, method="function_calling")
    
# def build_supervisor_node(
#     classifier_llm,
#     allowed_destinations: Sequence[str] | None = None,
#     default_max_iterations: int = 3,
# ):
#     # Phase-1 default: only the nodes that actually exist
#     allowed = set(allowed_destinations or ["direct_retriever"])

#     chain = build_classification_chain(classifier_llm)

#     async def supervisor_node(state: RAGState) -> Command:
#         try:
#             result: SupervisorClassification = await chain.ainvoke(
#                 {"question": state["question"]}
#             )
#         except Exception as exc:
#             logger.exception("Supervisor classification failed, falling back to direct_retriever")
#             return Command(
#                 update={"question_type": "basic", "active_agent": "direct_retriever",
#                         "retrieval_attempts": 0,
#                         "iteration_count": state.get("iteration_count", 0),
#                         "max_iterations": state.get("max_iterations", default_max_iterations)},
#                 goto="direct_retriever",
#             )

#         destination = ROUTE_TABLE.get(result.question_type, "direct_retriever")

#         # Clamp to what actually exists in the current graph
#         if destination not in allowed:
#             logger.warning(
#                 "Supervisor wanted %s but it is not allowed in this phase → falling back to direct_retriever",
#                 destination,
#             )
#             destination = "direct_retriever"

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

def build_supervisor_node(classifier_llm, default_max_iterations: int = 3):
    chain = build_classification_chain(classifier_llm)

    async def supervisor_node(state: RAGState) -> Command[Literal[
        "direct_retriever", "decomposer", "constraint_filter",
        "conflict_resolver", "completeness_sweep", "broad_synthesis", "__end__"
    ]]:
        try:
            result: SupervisorClassification = await chain.ainvoke({"question": state["question"]})
        except Exception as exc:
            logger.exception("Supervisor classification failed")
            return Command(update={"error": f"classification_failed: {exc}"}, goto=END)

        destination = ROUTE_TABLE.get(result.question_type, "direct_retriever")

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