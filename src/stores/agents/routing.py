"""
stores/agents/routing.py

Everything about *where the graph goes next*: node-name constants, the
question_type -> agent table, and the conditional-edge routing functions.

Node names used to be bare string literals repeated across supervisor.py,
nodes.py and graph.py; a typo in any of them only failed at graph-compile time.
"""

from typing import Literal

from .state import AgentName, RAGState

# --------------------------------------------------------------------------
# Node names
# --------------------------------------------------------------------------
SUPERVISOR = "supervisor"
DIRECT_RETRIEVER = "direct_retriever"
DECOMPOSER = "decomposer"
RETRIEVE_SUBQUERY = "retrieve_subquery"
CONSTRAINT_FILTER = "constraint_filter"
CONFLICT_RESOLVER = "conflict_resolver"
COMPLETENESS_SWEEP = "completeness_sweep"
BROAD_SYNTHESIS = "broad_synthesis"
GENERATOR = "generator"
VERIFIER = "verifier"
REFORMULATOR = "reformulator"
INFO_NOT_FOUND = "info_not_found"
FINALIZE = "finalize"

LOAD_CONTEXT = "load_context"
CONTEXTUALIZE = "contextualize"
RECALL_MEMORY = "recall_memory"
COMMIT_TURN = "commit_turn"

# Destinations the supervisor may route to, grouped by the phase that adds them.
CORE_DESTINATIONS = frozenset({DIRECT_RETRIEVER})
SPECIALIST_DESTINATIONS = frozenset({
    DECOMPOSER, CONSTRAINT_FILTER, CONFLICT_RESOLVER, COMPLETENESS_SWEEP, BROAD_SYNTHESIS,
})

# Centralized question_type -> agent-node mapping. Adding a category later is
# a one-line change here, not a rewritten conditional.
ROUTE_TABLE: dict[str, AgentName] = {
    "basic": DIRECT_RETRIEVER,
    "semantic": DIRECT_RETRIEVER,
    "miscellaneous": DIRECT_RETRIEVER,
    "intra_document": DECOMPOSER,
    "project_related": DECOMPOSER,
    "constrained": CONSTRAINT_FILTER,
    "conflicting_info": CONFLICT_RESOLVER,
    "completeness": COMPLETENESS_SWEEP,
    "high_level": BROAD_SYNTHESIS,
    "info_not_found": DIRECT_RETRIEVER,
}


# --------------------------------------------------------------------------
# Conditional-edge routers
# --------------------------------------------------------------------------

def build_route_after_completeness_sweep(max_docs: int, max_attempts: int):
    """Factory so the thresholds come from AgentConfig at graph-build time
    rather than being hardcoded constants."""

    def route_after_completeness_sweep(state: RAGState) -> Literal["completeness_sweep", "generator"]:
        total = len(state.get("retrieved_chunks", []))
        attempts = state.get("retrieval_attempts", 0)
        if total >= max_docs or attempts >= max_attempts:
            return GENERATOR
        return COMPLETENESS_SWEEP

    return route_after_completeness_sweep


def route_after_verifier(state: RAGState) -> Literal["reformulator", "finalize", "info_not_found"]:
    if state.get("grounding_passed"):
        return FINALIZE
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return INFO_NOT_FOUND
    return REFORMULATOR
