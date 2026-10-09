"""
stores/agents/state.py

Central state schema for the agent graph. See graph.py for how the reducers
here (Annotated[..., operator.add]) interact with parallel Send fan-out.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
 
from models.db_schemas.supportRag.schemas.dataChunk import RetrievedDocument

QuestionType = Literal[
    "basic", "semantic", "intra_document", "project_related",
    "constrained", "conflicting_info", "completeness",
    "miscellaneous", "high_level", "info_not_found",
]

AgentName = Literal[
    "direct_retriever", "decomposer", "constraint_filter",
    "conflict_resolver", "completeness_sweep", "broad_synthesis",
]

RESET = "__reset__"

def append_or_reset(old, new):
    """Append-only reducer (parallel Send branches + sweep loops still
    accumulate) that a turn can wipe by sending RESET."""
    if isinstance(new, str) and new == RESET:
        return []
    return list(old or []) + list(new or [])

class ClassificationResult(TypedDict):
    question_type: QuestionType
    estimated_source_types: list[str]
    requires_multi_doc: bool
    temporal_sensitivity: bool
    confidence: float


class VerifiedClaim(TypedDict):
    claim: str
    supported: bool
    supporting_doc_ids: list[str]
    chunk_ids: list[int]


class RAGState(TypedDict, total=False):
    # input
    question: str                     # working question (rewritten / reformulated)
    original_question: str            # exactly what the user typed this turn
    project_id: int
    user_id: Optional[str]
    
    # short-term memory (persisted per thread)
    messages: Annotated[list[AnyMessage], add_messages]
    conversation_summary: Optional[str]
    standalone_question: Optional[str]
    
    # long-term memory (read-only inside the graph)
    recalled_memories: list[dict]

    # routing
    classification: Optional[ClassificationResult]
    question_type: Optional[QuestionType]
    active_agent: Optional[AgentName]

    # retrieval -- append-only reducers, see graph.py
    retrieved_chunks: Annotated[list[RetrievedDocument], operator.add]
    excluded_chunk_ids: Annotated[list[int], operator.add]
    retrieval_filters: Optional[dict]
    sub_queries: list[str]
    retrieval_attempts: int

    # generation
    draft_answer: Optional[str]
    facets: list[str]
    covered_facets: list[str]

    # verification
    verified_claims: list[VerifiedClaim]
    grounding_passed: bool
    unsupported_claims: list[str]

    # control / loop-back
    iteration_count: int
    max_iterations: int

    # output
    final_answer: Optional[str]
    citations: list[str]
    error: Optional[str]
    

def per_turn_defaults() -> dict:
    """Fresh values for every channel that must NOT survive into the next turn.
    A function (not a module dict) so mutable defaults are never shared."""
    return {
        "classification": None,
        "question_type": None,
        "active_agent": None,
        "standalone_question": None,
        "recalled_memories": [],
        "retrieved_chunks": RESET,
        "excluded_chunk_ids": RESET,
        "retrieval_filters": None,
        "sub_queries": [],
        "retrieval_attempts": 0,
        "draft_answer": None,
        "facets": [],
        "covered_facets": [],
        "verified_claims": [],
        "grounding_passed": False,
        "unsupported_claims": [],
        "iteration_count": 0,
        "final_answer": None,
        "citations": [],
        "error": None,
    }
 
 
PER_TURN_KEYS = frozenset(per_turn_defaults())
PERSISTENT_KEYS = frozenset({"messages", "conversation_summary"})
INPUT_KEYS = frozenset({"question", "original_question", "project_id", "user_id", "max_iterations"})