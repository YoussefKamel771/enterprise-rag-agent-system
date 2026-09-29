"""
stores/agents/state.py

Central state schema for the agent graph. See graph.py for how the reducers
here (Annotated[..., operator.add]) interact with parallel Send fan-out.
"""

import operator
from typing import Annotated, Literal, Optional, TypedDict

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
    question: str
    project_id: int

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