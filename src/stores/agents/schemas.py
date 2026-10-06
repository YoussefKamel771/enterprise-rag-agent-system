"""
stores/agents/schemas.py

Passed to `chat_model.with_structured_output(Model)` -- LangChain's built-in
structured-output support, not hand-rolled JSON parsing.

Changes vs the original: `ClaimExtraction` / `DraftClaims` (claims + the doc_ids
the *generator* cited inline) are replaced by `ClaimVerdict` / `GroundingVerdict`
(claims judged against the retrieved context by the verifier itself).
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field

QuestionTypeLiteral = Literal[
    "basic", "semantic", "intra_document", "project_related",
    "constrained", "conflicting_info", "completeness",
    "miscellaneous", "high_level", "info_not_found",
]

SourceType = Literal["slack", "gmail", "linear", "google_drive",
                     "hubspot", "fireflies", "github", "jira", "confluence"]


class SupervisorClassification(BaseModel):
    question_type: QuestionTypeLiteral
    estimated_source_types: list[str] = Field(default_factory=list)
    requires_multi_doc: bool
    temporal_sensitivity: bool
    confidence: float = Field(ge=0.0, le=1.0)


class ExtractedFilters(BaseModel):
    source_types: list[SourceType] = Field(default_factory=list)
    created_after: Optional[str] = None
    created_before: Optional[str] = None
    other_equals: dict[str, str] = Field(default_factory=dict)


class SubQuestions(BaseModel):
    sub_queries: list[str]
    facets: list[str] = Field(default_factory=list)


class ClaimVerdict(BaseModel):
    claim: str
    supported: bool = Field(
        description="True only if a provided passage directly states or clearly entails the claim.")
    supporting_doc_ids: list[str] = Field(
        default_factory=list,
        description="doc_ids (from the [doc_id=...] labels) of the passages that support the claim.")


class GroundingVerdict(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)
