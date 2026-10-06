"""
stores/agents/config.py

Single, typed home for every tuning knob the agent graph uses.

Before the refactor these were literals scattered across nodes.py
(`candidate_k=20, top_k=5`, `s >= 0.3`, `[:8]`, ...). Centralising them means:
  * a retrieval-quality experiment is a one-line change here, not a hunt
    through six node factories;
  * nodes receive an immutable config object (Dependency Inversion) instead of
    reaching into global settings.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RetrievalBudget:
    """How wide to search and how much to keep.

    candidate_k: size of the fused (dense + lexical, RRF) candidate pool.
    top_k:       how many chunks survive the cross-encoder reranker.
                 None means "do not rerank". NOTE: un-reranked chunks carry
                 raw RRF scores (~0.01-0.03), which are NOT comparable to the
                 reranker's sigmoid scores (0-1) used by the generator's
                 `min_generation_score` filter.
    """

    candidate_k: int
    top_k: Optional[int]


@dataclass(frozen=True)
class AgentConfig:
    # --- graph control -----------------------------------------------------
    graph_phase: int = 1
    max_iterations: int = 3            # verifier -> reformulator loop cap
    recursion_limit: int = 25          # LangGraph hard stop

    # --- completeness sweep loop ------------------------------------------
    completeness_max_docs: int = 12
    completeness_max_attempts: int = 4

    # --- generator context selection ---------------------------------------
    min_generation_score: float = 0.3
    max_generation_chunks: int = 8

    # --- per-agent retrieval budgets ---------------------------------------
    direct: RetrievalBudget = RetrievalBudget(candidate_k=20, top_k=5)
    subquery: RetrievalBudget = RetrievalBudget(candidate_k=20, top_k=6)
    constrained: RetrievalBudget = RetrievalBudget(candidate_k=40, top_k=10)
    # Was top_k=None (no rerank). That made every chunk's RRF score fall below
    # `min_generation_score`, so the generator discarded ALL of them. Reranking
    # to 12 keeps the "wide net" intent while giving comparable scores.
    conflict: RetrievalBudget = RetrievalBudget(candidate_k=60, top_k=12)
    sweep: RetrievalBudget = RetrievalBudget(candidate_k=30, top_k=10)

    @classmethod
    def from_settings(cls, settings) -> "AgentConfig":
        """Build from helpers.config.Settings; budgets keep their defaults."""
        return cls(
            graph_phase=settings.AGENT_GRAPH_PHASE,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            recursion_limit=settings.AGENT_RECURSION_LIMIT,
            completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
            completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
        )
