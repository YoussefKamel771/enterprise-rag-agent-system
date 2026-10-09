"""
stores/agents/config.py

Single, typed home for every tuning knob the agent graph uses.
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
    # Was top_k=None (no rerank) -> every RRF score fell below
    # `min_generation_score` and the generator discarded ALL chunks.
    conflict: RetrievalBudget = RetrievalBudget(candidate_k=60, top_k=12)
    sweep: RetrievalBudget = RetrievalBudget(candidate_k=30, top_k=10)

    # --- memory ------------------------------------------------------------
    memory_enabled: bool = False          # short-term memory layer
    history_max_turns: int = 6            # turns kept verbatim (1 turn = user + assistant)
    summary_trigger_tokens: int = 1500    # summarize older turns past this
    generator_sees_history: bool = False  # ablation flag; default: rewrite carries context
    ltm_enabled: bool = False             # long-term memory (requires memory_enabled)
    ltm_top_k: int = 4
    ltm_min_score: float = 0.35

    @classmethod
    def from_settings(cls, settings) -> "AgentConfig":
        """Build from helpers.config.Settings; budgets keep their defaults."""
        if settings.AGENT_LTM_ENABLED and not settings.AGENT_MEMORY_ENABLED:
            raise ValueError("AGENT_LTM_ENABLED requires AGENT_MEMORY_ENABLED "
                             "(recall uses the standalone question produced by short-term memory)")
        return cls(
            graph_phase=settings.AGENT_GRAPH_PHASE,
            max_iterations=settings.AGENT_MAX_ITERATIONS,
            recursion_limit=settings.AGENT_RECURSION_LIMIT,
            completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
            completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
            memory_enabled=settings.AGENT_MEMORY_ENABLED,
            history_max_turns=max(1, settings.AGENT_HISTORY_MAX_TURNS),
            summary_trigger_tokens=settings.AGENT_SUMMARY_TRIGGER_TOKENS,
            generator_sees_history=settings.AGENT_GENERATOR_SEES_HISTORY,
            ltm_enabled=settings.AGENT_LTM_ENABLED,
            ltm_top_k=settings.AGENT_LTM_TOP_K,
            ltm_min_score=settings.AGENT_LTM_MIN_SCORE,
        )