"""
stores/agents/nodes/retrieval.py

Nodes whose job is to fetch chunks. Each factory takes the RetrievalGateway and
AgentConfig (injected from graph.py) and returns an async node.

All nodes return the same two-key retrieval update via `_as_update`; the
reducers in state.py (operator.add) append it, so parallel Send branches and
sweep loops accumulate rather than overwrite.
"""

import logging

from stores.vectordb.VectorDBInterface import MetadataFilter

from ..config import AgentConfig
from ..prompts import FILTER_EXTRACTION_PROMPT
from ..retrieval import RetrievalGateway
from ..schemas import ExtractedFilters
from ..state import RAGState

logger = logging.getLogger("uvicorn")


def _as_update(docs) -> dict:
    """Retrieval update: new chunks + their ids (so sweeps can exclude them)."""
    return {
        "retrieved_chunks": docs,
        "excluded_chunk_ids": [d.chunk_id for d in docs],
    }


def build_direct_retriever_node(gateway: RetrievalGateway, config: AgentConfig):
    async def direct_retriever_node(state: RAGState) -> dict:
        result = await gateway.search(state["project_id"], state["question"], config.direct)
        if not result.ok:
            return {"error": result.error or "retrieval_failed"}
        return {
            **_as_update(result.documents),
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return direct_retriever_node


def build_retrieve_subquery_node(gateway: RetrievalGateway, config: AgentConfig):
    """Worker for Send fan-out (decomposer / broad_synthesis)."""
    async def retrieve_subquery_node(state: RAGState) -> dict:
        result = await gateway.search(state["project_id"], state["question"], config.subquery)
        if not result.ok:
            logger.warning("[agent] retrieve_subquery failed q=%r err=%s", state["question"], result.error)
            return {}
        return _as_update(result.documents)
    return retrieve_subquery_node


def build_constraint_filter_node(filter_llm, gateway: RetrievalGateway, config: AgentConfig):
    chain = FILTER_EXTRACTION_PROMPT | filter_llm.with_structured_output(
        ExtractedFilters, method="function_calling")

    async def constraint_filter_node(state: RAGState) -> dict:
        try:
            extracted = await chain.ainvoke({"question": state["question"]})
        except Exception:
            logger.exception("[agent] constraint_filter extraction failed, using no filters")
            extracted = ExtractedFilters()

        filters = MetadataFilter(
            equals=extracted.other_equals,
            in_={"source_type": extracted.source_types} if extracted.source_types else {},
        )
        result = await gateway.search(
            state["project_id"], state["question"], config.constrained, filters=filters)
        if result.ok:
            return {"retrieval_filters": extracted.model_dump(), **_as_update(result.documents)}

        # Filtered pass came up empty: fall back to the unfiltered search.
        logger.info("[agent] constraint_filter: filtered pass empty, filters=%s -> falling back",
                    extracted.model_dump())
        fallback = await gateway.search(state["project_id"], state["question"], config.constrained)
        return {
            "retrieval_filters": {**extracted.model_dump(), "fallback_unfiltered": True},
            **_as_update(fallback.documents if fallback.ok else []),
        }
    return constraint_filter_node


def build_conflict_resolver_node(gateway: RetrievalGateway, config: AgentConfig):
    """Wide net so both the old and the new version of a fact are retrieved."""
    async def conflict_resolver_node(state: RAGState) -> dict:
        result = await gateway.search(state["project_id"], state["question"], config.conflict)
        if not result.ok:
            return {"error": result.error or "retrieval_failed"}
        return _as_update(result.documents)
    return conflict_resolver_node


def build_completeness_sweep_node(gateway: RetrievalGateway, config: AgentConfig):
    """Loops (via route_after_completeness_sweep) pulling chunks not seen yet."""
    async def completeness_sweep_node(state: RAGState) -> dict:
        excluded = state.get("excluded_chunk_ids", [])
        result = await gateway.search(
            state["project_id"], state["question"], config.sweep, exclude_chunk_ids=excluded)
        if not result.ok:
            logger.warning("[agent] completeness_sweep failed err=%s", result.error)

        new_docs = result.documents if result.ok else []
        logger.info("[agent] completeness_sweep: excluded=%d new=%d", len(excluded), len(new_docs))
        return {
            **_as_update(new_docs),
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return completeness_sweep_node
