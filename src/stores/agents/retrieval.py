"""
stores/agents/retrieval.py

A thin anti-corruption layer between graph nodes and NLPController.

Why it exists: `NLPController.search_vector_db_collection` returns EITHER a
`RetrievalResult` OR plain `False` (no embedding / no hits). The old nodes did
`result.ok` directly, which raises AttributeError on `False`. The gateway
always returns a `RetrievalResult`, with `.error` set on failure, so every node
can use one uniform `if not result.ok` check. It also owns project lookup, so
nodes no longer take a `get_project` callable.
"""

from typing import Awaitable, Callable, List, Optional

from models.db_schemas.supportRag.schemas.dataChunk import RetrievalResult
from stores.vectordb.VectorDBInterface import MetadataFilter

from .config import RetrievalBudget


class RetrievalGateway:
    def __init__(self, nlp_controller, get_project: Callable[[int], Awaitable[object]]):
        self._nlp = nlp_controller
        self._get_project = get_project

    async def search(
        self,
        project_id: int,
        text: str,
        budget: RetrievalBudget,
        *,
        filters: Optional[MetadataFilter] = None,
        exclude_chunk_ids: Optional[List[int]] = None,
    ) -> RetrievalResult:
        project = await self._get_project(project_id)
        raw = await self._nlp.search_vector_db_collection(
            project=project,
            text=text,
            candidate_k=budget.candidate_k,
            top_k=budget.top_k,
            filters=filters,
            exclude_chunk_ids=exclude_chunk_ids,
        )

        if not raw:  # controller signals failure with False
            return RetrievalResult(documents=[], error="retrieval_failed")
        if raw.ok and not raw.documents:
            return RetrievalResult(documents=[], error="no_results")
        return raw
