"""
helpers/observability.py

The only tracing code left in the repo. LangGraph nodes and LangChain chains
(supervisor, decomposer, verifier, judge, ...) are traced by LangSmith
automatically once LANGSMITH_TRACING=true is in the environment. What is NOT
traced automatically is our own retrieval code, so it gets one decorator that
renders it as a proper "retriever" span (documents + scores show up in the
LangSmith UI).

Usage:
    from helpers.observability import traced_retriever

    @traced_retriever
    async def search_vector_db_collection(self, project, text, ...): ...
"""

from langsmith import traceable


def _retrieval_inputs(inputs: dict) -> dict:
    # Keep the span input readable: drop `self` and the SQLAlchemy Project object.
    project = inputs.get("project")
    filters = inputs.get("filters")
    return {
        "query": inputs.get("text"),
        "project_id": getattr(project, "project_id", None),
        "candidate_k": inputs.get("candidate_k"),
        "top_k": inputs.get("top_k"),
        "filters": repr(filters) if filters else None,
        "n_excluded_chunks": len(inputs.get("exclude_chunk_ids") or []),
    }


def _retrieval_outputs(result) -> dict:
    # RetrievalResult (pgvector) or a plain list (qdrant) or False (failure)
    docs = getattr(result, "documents", result if isinstance(result, list) else None)
    if not docs:
        return {"documents": [], "error": getattr(result, "error", None) or "no_results"}
    return {
        "documents": [
            {
                "page_content": d.text,
                "type": "Document",
                "metadata": {
                    "doc_id": d.doc_id,
                    "chunk_id": d.chunk_id,
                    "source_type": d.source_type,
                    "score": float(d.score),
                },
            }
            for d in docs
        ]
    }


traced_retriever = traceable(
    run_type="retriever",
    name="hybrid_search_and_rerank",
    process_inputs=_retrieval_inputs,
    process_outputs=_retrieval_outputs,
)