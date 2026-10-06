"""
stores/agents/runtime.py

Helpers for *running* a compiled graph (initial state, run config, invoke).
Separate from graph.py, which only knows how to *build* one.
"""

import logging
import time

from .state import RAGState

logger = logging.getLogger("uvicorn")


def make_initial_state(question, project_id, max_iterations=3) -> RAGState:
    return {"question": question, "project_id": project_id, "retrieved_chunks": [],
            "excluded_chunk_ids": [], "sub_queries": [], "verified_claims": [],
            "unsupported_claims": [], "iteration_count": 0, "max_iterations": max_iterations}


def make_config(thread_id, recursion_limit=25, run_name="rag_agent", tags=None, metadata=None):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit,
            "run_name": run_name, "tags": tags or [], "metadata": metadata or {}}


async def run_rag_graph(app_graph, question: str, project_id: int, thread_id: str,
                        max_iterations: int = 3, recursion_limit: int = 25):
    initial_state = make_initial_state(question, project_id, max_iterations)
    config = make_config(thread_id, recursion_limit)

    logger.info("[agent] RUN start thread=%s project=%s q=%r", thread_id, project_id, question)
    t0 = time.perf_counter()

    result = await app_graph.ainvoke(initial_state, config=config)

    logger.info("[agent] RUN end (%.0fs) type=%s agent=%s iters=%s chunks=%d error=%s",
                time.perf_counter() - t0, result.get("question_type"),
                result.get("active_agent"), result.get("iteration_count"),
                len(result.get("retrieved_chunks", [])), result.get("error"))
    return result
