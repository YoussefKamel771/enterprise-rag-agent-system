"""
stores/agents/runtime.py

Helpers for *running* a compiled graph (initial state, run config, invoke).
Separate from graph.py, which only knows how to *build* one.
"""

import logging
import time

from langchain_core.messages import HumanMessage

from .state import RAGState, per_turn_defaults

logger = logging.getLogger("uvicorn")


def make_initial_state(question, project_id, max_iterations=3, user_id=None) -> RAGState:
    """State for ONE turn. On a persisted thread this is merged into the
    checkpoint: per-turn channels are reset, `messages` gets the new human
    message appended, `conversation_summary` is left untouched."""
    return {
        **per_turn_defaults(),
        "question": question,
        "original_question": question,
        "project_id": project_id,
        "user_id": user_id,
        "max_iterations": max_iterations,
        "messages": [HumanMessage(content=question)],
    }


def make_config(thread_id, recursion_limit=25, run_name="rag_agent", tags=None, metadata=None):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit,
            "run_name": run_name, "tags": tags or [], "metadata": metadata or {}}


async def run_rag_graph(app_graph, question: str, project_id: int, thread_id: str,
                        max_iterations: int = 3, recursion_limit: int = 25,
                        user_id: str = None):
    initial_state = make_initial_state(question, project_id, max_iterations, user_id=user_id)
    config = make_config(thread_id, recursion_limit)

    logger.info("[agent] RUN start thread=%s project=%s q=%r", thread_id, project_id, question)
    t0 = time.perf_counter()

    result = await app_graph.ainvoke(initial_state, config=config)

    logger.info("[agent] RUN end (%.0fs) type=%s agent=%s iters=%s chunks=%d error=%s",
                time.perf_counter() - t0, result.get("question_type"),
                result.get("active_agent"), result.get("iteration_count"),
                len(result.get("retrieved_chunks", [])), result.get("error"))
    return result