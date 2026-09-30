"""
eval/run_agent.py

`build_eval_dependencies` mirrors main.py's lifespan wiring (same factories),
built once per benchmark process. `run_question` is the LangSmith *target*:
it drives one question through the graph and returns a small JSON-safe dict
that the evaluators read.

No custom tracing here: LangSmith records the full span tree (nodes, LLM
calls, retriever, tokens, latency). We only stream node updates to record
WHICH nodes ran, because the evaluators (routing / sweep / filter checks)
need that as data rather than as a trace to browse.
"""

import logging

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from helpers.config import get_settings
from stores.llm import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory
from stores.reranker import RerankerProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from stores.agents import build_rag_graph
from stores.agents.graph import make_initial_state, make_config
from controllers import NLPController
from models import ProjectModel

logger = logging.getLogger("uvicorn.eval")


async def build_eval_dependencies(settings=None) -> dict:
    settings = settings or get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    llm_factory = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(settings, db_client=db_client)
    reranker_factory = RerankerProviderFactory(settings)

    generation_client = llm_factory.create(settings.GENERATION_BACKEND)
    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    embedding_client = llm_factory.create(settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)

    vectordb_client = vectordb_factory.create(settings.VECTOR_DB_BACKEND)
    await vectordb_client.connect()

    reranker_client = reranker_factory.create(settings.RERANKER_BACKEND)
    reranker_client.set_reranker_model(settings.RERANKER_MODEL_ID)

    template_parser = TemplateParser(language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG)

    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        template_parser=template_parser,
    )

    project_model_holder: dict = {}

    async def get_project(project_id: int):
        if "model" not in project_model_holder:
            project_model_holder["model"] = await ProjectModel.create_instance(db_client=db_client)
        return await project_model_holder["model"].get_project_or_create_one(project_id=project_id)

    rag_graph = build_rag_graph(
        settings=settings,
        nlp_controller=nlp_controller,
        generation_client=generation_client,
        template_parser=template_parser,
        get_project=get_project,
    )

    return {"settings": settings, "engine": engine, "rag_graph": rag_graph}


def _summarize_step(node: str, update) -> dict:
    """Compact, JSON-safe record of one node execution -- only the fields
    the evaluators need. Full detail lives in the LangSmith trace."""
    step = {"node": node}
    if isinstance(update, dict):
        if "retrieved_chunks" in update:
            step["n_chunks"] = len(update["retrieved_chunks"] or [])
        if "retrieval_filters" in update:
            step["retrieval_filters"] = update["retrieval_filters"]
        if update.get("draft_answer"):
            step["draft_answer"] = update["draft_answer"]
        if "grounding_passed" in update:
            step["grounding_passed"] = update["grounding_passed"]
    return step


def _doc_dict(d) -> dict:
    return {
        "doc_id": d.doc_id,
        "chunk_id": d.chunk_id,
        "source_type": d.source_type,
        "score": round(float(d.score), 4),
        "metadata": d.metadata,
    }


async def run_question(deps: dict, inputs: dict, project_id: int, phase: int, run_tag: str) -> dict:
    """LangSmith target. Exceptions are NOT swallowed: a crashed run shows up
    as an errored run in the experiment (and is skipped by evaluators)
    instead of being scored as a wrong answer."""
    settings = deps["settings"]
    graph = deps["rag_graph"]
    question_id = inputs["question_id"]

    thread_id = f"eval-{run_tag}-{question_id}"
    config = make_config(
        thread_id,
        recursion_limit=settings.AGENT_RECURSION_LIMIT,
        run_name=f"rag_agent_phase{phase}",
        tags=[f"phase:{phase}", f"run_tag:{run_tag}"],
        metadata={"question_id": question_id, "phase": phase},
    )
    state = make_initial_state(inputs["question"], project_id, settings.AGENT_MAX_ITERATIONS)

    steps: list[dict] = []
    async for chunk in graph.astream(state, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            steps.append(_summarize_step(node, update))

    steps: list[dict] = []
    final: dict = {}
    async for mode, payload in graph.astream(
        state, config=config, stream_mode=["updates", "values"]
    ):
        if mode == "updates":
            for node, update in payload.items():
                steps.append(_summarize_step(node, update))
        else:                      # "values": full state after each step; last one wins
            final = payload

    # A supervisor/graph failure is an infra error, not a wrong answer:
    # raise so LangSmith marks the run errored and evaluators skip it.
    if final.get("error") and not (final.get("final_answer") or final.get("draft_answer")):
        raise RuntimeError(final["error"])

    return {
        "answer": final.get("final_answer") or final.get("draft_answer") or "",
        "draft_answers": [s["draft_answer"] for s in steps if s.get("draft_answer")],
        "citations": final.get("citations", []),
        "question_type": final.get("question_type"),
        "active_agent": final.get("active_agent"),
        "iteration_count": final.get("iteration_count", 0),
        "max_iterations": final.get("max_iterations", settings.AGENT_MAX_ITERATIONS),
        "retrieval_attempts": final.get("retrieval_attempts", 0),
        "retrieved_docs": [_doc_dict(d) for d in final.get("retrieved_chunks", [])],
        "steps": steps,
        "error": final.get("error"),
    }