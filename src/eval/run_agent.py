"""
eval/run_agent.py -- Section 6.2

`build_eval_dependencies` mirrors main.py's lifespan client construction
(same factories, same shape) so the benchmark exercises the exact same
provider wiring the FastAPI app uses -- built once per benchmark process,
reused across every question. `run_one` drives a single question through
the graph inside a tracing `run_context`, then immediately judges the
result; run_benchmark.py wraps this in a concurrency-limited gather.
"""

import logging
import time

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from helpers.config import get_settings
from stores.llm import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory
from stores.reranker import RerankerProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from stores.agents import build_rag_graph, run_rag_graph
from stores.agents.tracing import run_context, configure_tracing
from controllers import NLPController
from models import ProjectModel

from .load_questions import BenchQuestion
from .retrieval_metrics import compute_retrieval_metrics
from .routing_checks import run_category_checks
from .telemetry import extract_telemetry, token_usage, extract_draft_answers
from .judge import build_judge_llm, build_judge_chain, judge_one

logger = logging.getLogger("uvicorn.eval")


async def build_eval_dependencies(settings=None) -> dict:
    settings = settings or get_settings()
    # Wide preview so draft/final answer text isn't truncated before the
    # judge or report.py's aggregation ever sees it (see telemetry.py's
    # extract_draft_answers docstring).
    configure_tracing(level="INFO", preview_chars=settings.EVAL_PREVIEW_CHARS)

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

    judge_llm = build_judge_llm(settings)
    judge_chain = build_judge_chain(judge_llm)

    return {
        "settings": settings,
        "engine": engine,
        "db_client": db_client,
        "rag_graph": rag_graph,
        "judge_chain": judge_chain,
    }


async def run_one(question: BenchQuestion, deps: dict, project_id: int, phase: int, run_tag: str) -> dict:
    settings = deps["settings"]
    thread_id = f"eval-{run_tag}-{question.question_id}"

    started = time.perf_counter()
    error = None
    final_state: dict = {}
    trace: list = []

    with run_context(thread_id, debug=True) as (_run_id, trace):
        try:
            final_state = await run_rag_graph(
                deps["rag_graph"], question=question.question, project_id=project_id,
                thread_id=thread_id, max_iterations=settings.AGENT_MAX_ITERATIONS,
                recursion_limit=settings.AGENT_RECURSION_LIMIT,
            )
        except Exception as exc:
            logger.exception("Agent run failed for question_id=%s", question.question_id)
            error = repr(exc)
    wall_ms = round((time.perf_counter() - started) * 1000)

    generated_answer = final_state.get("final_answer") or final_state.get("draft_answer") or ""

    retrieval = compute_retrieval_metrics(final_state, question.expected_doc_ids)
    routing = run_category_checks(question.question_type, trace, final_state, question.expected_doc_ids)
    telem = extract_telemetry(trace, final_state)
    telem.update(token_usage(trace))
    telem["wall_clock_ms"] = wall_ms
    draft_answers = extract_draft_answers(trace)

    verdict = None
    if error is None:
        try:
            verdict = await judge_one(
                deps["judge_chain"], question.question, question.gold_answer,
                question.answer_facts, generated_answer, question.question_type,
            )
        except Exception:
            logger.exception("Judge call failed for question_id=%s", question.question_id)

    return {
        "question_id": question.question_id,
        "question_type": question.question_type,
        "phase": phase,
        "run_tag": run_tag,
        "error": error,
        "final_state": final_state,
        "generated_answer": generated_answer,
        "draft_answers": draft_answers,
        "retrieval": retrieval,
        "routing": routing,
        "telemetry": telem,
        "judge_verdict": verdict,
    }