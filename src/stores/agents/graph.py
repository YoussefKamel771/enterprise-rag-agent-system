"""
stores/agents/graph.py

Public surface: `build_rag_graph(settings, nlp_controller, generation_client,
template_parser, get_project)`. Mirrors the existing *ProviderFactory
pattern (LLMProviderFactory.create(provider), VectorDBProviderFactory.create
(provider)) -- one config-driven switch (`settings.AGENT_GRAPH_PHASE`) picks
which compiled graph comes back, exactly like GENERATION_BACKEND /
VECTOR_DB_BACKEND pick a provider. Callers (main.py) never construct a
LangChain chat model or touch StateGraph directly, same as they never
construct an OpenAIProvider directly -- they go through the factory.
"""

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import time
from .state import RAGState
from .supervisor import build_supervisor_node
import logging
from .nodes import (
    build_direct_retriever_node,
    build_generator_node,
    build_decomposer_node,
    build_retrieve_subquery_node,
    build_constraint_filter_node,
    build_conflict_resolver_node,
    build_completeness_sweep_node,
    build_route_after_completeness_sweep,
    build_broad_synthesis_node,
    build_verifier_node,
    build_reformulator_node,
    build_info_not_found_node,
    route_after_verifier,
    build_finalize_node,
)
from .tracing import TracedStateGraph, AgentLogCallbackHandler, agent_logger

logger = logging.getLogger("uvicorn")



def _build_classifier_llm(settings):
    """The LLM used by the Supervisor/Decomposer/Filter/Verifier nodes for
    structured-output calls -- a LangChain chat model, separate from
    `generation_client` (your existing LLMInterface provider), which stays
    dedicated to answer generation inside build_generator_node.
    Add an ChatCohere branch here later the same way LLMProviderFactory
    branches on COHERE, if AGENT_CLASSIFIER_BACKEND ever needs it.
    """
    if settings.AGENT_CLASSIFIER_BACKEND == "OPENAI":
        return ChatOpenAI(
            model=settings.AGENT_CLASSIFIER_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL or None,
            temperature=0,
            callbacks=[AgentLogCallbackHandler()],
        )
    raise ValueError(f"Unsupported AGENT_CLASSIFIER_BACKEND: {settings.AGENT_CLASSIFIER_BACKEND}")


def _build_phase1(*, classifier_llm, nlp_controller, generation_llm, template_parser, get_project):
    graph = TracedStateGraph(RAGState)
    graph.add_node("supervisor", build_supervisor_node(classifier_llm))
    graph.add_node("direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_node("generator", build_generator_node(generation_llm, template_parser))
 
    graph.add_edge(START, "supervisor")
    graph.add_edge("direct_retriever", "generator")
    graph.add_edge("generator", END)
    return graph.compile(checkpointer=MemorySaver())


def _build_phase2(*, classifier_llm, decomposer_llm, filter_llm, expansion_llm,
                   nlp_controller, generation_llm, template_parser, get_project,
                   completeness_max_docs, completeness_max_attempts):
    graph = StateGraph(RAGState)
    graph.add_edge(graph,"supervisor", build_supervisor_node(classifier_llm))
    graph.add_edge(graph,"direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_edge(graph,"decomposer", build_decomposer_node(decomposer_llm))
    graph.add_edge(graph,"retrieve_subquery", build_retrieve_subquery_node(nlp_controller, get_project))
    graph.add_edge(graph,"constraint_filter", build_constraint_filter_node(filter_llm, nlp_controller, get_project))
    graph.add_edge(graph,"conflict_resolver", build_conflict_resolver_node(nlp_controller, get_project))
    graph.add_edge(graph,"completeness_sweep", build_completeness_sweep_node(nlp_controller, get_project))
    graph.add_edge(graph,"broad_synthesis", build_broad_synthesis_node(expansion_llm))
    graph.add_edge(graph,"generator", build_generator_node(generation_llm, template_parser))

    graph.add_edge(START, "supervisor")
    graph.add_edge("direct_retriever", "generator")
    graph.add_edge("retrieve_subquery", "generator")
    graph.add_edge("constraint_filter", "generator")
    graph.add_edge("conflict_resolver", "generator")
    graph.add_edge("broad_synthesis", "generator")

    graph.add_conditional_edges(
        "completeness_sweep",
        build_route_after_completeness_sweep(completeness_max_docs, completeness_max_attempts),
        {"completeness_sweep": "completeness_sweep", "generator": "generator"},
    )
    graph.add_edge("generator", END)
    return graph.compile(checkpointer=MemorySaver())


def _build_phase3(*, classifier_llm, decomposer_llm, filter_llm, expansion_llm,
                   verifier_llm, reform_llm, nlp_controller, generation_llm,
                   template_parser, get_project, completeness_max_docs,
                   completeness_max_attempts):
    graph = StateGraph(RAGState)
    graph.add_edge(graph,"supervisor", build_supervisor_node(classifier_llm))
    graph.add_edge(graph,"direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_edge(graph,"decomposer", build_decomposer_node(decomposer_llm))
    graph.add_edge(graph,"retrieve_subquery", build_retrieve_subquery_node(nlp_controller, get_project))
    graph.add_edge(graph,"constraint_filter", build_constraint_filter_node(filter_llm, nlp_controller, get_project))
    graph.add_edge(graph,"conflict_resolver", build_conflict_resolver_node(nlp_controller, get_project))
    graph.add_edge(graph,"completeness_sweep", build_completeness_sweep_node(nlp_controller, get_project))
    graph.add_edge(graph,"broad_synthesis", build_broad_synthesis_node(expansion_llm))
    graph.add_edge(graph,"generator", build_generator_node(generation_llm, template_parser))
    graph.add_edge(graph,"verifier", build_verifier_node(verifier_llm))
    graph.add_edge(graph,"reformulator", build_reformulator_node(reform_llm))
    graph.add_edge(graph,"info_not_found", build_info_not_found_node())
    graph.add_edge(graph,"finalize", build_finalize_node())

    graph.add_edge(START, "supervisor")
    graph.add_edge("direct_retriever", "generator")
    graph.add_edge("retrieve_subquery", "generator")
    graph.add_edge("constraint_filter", "generator")
    graph.add_edge("conflict_resolver", "generator")
    graph.add_edge("broad_synthesis", "generator")

    graph.add_conditional_edges(
        "completeness_sweep",
        build_route_after_completeness_sweep(completeness_max_docs, completeness_max_attempts),
        {"completeness_sweep": "completeness_sweep", "generator": "generator"},
    )
 
    graph.add_edge("generator", "verifier")
    graph.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"finalize": "finalize", "reformulator": "reformulator", "info_not_found": "info_not_found"},
    )
    graph.add_edge("reformulator", "direct_retriever")
    graph.add_edge("finalize", END)
    graph.add_edge("info_not_found", END)
    return graph.compile(checkpointer=MemorySaver())



def build_rag_graph(settings, nlp_controller, generation_client, template_parser, get_project):
    """Single entry point -- called once from main.py's lifespan, result
    stored on app.state.rag_graph, exactly like app.state.vectordb_client
    is built once via VectorDBProviderFactory.create(...).
    """
    classifier_llm = _build_classifier_llm(settings)
    common = dict(
        nlp_controller=nlp_controller,
        generation_llm=generation_client,
        template_parser=template_parser,
        get_project=get_project,
    )

    phase = settings.AGENT_GRAPH_PHASE
    return _build_phase1(classifier_llm=classifier_llm, **common)
    # if phase == 1:
    #     return _build_phase1(classifier_llm=classifier_llm, **common)
    # if phase == 2:
    #     return _build_phase2(
    #         classifier_llm=classifier_llm, decomposer_llm=classifier_llm,
    #         filter_llm=classifier_llm, expansion_llm=classifier_llm,
    #         completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
    #         completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
    #         **common,
    #     )
    # return _build_phase3(
    #     classifier_llm=classifier_llm, decomposer_llm=classifier_llm,
    #     filter_llm=classifier_llm, expansion_llm=classifier_llm,
    #     verifier_llm=classifier_llm, reform_llm=classifier_llm,
    #     completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
    #     completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
    #     **common,
    # )


async def run_rag_graph(app_graph, question: str, project_id: int, thread_id: str,
                         max_iterations: int = 3, recursion_limit: int = 25):
    initial_state: RAGState = {
        "question": question,
        "project_id": project_id,
        "retrieved_chunks": [],
        "excluded_chunk_ids": [],
        "sub_queries": [],
        "verified_claims": [],
        "unsupported_claims": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
    }
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit}
    
    t0 = time.perf_counter()
    
    agent_logger.info("RUN start project=%s q=%r", project_id, question)
    try:
        result = await app_graph.ainvoke(initial_state, config=config)
    except Exception:
        agent_logger.exception("RUN failed")
        raise
    
    agent_logger.info(
        "RUN end type=%s agent=%s iters=%s chunks=%s answer_len=%s RUN end (%.0fs)",
        result.get("question_type"), result.get("active_agent"), result.get("iteration_count"),
        len(result.get("retrieved_chunks", [])),
        len((result.get("final_answer") or result.get("draft_answer") or "")),
        (time.perf_counter() - t0)
    )
    
    return result