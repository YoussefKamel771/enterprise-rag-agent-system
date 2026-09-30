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
# from .tracing import traced

logger = logging.getLogger("uvicorn")

# def _node(graph, name, fn):
#     graph.add_node(name, traced(name)(fn))

def build_chat_model(settings, model_id: str):
    return ChatOpenAI(
        model=model_id,
        api_key="ollama",
        base_url=settings.OPENAI_API_URL,
        temperature=0,
    )

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
            max_tokens=512, # remove after
            temperature=0,
        )
    raise ValueError(f"Unsupported AGENT_CLASSIFIER_BACKEND: {settings.AGENT_CLASSIFIER_BACKEND}")


def _build_phase1(*, classifier_llm, nlp_controller, generation_llm, template_parser, get_project):
    graph = StateGraph(RAGState)
    graph.add_node("supervisor", build_supervisor_node(
                                    classifier_llm,
                                    allowed_destinations=["direct_retriever", "decomposer", "constraint_filter",
                                                        "conflict_resolver", "completeness_sweep", "broad_synthesis"],
                                ))
    graph.add_node("direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_node("generator", build_generator_node(generation_llm, template_parser))

    graph.add_edge(START, "supervisor")
    graph.add_edge("direct_retriever", "generator")
    graph.add_edge("generator", END)
    return graph.compile()


def _build_phase2(*, classifier_llm, decomposer_llm, filter_llm, expansion_llm,
                   nlp_controller, generation_llm, template_parser, get_project,
                   completeness_max_docs, completeness_max_attempts):
    graph = StateGraph(RAGState)
    graph.add_node("supervisor", build_supervisor_node(classifier_llm))
    graph.add_node("direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_node("decomposer", build_decomposer_node(decomposer_llm))
    graph.add_node("retrieve_subquery", build_retrieve_subquery_node(nlp_controller, get_project))
    graph.add_node("constraint_filter", build_constraint_filter_node(filter_llm, nlp_controller, get_project))
    graph.add_node("conflict_resolver", build_conflict_resolver_node(nlp_controller, get_project))
    graph.add_node("completeness_sweep", build_completeness_sweep_node(nlp_controller, get_project))
    graph.add_node("broad_synthesis", build_broad_synthesis_node(expansion_llm))
    graph.add_node("generator", build_generator_node(generation_llm, template_parser))

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
    graph.add_node("supervisor", build_supervisor_node(classifier_llm))
    graph.add_node("direct_retriever", build_direct_retriever_node(nlp_controller, get_project))
    graph.add_node("decomposer", build_decomposer_node(decomposer_llm))
    graph.add_node("retrieve_subquery", build_retrieve_subquery_node(nlp_controller, get_project))
    graph.add_node("constraint_filter", build_constraint_filter_node(filter_llm, nlp_controller, get_project))
    graph.add_node("conflict_resolver", build_conflict_resolver_node(nlp_controller, get_project))
    graph.add_node("completeness_sweep", build_completeness_sweep_node(nlp_controller, get_project))
    graph.add_node("broad_synthesis", build_broad_synthesis_node(expansion_llm))
    graph.add_node("generator", build_generator_node(generation_llm, template_parser))
    graph.add_node("verifier", build_verifier_node(verifier_llm))
    graph.add_node("reformulator", build_reformulator_node(reform_llm))
    graph.add_node("info_not_found", build_info_not_found_node())
    graph.add_node("finalize", build_finalize_node())

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

    if phase == 1:
        return _build_phase1(classifier_llm=classifier_llm, **common)
    if phase == 2:
        return _build_phase2(
            classifier_llm=classifier_llm, decomposer_llm=classifier_llm,
            filter_llm=classifier_llm, expansion_llm=classifier_llm,
            completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
            completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
            **common,
        )
    return _build_phase3(
        classifier_llm=classifier_llm, decomposer_llm=classifier_llm,
        filter_llm=classifier_llm, expansion_llm=classifier_llm,
        verifier_llm=classifier_llm, reform_llm=classifier_llm,
        completeness_max_docs=settings.AGENT_COMPLETENESS_MAX_DOCS,
        completeness_max_attempts=settings.AGENT_COMPLETENESS_MAX_ATTEMPTS,
        **common,
    )


async def run_rag_graph(app_graph, question: str, project_id: int, thread_id: str,
                         max_iterations: int = 3, recursion_limit: int = 25):
    initial_state: RAGState = make_initial_state(question, project_id, max_iterations)
    
    config = make_config(thread_id, recursion_limit)
    
    logger.info("[agent] RUN start thread=%s project=%s q=%r", thread_id, project_id, question)
    t0 = time.perf_counter()
    
    result = await app_graph.ainvoke(initial_state, config=config)
    
    logger.info("[agent] RUN end (%.0fs) type=%s agent=%s iters=%s chunks=%d error=%s",
                (time.perf_counter() - t0) , result.get("question_type"),
                result.get("active_agent"), result.get("iteration_count"),
                len(result.get("retrieved_chunks", [])), result.get("error"))
    
    return result


def make_initial_state(question, project_id, max_iterations=3) -> RAGState:
    return {"question": question, "project_id": project_id, "retrieved_chunks": [],
            "excluded_chunk_ids": [], "sub_queries": [], "verified_claims": [],
            "unsupported_claims": [], "iteration_count": 0, "max_iterations": max_iterations}

def make_config(thread_id, recursion_limit=25, run_name="rag_agent", tags=None, metadata=None):
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": recursion_limit,
            "run_name": run_name, "tags": tags or [], "metadata": metadata or {}}