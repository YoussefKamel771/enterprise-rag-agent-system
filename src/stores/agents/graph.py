"""
stores/agents/graph.py
 
Public surface: `build_rag_graph(settings, nlp_controller, generation_client,
template_parser, get_project, checkpointer=None, memory_service=None)`.
The two new kwargs are optional, so existing callers keep working.
 
Layers (additive):
    phase 1: core          supervisor -> direct_retriever -> generator
    phase 2: + specialists decomposer / constraint_filter / conflict_resolver /
                           completeness_sweep / broad_synthesis
    phase 3: + verification generator -> verifier -> (finalize | reformulator
                           | info_not_found)
    memory : + short-term  START -> load_context -> contextualize
                                 [-> recall_memory] -> supervisor ... -> commit_turn -> END
 
Memory is orthogonal to the phases: the entry node and the terminal node are
variables, so EVERY exit (generator in phases 1-2, finalize / info_not_found in
phase 3, supervisor failure) funnels through `commit_turn`.
"""

import logging
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from .config import AgentConfig
from .llm import AgentLLMs, build_agent_llms, build_classifier_llm
from .nodes import (
    build_broad_synthesis_node, build_commit_turn_node, build_completeness_sweep_node,
    build_conflict_resolver_node, build_constraint_filter_node, build_contextualize_node,
    build_decomposer_node, build_direct_retriever_node, build_finalize_node,
    build_generator_node, build_info_not_found_node, build_load_context_node,
    build_recall_memory_node, build_reformulator_node, build_retrieve_subquery_node,
    build_verifier_node,
)
from .retrieval import RetrievalGateway
from .routing import (
    BROAD_SYNTHESIS, COMMIT_TURN, COMPLETENESS_SWEEP, CONFLICT_RESOLVER, CONSTRAINT_FILTER,
    CONTEXTUALIZE, CORE_DESTINATIONS, DECOMPOSER, DIRECT_RETRIEVER, FINALIZE, GENERATOR,
    INFO_NOT_FOUND, LOAD_CONTEXT, RECALL_MEMORY, REFORMULATOR, RETRIEVE_SUBQUERY,
    SPECIALIST_DESTINATIONS, SUPERVISOR, VERIFIER,
    build_route_after_completeness_sweep, route_after_verifier,
)
from .runtime import make_config, make_initial_state, run_rag_graph  # noqa: F401 (re-exported)
from .state import RAGState
from .supervisor import build_supervisor_node
from .tracing import traced

logger = logging.getLogger("uvicorn")

# Backward-compatible alias: eval/run_supervisor.py imports this private name.
_build_classifier_llm = build_classifier_llm


@dataclass(frozen=True)
class PhaseSpec:
    """Which additive layers a phase includes."""
    specialists: bool
    verification: bool


PHASES = {
    1: PhaseSpec(specialists=False, verification=False),
    2: PhaseSpec(specialists=True, verification=False),
    3: PhaseSpec(specialists=True, verification=True),
}


class AgentGraphBuilder:
    def __init__(self, *, config: AgentConfig, llms: AgentLLMs, gateway: RetrievalGateway,
                 generation_client, template_parser, checkpointer=None, memory_service=None):
        self._config = config
        self._llms = llms
        self._gateway = gateway
        self._generation_client = generation_client
        self._template_parser = template_parser
        self._checkpointer = checkpointer
        self._memory_service = memory_service

    # ------------------------------------------------------------------ build
    def build(self):
        spec = PHASES.get(self._config.graph_phase)
        if spec is None:
            raise ValueError(
                f"Unsupported AGENT_GRAPH_PHASE={self._config.graph_phase}; expected one of {sorted(PHASES)}")

        graph = StateGraph(RAGState)
        
        entry, exit_ = SUPERVISOR, END
        if self._config.memory_enabled:
            entry, exit_ = self._add_memory(graph)
        
        self._add_core(graph, spec, on_failure=exit_)

        if spec.specialists:
            self._add_specialists(graph)
        if spec.verification:
            self._add_verification(graph)
        else:
            graph.add_edge(GENERATOR, END)
            
        graph.add_edge(START, entry)
        return graph.compile()

    # ----------------------------------------------------------------- layers
    @staticmethod
    def _add(graph, name, fn):
        """Register a node wrapped with the logging decorator."""
        graph.add_node(name, traced(name)(fn))

    @staticmethod
    def _add_conditional(graph, source, routing_fn, path_map, name):
        graph.add_conditional_edges(source, traced(name)(routing_fn), path_map)

    def _add_core(self, graph, spec: PhaseSpec):
        cfg, gw = self._config, self._gateway
        # Only nodes that exist in this phase are valid supervisor jumps.
        allowed = CORE_DESTINATIONS | (SPECIALIST_DESTINATIONS if spec.specialists else frozenset())

        self._add(graph, SUPERVISOR, build_supervisor_node(
            self._llms.classifier, cfg.max_iterations, allowed_destinations=allowed))
        self._add(graph, DIRECT_RETRIEVER, build_direct_retriever_node(gw, cfg))
        self._add(graph, GENERATOR, build_generator_node(self._generation_client, self._template_parser, cfg))

        graph.add_edge(START, SUPERVISOR)
        graph.add_edge(DIRECT_RETRIEVER, GENERATOR)

    def _add_specialists(self, graph):
        cfg, gw, llms = self._config, self._gateway, self._llms

        self._add(graph, DECOMPOSER, build_decomposer_node(llms.decomposer))
        self._add(graph, BROAD_SYNTHESIS, build_broad_synthesis_node(llms.expansion))
        self._add(graph, RETRIEVE_SUBQUERY, build_retrieve_subquery_node(gw, cfg))
        self._add(graph, CONSTRAINT_FILTER, build_constraint_filter_node(llms.constraint, gw, cfg))
        self._add(graph, CONFLICT_RESOLVER, build_conflict_resolver_node(gw, cfg))
        self._add(graph, COMPLETENESS_SWEEP, build_completeness_sweep_node(gw, cfg))

        # decomposer / broad_synthesis reach retrieve_subquery via Send (Command.goto)
        for node in (RETRIEVE_SUBQUERY, CONSTRAINT_FILTER, CONFLICT_RESOLVER):
            graph.add_edge(node, GENERATOR)

        # completeness_sweep loops on itself until enough docs / attempts
        self._add_conditional(
            graph, COMPLETENESS_SWEEP,
            build_route_after_completeness_sweep(cfg.completeness_max_docs, cfg.completeness_max_attempts),
            {COMPLETENESS_SWEEP: COMPLETENESS_SWEEP, GENERATOR: GENERATOR},
            name="route_after_completeness_sweep",
        )

    def _add_verification(self, graph):
        self._add(graph, VERIFIER, build_verifier_node(self._llms.verifier, self._config))
        self._add(graph, REFORMULATOR, build_reformulator_node(self._llms.reformulator))
        self._add(graph, INFO_NOT_FOUND, build_info_not_found_node())
        self._add(graph, FINALIZE, build_finalize_node())

        graph.add_edge(GENERATOR, VERIFIER)
        self._add_conditional(
            graph, VERIFIER, route_after_verifier,
            {FINALIZE: FINALIZE, REFORMULATOR: REFORMULATOR, INFO_NOT_FOUND: INFO_NOT_FOUND},
            name="route_after_verifier",
        )
        graph.add_edge(REFORMULATOR, DIRECT_RETRIEVER)
        graph.add_edge(FINALIZE, END)
        graph.add_edge(INFO_NOT_FOUND, END)
        
    def _add_memory(self, graph):
        """Returns (entry_node, exit_node)."""
        cfg, llms = self._config, self._llms
        if self._checkpointer is None:
            logger.warning("[agent] memory_enabled but no checkpointer: history will NOT persist "
                           "between calls (set AGENT_CHECKPOINTER_BACKEND)")
 
        self._add(graph, LOAD_CONTEXT, build_load_context_node(llms.summarizer, cfg))
        self._add(graph, CONTEXTUALIZE, build_contextualize_node(llms.contextualizer, cfg))
        self._add(graph, COMMIT_TURN, build_commit_turn_node())
 
        graph.add_edge(LOAD_CONTEXT, CONTEXTUALIZE)
        if cfg.ltm_enabled and self._memory_service is not None:
            self._add(graph, RECALL_MEMORY, build_recall_memory_node(self._memory_service, cfg))
            graph.add_edge(CONTEXTUALIZE, RECALL_MEMORY)
            graph.add_edge(RECALL_MEMORY, SUPERVISOR)
        else:
            if cfg.ltm_enabled:
                logger.warning("[agent] ltm_enabled but no memory_service: long-term recall disabled")
            graph.add_edge(CONTEXTUALIZE, SUPERVISOR)
 
        graph.add_edge(COMMIT_TURN, END)
        return LOAD_CONTEXT, COMMIT_TURN


def build_rag_graph(settings, nlp_controller, generation_client, template_parser, get_project):
    """Single entry point -- called once from main.py's lifespan (and from
    eval.run_agent.build_eval_dependencies). Which compiled graph comes back is
    a config switch (settings.AGENT_GRAPH_PHASE), same pattern as
    GENERATION_BACKEND / VECTOR_DB_BACKEND."""
    return AgentGraphBuilder(
        config=AgentConfig.from_settings(settings),
        llms=build_agent_llms(settings),
        gateway=RetrievalGateway(nlp_controller, get_project),
        generation_client=generation_client,
        template_parser=template_parser,
    ).build()
