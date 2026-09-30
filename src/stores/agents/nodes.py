"""
stores/agents/nodes.py

Every node takes (state: RAGState) and returns a partial dict (merged via the
reducers in state.py) or a Command(update=..., goto=...) when it also decides
the next hop. `nlp_controller` / `generation_llm` / `get_project` are
injected via closures from graph.py -- built once in main.py's lifespan and
reused, same as NLPController's clients.
"""

import logging
from typing import Literal

from langgraph.graph import END
from langgraph.types import Command, Send
from langchain_core.prompts import ChatPromptTemplate
from stores.vectordb.VectorDBInterface import MetadataFilter

from .schemas import ExtractedFilters, SubQuestions, DraftClaims
from .state import RAGState

logger = logging.getLogger("uvicorn")


# ---------------------------------------------------------------------------
# Direct Retriever + Generator (Phase 1)
# ---------------------------------------------------------------------------

def build_direct_retriever_node(nlp_controller, get_project):
    async def direct_retriever_node(state: RAGState) -> dict:
        project = await get_project(state["project_id"])
        result = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=20, top_k=5,
        )
        if not result.ok:
            return {"error": result.error or "retrieval_failed"}
        return {
            "retrieved_chunks": result.documents,
            "excluded_chunk_ids": [d.chunk_id for d in result.documents],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return direct_retriever_node


def build_generator_node(generation_llm, template_parser):
    async def generator_node(state: RAGState) -> dict:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return {"draft_answer": None, "error": "no_chunks_to_generate_from"}

        system_prompt = template_parser.get("rag", "system_prompt")
        documents_prompt = "\n".join(
            template_parser.get("rag", "document_prompt", {"doc_num": i + 1, "chunk_text": doc.text})
            for i, doc in enumerate(chunks)
        )
        footer_prompt = template_parser.get("rag", "footer_prompt", {"query": state["question"]})
        full_prompt = "\n\n".join([documents_prompt, footer_prompt])

        logger.info("[agent] generator: %d chunks, prompt=%d chars, docs=%s",
            len(chunks), len(full_prompt), sorted({d.doc_id for d in chunks}))
        answer = await generation_llm.generate_text(
            prompt=full_prompt,
            chat_history=[generation_llm.construct_prompt(system_prompt, role="system")],
        )
        logger.info("[agent] generator: answer=%d chars", len(answer or ""))
        return {"draft_answer": answer, "citations": sorted({doc.doc_id for doc in chunks})}
    return generator_node


# ---------------------------------------------------------------------------
# Specialized sub-agents (Phase 2)
# ---------------------------------------------------------------------------

_DECOMPOSE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Break the question into 3-6 focused sub-questions that together cover "
               "every distinct fact needed to answer it fully. Also list the distinct "
               "answerable facets."),
    ("human", "{question}"),
])


def build_decomposer_node(decomposer_llm):
    """Fans out via `Send` -- LangGraph's built-in dynamic-parallelism
    primitive -- to the shared `retrieve_subquery` worker node."""
    chain = _DECOMPOSE_PROMPT | decomposer_llm.with_structured_output(SubQuestions)

    async def decomposer_node(state: RAGState) -> Command[Literal["retrieve_subquery"]]:
        result: SubQuestions = await chain.ainvoke({"question": state["question"]})
        sends = [Send("retrieve_subquery", {**state, "question": q, "sub_queries": [q]}) for q in result.sub_queries]
        return Command(update={"facets": result.facets, "sub_queries": result.sub_queries}, goto=sends)
    return decomposer_node


def build_retrieve_subquery_node(nlp_controller, get_project):
    async def retrieve_subquery_node(state: RAGState) -> dict:
        project = await get_project(state["project_id"])
        result = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=20, top_k=6,
        )
        if not result.ok:
            logger.warning("[agent] retrieve_subquery failed q=%r err=%s", state["question"], result.error)
            return {}
        return {
            "retrieved_chunks": result.documents,
            "excluded_chunk_ids": [d.chunk_id for d in result.documents],
        }
    return retrieve_subquery_node


_FILTER_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Extract any explicit qualifiers (source type, date range, status, team, "
               "product) from the question that would narrow a document search. Leave "
               "fields empty if no qualifier of that kind is present."),
    ("human", "{question}"),
])


def build_constraint_filter_node(filter_llm, nlp_controller, get_project):
    chain = _FILTER_EXTRACTION_PROMPT | filter_llm.with_structured_output(ExtractedFilters)

    async def constraint_filter_node(state: RAGState) -> dict:
        extracted: ExtractedFilters = await chain.ainvoke({"question": state["question"]})
        
        filters = MetadataFilter(
            equals=extracted.other_equals,
            in_={"source_type": extracted.source_types} if extracted.source_types else {},
        )
        project = await get_project(state["project_id"])
        result = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=40, top_k=10, filters=filters,
        )
        if result.ok and result.documents:
            return {
                "retrieval_filters": extracted.model_dump(),
                "retrieved_chunks": result.documents,
                "excluded_chunk_ids": [d.chunk_id for d in result.documents],
            }
        logger.info("[agent] constraint_filter: filtered pass empty, filters=%s -> falling back",
            extracted.model_dump())
        # filtered pass came up empty -- fall back to unfiltered
        fallback = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=40, top_k=10,
        )
        return {
            "retrieval_filters": {**extracted.model_dump(), "fallback_unfiltered": True},
            "retrieved_chunks": fallback.documents if fallback.ok else [],
            "excluded_chunk_ids": [d.chunk_id for d in fallback.documents] if fallback.ok else [],
        }
    return constraint_filter_node


def build_conflict_resolver_node(nlp_controller, get_project):
    async def conflict_resolver_node(state: RAGState) -> dict:
        project = await get_project(state["project_id"])
        result = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=60, top_k=None,
        )
        if not result.ok:
            return {"error": result.error or "retrieval_failed"}
        return {
            "retrieved_chunks": result.documents,
            "excluded_chunk_ids": [d.chunk_id for d in result.documents],
        }
    return conflict_resolver_node


def build_completeness_sweep_node(nlp_controller, get_project):
    async def completeness_sweep_node(state: RAGState) -> dict:
        project = await get_project(state["project_id"])
        excluded = state.get("excluded_chunk_ids", [])
        result = await nlp_controller.search_vector_db_collection(
            project=project, text=state["question"], candidate_k=30, top_k=10,
            exclude_chunk_ids=excluded,
        )
        if not result.ok:
            logger.warning("[agent] completeness_sweep failed err=%s", result.error)
            
        new_docs = result.documents if result.ok else []
        logger.info("[agent] completeness_sweep: excluded=%d new=%d", len(excluded), len(new_docs))
        return {
            "retrieved_chunks": new_docs,
            "excluded_chunk_ids": [d.chunk_id for d in new_docs],
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        }
    return completeness_sweep_node


def build_route_after_completeness_sweep(max_docs: int, max_attempts: int):
    """Factory instead of a bare function so the thresholds come from
    settings (AGENT_COMPLETENESS_MAX_DOCS / _MAX_ATTEMPTS) at graph-build
    time, not hardcoded constants."""
    def route_after_completeness_sweep(state: RAGState) -> Literal["completeness_sweep", "generator"]:
        total = len(state.get("retrieved_chunks", []))
        attempts = state.get("retrieval_attempts", 0)
        if total >= max_docs or attempts >= max_attempts:
            return "generator"
        return "completeness_sweep"
    return route_after_completeness_sweep


def build_broad_synthesis_node(expansion_llm):
    chain = _DECOMPOSE_PROMPT | expansion_llm.with_structured_output(SubQuestions)

    async def broad_synthesis_node(state: RAGState) -> Command[Literal["retrieve_subquery"]]:
        result: SubQuestions = await chain.ainvoke({"question": state["question"]})
        sends = [Send("retrieve_subquery", {**state, "question": q, "sub_queries": [q]}) for q in result.sub_queries]
        return Command(update={"sub_queries": result.sub_queries}, goto=sends)
    return broad_synthesis_node


# ---------------------------------------------------------------------------
# Verifier gate + reformulation loop (Phase 3)
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "List each discrete factual claim in the draft answer, with the doc_id(s) "
               "it cites, if any were given inline."),
    ("human", "Draft answer:\n{draft_answer}"),
])


def build_verifier_node(verifier_llm):
    chain = _CLAIM_EXTRACTION_PROMPT | verifier_llm.with_structured_output(DraftClaims)

    async def verifier_node(state: RAGState) -> dict:
        draft = state.get("draft_answer")
        if not draft:
            return {"grounding_passed": False, "unsupported_claims": ["no_draft_answer"]}

        extraction: DraftClaims = await chain.ainvoke({"draft_answer": draft})
        chunk_by_doc_id: dict[str, list] = {}
        for doc in state.get("retrieved_chunks", []):
            chunk_by_doc_id.setdefault(doc.doc_id, []).append(doc)

        verified, unsupported = [], []
        for c in extraction.claims:
            supported = any(doc_id in chunk_by_doc_id for doc_id in c.cited_doc_ids)
            verified.append({
                "claim": c.claim,
                "supported": supported,
                "supporting_doc_ids": [d for d in c.cited_doc_ids if d in chunk_by_doc_id],
                "chunk_ids": [doc.chunk_id for d in c.cited_doc_ids for doc in chunk_by_doc_id.get(d, [])],
            })
            if not supported:
                unsupported.append(c.claim)

        logger.info("[agent] verifier: extracted %d claims, %d have no citation",
            len(extraction.claims), sum(1 for c in extraction.claims if not c.cited_doc_ids))
        return {
            "verified_claims": verified,
            "unsupported_claims": unsupported,
            "grounding_passed": len(unsupported) == 0 and len(verified) > 0,
        }
    return verifier_node


def build_reformulator_node(reform_llm):
    reform_prompt = ChatPromptTemplate.from_messages([
        ("system", "The previous retrieval + draft answer failed grounding for these claims: "
                   "{unsupported}. Rewrite the original question to target the missing "
                   "information more precisely."),
        ("human", "{question}"),
    ])
    chain = reform_prompt | reform_llm

    async def reformulator_node(state: RAGState) -> dict:
        new_question = await chain.ainvoke({
            "question": state["question"],
            "unsupported": "; ".join(state.get("unsupported_claims", [])),
        })
        return {
            "question": new_question.content if hasattr(new_question, "content") else str(new_question),
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    return reformulator_node


def build_info_not_found_node():
    async def info_not_found_node(state: RAGState) -> dict:
        return {
            "final_answer": (
                "I couldn't find information supporting an answer to this question "
                "in the available documents after an exhaustive search."
            ),
            "citations": [],
        }
    return info_not_found_node


def route_after_verifier(state: RAGState) -> Literal["reformulator", "finalize", "info_not_found"]:
    if state.get("grounding_passed"):
        return "finalize"
    if state.get("iteration_count", 0) >= state.get("max_iterations", 3):
        return "info_not_found"
    return "reformulator"


def build_finalize_node():
    async def finalize_node(state: RAGState) -> dict:
        return {
            "final_answer": state.get("draft_answer"),
            "citations": sorted({
                doc_id for claim in state.get("verified_claims", [])
                for doc_id in claim["supporting_doc_ids"]
            }) or state.get("citations", []),
        }
    return finalize_node