"""
stores/agents/nodes/generation.py

Answer generation. Context selection is a pure function
(`select_context_chunks`) so it can be unit-tested without an LLM.
"""

import logging
from typing import List

from ..config import AgentConfig
from ..state import RAGState

logger = logging.getLogger("uvicorn")


def select_context_chunks(chunks: List, *, min_score: float, max_chunks: int) -> List:
    """Dedupe by chunk_id (keep best score), drop weak chunks, cap the total.

    Chunks arrive from several retrieval branches via an append-only reducer,
    so duplicates are expected.
    """
    best: dict = {}
    for d in chunks:
        score = getattr(d, "score", 0) or 0
        if d.chunk_id not in best or score > best[d.chunk_id][0]:
            best[d.chunk_id] = (score, d)
    ranked = sorted(best.values(), key=lambda pair: pair[0], reverse=True)
    return [d for score, d in ranked if score >= min_score][:max_chunks]


def build_generator_node(generation_llm, template_parser, config: AgentConfig):
    async def generator_node(state: RAGState) -> dict:
        chunks = select_context_chunks(
            state.get("retrieved_chunks", []),
            min_score=config.min_generation_score,
            max_chunks=config.max_generation_chunks,
        )
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

        try:
            answer = await generation_llm.generate_text(
                prompt=full_prompt,
                chat_history=[generation_llm.construct_prompt(system_prompt, role="system")],
            )
        except Exception as exc:
            logger.exception("[agent] generator failed")
            return {"draft_answer": None, "error": f"generation_failed: {exc}"}

        if not answer:
            return {"draft_answer": None, "error": "generation_returned_empty"}

        logger.info("[agent] generator: answer=%d chars", len(answer))
        return {"draft_answer": answer, "citations": sorted({d.doc_id for d in chunks})}
    return generator_node
