"""
stores/agents/nodes/generation.py
 
Answer generation. Context selection is a pure function
(`select_context_chunks`) so it can be unit-tested without an LLM.
 
Memory additions:
  * recalled long-term memories go in a clearly-labelled "background, not
    evidence" block. They never enter `citations`, and the verifier only ever
    sees chunks, so a memory can't "ground" a claim.
  * optional chat history (AGENT_GENERATOR_SEES_HISTORY, default off): the
    standalone question already carries the context.
"""
 
import logging
from typing import List
 
from langchain_core.messages import AIMessage, HumanMessage
 
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

def _history_as_chat(generation_llm, messages, max_turns: int) -> list:
    msgs = list(messages or [])
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]
    msgs = msgs[-max(2, max_turns * 2):]
    out = []
    for m in msgs:
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        else:
            continue
        out.append(generation_llm.construct_prompt(
            m.content if isinstance(m.content, str) else str(m.content), role=role))
    return out

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
        
        parts = [documents_prompt, footer_prompt]
        memories = state.get("recalled_memories") or []
        if memories:
            memory_prompt = template_parser.get("rag", "memory_prompt", {
                "memories": "\n".join(f"- {m['text']}" for m in memories)})
            parts.insert(0, memory_prompt)  # before the documents: background, not evidence
        full_prompt = "\n\n".join(parts)
        
        chat_history = [generation_llm.construct_prompt(system_prompt, role="system")]
        if config.generator_sees_history:
            chat_history += _history_as_chat(
                generation_llm, state.get("messages"), config.history_max_turns)
            
        logger.info("[agent] generator: %d chunks, %d memories, prompt=%d chars, docs=%s",
                    len(chunks), len(memories), len(full_prompt), sorted({d.doc_id for d in chunks}))

        try:
            answer = await generation_llm.generate_text(
                prompt=full_prompt,
                chat_history=chat_history,
            )
        except Exception as exc:
            logger.exception("[agent] generator failed")
            return {"draft_answer": None, "error": f"generation_failed: {exc}"}

        if not answer:
            return {"draft_answer": None, "error": "generation_returned_empty"}

        logger.info("[agent] generator: answer=%d chars", len(answer))
        return {"draft_answer": answer, "citations": sorted({d.doc_id for d in chunks})}
    return generator_node
