"""
stores/agents/nodes/memory.py

Short-term memory nodes (+ the long-term recall node).

  load_context   keep history bounded: summarize old turns past a token budget
  contextualize  rewrite a follow-up into a standalone question (skipped, with no
                 LLM call, on the first turn of a thread); sets `question`
  recall_memory  long-term recall into `recalled_memories` (read-only context)
  commit_turn    append the assistant reply to `messages`; single exit for ALL paths

Every LLM-dependent step fails open: a memory failure must never break answering.
"""

import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from ..config import AgentConfig
from ..memory.tokens import count_message_tokens
from ..prompts import CONTEXTUALIZE_PROMPT, SUMMARIZE_PROMPT
from ..state import RAGState

logger = logging.getLogger("uvicorn")


def _text(m) -> str:
    return m.content if isinstance(m.content, str) else str(m.content)


def prior_messages(state: RAGState) -> List:
    """History excluding the human message of the current turn."""
    msgs = list(state.get("messages", []))
    if msgs and isinstance(msgs[-1], HumanMessage):
        msgs = msgs[:-1]
    return [m for m in msgs if isinstance(m, (HumanMessage, AIMessage))]


def format_transcript(messages, assistant_chars: int = 600) -> str:
    lines = []
    for m in messages:
        text = _text(m)
        if isinstance(m, AIMessage):
            if len(text) > assistant_chars:
                text = text[:assistant_chars] + "..."
            lines.append(f"Assistant: {text}")
        else:
            lines.append(f"User: {text}")
    return "\n".join(lines)


def build_load_context_node(summarizer_llm, config: AgentConfig):
    chain = SUMMARIZE_PROMPT | summarizer_llm
    keep = max(2, config.history_max_turns * 2)

    async def load_context_node(state: RAGState) -> dict:
        history = prior_messages(state)
        if count_message_tokens(history) <= config.summary_trigger_tokens:
            return {}

        old, _recent = history[:-keep], history[-keep:]
        if not old:
            return {}

        try:
            res = await chain.ainvoke({
                "summary": state.get("conversation_summary") or "(none)",
                "transcript": format_transcript(old, assistant_chars=1500),
            })
        except Exception:
            logger.exception("[agent] load_context: summarization failed, keeping full history")
            return {}

        summary = (res.content if hasattr(res, "content") else str(res)).strip()
        if not summary:
            return {}

        logger.info("[agent] load_context: summarized %d old messages", len(old))
        return {
            "conversation_summary": summary,
            "messages": [RemoveMessage(id=m.id) for m in old if m.id],
        }
    return load_context_node


def build_contextualize_node(contextualizer_llm, config: AgentConfig):
    chain = CONTEXTUALIZE_PROMPT | contextualizer_llm
    keep = max(2, config.history_max_turns * 2)

    async def contextualize_node(state: RAGState) -> dict:
        original = state.get("original_question") or state["question"]
        history = prior_messages(state)
        summary = state.get("conversation_summary")

        if not history and not summary:  # first turn: nothing to resolve
            return {"standalone_question": original}

        try:
            res = await chain.ainvoke({
                "summary": summary or "(none)",
                "history": format_transcript(history[-keep:]),
                "question": original,
            })
            rewritten = (res.content if hasattr(res, "content") else str(res)).strip().strip('"')
        except Exception:
            logger.exception("[agent] contextualize failed, using the original question")
            return {"standalone_question": original}

        # Guard against the model answering instead of rewriting, or going off the rails.
        if not rewritten or len(rewritten) > 4 * len(original) + 400:
            return {"standalone_question": original}

        logger.info("[agent] contextualize: %r -> %r", original, rewritten)
        return {"standalone_question": rewritten, "question": rewritten}
    return contextualize_node


def build_recall_memory_node(memory_service, config: AgentConfig):
    async def recall_memory_node(state: RAGState) -> dict:
        user_id = state.get("user_id")
        if not user_id:
            return {"recalled_memories": []}
        query = state.get("standalone_question") or state["question"]
        try:
            memories = await memory_service.recall(
                project_id=state["project_id"], user_id=user_id, query=query,
                top_k=config.ltm_top_k, min_score=config.ltm_min_score)
        except Exception:
            logger.exception("[agent] recall_memory failed, continuing without memories")
            memories = []
        logger.info("[agent] recall_memory: %d memories", len(memories))
        return {"recalled_memories": memories}
    return recall_memory_node


def build_commit_turn_node():
    """Does NOT clear chunks: eval reads retrieved_chunks from the final state."""
    async def commit_turn_node(state: RAGState) -> dict:
        answer = state.get("final_answer") or state.get("draft_answer")
        if not answer:  # keep user/assistant alternation intact on failures
            answer = "I wasn't able to produce an answer to that question."
        return {"messages": [AIMessage(
            content=answer,
            additional_kwargs={
                "citations": state.get("citations", []),
                "question_type": state.get("question_type"),
            },
        )]}
    return commit_turn_node