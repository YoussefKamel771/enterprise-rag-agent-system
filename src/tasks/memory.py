"""
tasks/memory.py

Long-term memory writer. Dispatched by AgentController AFTER the HTTP response is
built, so it adds no request latency, and benchmark runs (which never go through
the controller) can never write memories.

What gets written:
  * episode  -- deterministic, no LLM: the standalone question + the doc_ids that
                answered it (a POINTER, never the answer text: corpus facts go stale).
  * profile  -- LLM-extracted durable facts about the USER, taken from the user's
                own message only (never from the answer, so company facts can't leak in).

Each run opens (and closes) its own store connection: tasks run under asyncio.run,
a fresh event loop per call, so no module-level async pool.
"""

import asyncio
import hashlib
import logging
import re
import uuid

from celery_app import celery_app
from helpers.config import get_settings
from stores.llm import LLMProviderFactory
from stores.agents.llm import build_memory_llm
from stores.agents.prompts import EXTRACT_PROFILE_PROMPT
from stores.agents.schemas import ProfileFacts
from stores.agents.memory.backends import make_embed_fn, store_session
from stores.agents.memory.service import EPISODES, PROFILE, MemoryService

logger = logging.getLogger("celery.memory")

_SENSITIVE_PATTERNS = [
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),                                  # email
    re.compile(r"\+?\d[\d\s().-]{8,}\d"),                                     # phone-ish
    re.compile(r"\b(?:sk|pk|api|key|token|secret|passwd|password)[-_=:\s]+\S{6,}", re.I),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),                                     # AWS key id
]
_SIMILARITY_THRESHOLD = 0.9


def _looks_sensitive(text: str) -> bool:
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)


async def _write_memory(payload: dict) -> dict:
    settings = get_settings()
    if not settings.AGENT_LTM_ENABLED:
        return {"signal": "ltm_disabled"}

    project_id, user_id = payload["project_id"], payload["user_id"]

    embedding_client = LLMProviderFactory(settings).create(settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)
    extractor = EXTRACT_PROFILE_PROMPT | build_memory_llm(settings).with_structured_output(
        ProfileFacts, method="function_calling")

    written = {"episode": 0, "profile": 0}
    async with store_session(settings, make_embed_fn(embedding_client)) as store:
        service = MemoryService(store, ttl_days=settings.AGENT_LTM_TTL_DAYS)

        # 1) episode (deterministic)
        question = payload.get("standalone_question") or payload["original_question"]
        key = hashlib.sha1(question.strip().lower().encode()).hexdigest()[:16]
        await service.upsert(project_id, user_id, EPISODES, key, {
            "text": f"User previously asked: {question}",
            "question": question,
            "doc_ids": payload.get("citations", []),
            "question_type": payload.get("question_type"),
        })
        written["episode"] += 1

        # 2) profile facts (LLM, user's own words only)
        try:
            facts = await extractor.ainvoke({"message": payload["original_question"]})
        except Exception:
            logger.exception("profile extraction failed; episode already written")
            return {"signal": "partial", **written}

        for fact in facts.facts:
            text = fact.text.strip()
            if (not text or fact.sensitive or _looks_sensitive(text)
                    or fact.confidence < settings.AGENT_LTM_MIN_CONFIDENCE):
                continue
            similar = await service.find_similar(project_id, user_id, PROFILE, text)
            fact_key = (similar["key"] if similar and (similar["score"] or 0) >= _SIMILARITY_THRESHOLD
                        else uuid.uuid4().hex)
            await service.upsert(project_id, user_id, PROFILE, fact_key, {"text": text})
            written["profile"] += 1

    return {"signal": "memory_written", **written}


@celery_app.task(bind=True, name="tasks.memory.write_memory_task", max_retries=2)
def write_memory_task(self, payload: dict):
    try:
        return asyncio.run(_write_memory(payload))
    except Exception as exc:
        logger.exception("write_memory_task failed")
        raise self.retry(exc=exc, countdown=30)