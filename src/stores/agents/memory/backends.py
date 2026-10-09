"""
stores/agents/memory/backends.py

Builds the persistence resources LangGraph needs:
  * a checkpointer  (short-term memory, per thread)
  * a store         (long-term memory, pgvector-indexed)

LangGraph's Postgres packages use psycopg3, NOT asyncpg, so they get their own
connection pool (same credentials, plain postgresql:// URL).

Heavy imports are lazy so the app still starts without psycopg / the postgres
checkpoint package when memory is disabled.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import quote_plus

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from langgraph.store.postgres import AsyncPostgresStore
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger("uvicorn")


def postgres_conninfo(settings) -> str:
    return (
        f"postgresql://{quote_plus(settings.POSTGRES_USERNAME)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )


def make_embed_fn(embedding_client):
    """Async embed function for the store index. ALWAYS passes a list: OpenAIProvider
    returns a flat vector for a bare string but a list of vectors for a list."""
    async def embed(texts):
        vectors = await embedding_client.embed_text(
            text=list(texts), document_type="document", batch_size=16)
        if not vectors:
            raise RuntimeError("embedding client returned no vectors")
        return vectors
    return embed


async def _open_pool(settings, max_size: int = 10):
    
    pool = AsyncConnectionPool(
        conninfo=postgres_conninfo(settings),
        max_size=max_size,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
    await pool.open()
    return pool


def _make_store(pool, settings, embed_fn):

    return AsyncPostgresStore(
        pool,
        index={"dims": settings.EMBEDDING_MODEL_SIZE, "embed": embed_fn, "fields": ["text"]},
    )


@dataclass
class MemoryResources:
    checkpointer: Optional[Any] = None
    store: Optional[Any] = None
    pool: Optional[Any] = None

    async def close(self):
        if self.pool is not None:
            await self.pool.close()


async def build_memory_resources(settings, embedding_client) -> MemoryResources:
    """Always returns an object; unused parts are None."""
    backend = (settings.AGENT_CHECKPOINTER_BACKEND or "none").lower()
    need_pool = backend == "postgres" or settings.AGENT_LTM_ENABLED
    pool = await _open_pool(settings) if need_pool else None

    checkpointer = None
    if backend == "postgres":
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
    elif backend == "memory":
        
        checkpointer = MemorySaver()
    elif backend != "none":
        raise ValueError(f"Unsupported AGENT_CHECKPOINTER_BACKEND: {backend}")

    store = None
    if settings.AGENT_LTM_ENABLED:
        store = _make_store(pool, settings, make_embed_fn(embedding_client))
        await store.setup()

    logger.info("[memory] checkpointer=%s ltm_store=%s", backend, store is not None)
    return MemoryResources(checkpointer=checkpointer, store=store, pool=pool)


@asynccontextmanager
async def store_session(settings, embed_fn):
    """Short-lived store for Celery tasks. Every task runs under asyncio.run (a new
    event loop per call), so a module-level async pool would break: open and close here."""
    pool = await _open_pool(settings, max_size=3)
    try:
        yield _make_store(pool, settings, embed_fn)
    finally:
        await pool.close()