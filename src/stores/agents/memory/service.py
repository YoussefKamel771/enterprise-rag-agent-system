"""
stores/agents/memory/service.py

Thin wrapper around the LangGraph store so nodes / controllers / Celery never
touch the raw store (same idea as RetrievalGateway).

Namespaces:  (project_id, user_id, "profile" | "episodes")
Values:      {"text": ..., "kind": ..., "created_at", "updated_at", "expires_at", ...}
`text` is the indexed field. Expiry is stored in the value and filtered on read
(version-independent); swap in the store's native TTL if your version has it.
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

PROFILE = "profile"
EPISODES = "episodes"
KINDS = (PROFILE, EPISODES)

# Namespace labels may not contain "."; also escape whitespace and "/".
_UNSAFE = re.compile(r"[.\s/]")


def _label(value) -> str:
    return _UNSAFE.sub(lambda m: f"%{ord(m.group()):02X}", str(value))


def _alive(value: dict) -> bool:
    exp = value.get("expires_at")
    if not exp:
        return True
    try:
        return datetime.fromisoformat(exp) > datetime.now(timezone.utc)
    except ValueError:
        return True


def _to_dict(item, kind: str) -> Dict[str, Any]:
    return {
        "key": item.key,
        "kind": kind,
        "text": item.value.get("text", ""),
        "doc_ids": item.value.get("doc_ids", []),
        "score": getattr(item, "score", None),
    }


class MemoryService:
    def __init__(self, store, ttl_days: int = 180):
        self._store = store
        self._ttl_days = ttl_days

    def _ns(self, project_id, user_id, kind: Optional[str] = None) -> tuple:
        base = (_label(project_id), _label(user_id))
        return base + (kind,) if kind else base

    # ------------------------------------------------------------------ read
    async def recall(self, project_id, user_id, query: str,
                     top_k: int = 4, min_score: float = 0.35) -> List[Dict[str, Any]]:
        """Profile facts (small, always relevant) + the episodes most similar to `query`."""
        out: List[Dict[str, Any]] = []

        profile = await self._store.asearch(self._ns(project_id, user_id, PROFILE), limit=3)
        out += [_to_dict(i, PROFILE) for i in profile if _alive(i.value)]

        episodes = await self._store.asearch(
            self._ns(project_id, user_id, EPISODES), query=query, limit=top_k * 2)
        out += [_to_dict(i, EPISODES) for i in episodes
                if (i.score or 0) >= min_score and _alive(i.value)][:top_k]
        return out

    async def find_similar(self, project_id, user_id, kind: str, text: str) -> Optional[Dict[str, Any]]:
        items = await self._store.asearch(self._ns(project_id, user_id, kind), query=text, limit=1)
        if not items or not _alive(items[0].value):
            return None
        return _to_dict(items[0], kind)

    async def _iter_items(self, prefix: tuple, page: int = 100) -> AsyncIterator[Any]:
        offset = 0
        while True:
            batch = await self._store.asearch(prefix, limit=page, offset=offset)
            if not batch:
                return
            for item in batch:
                yield item
            offset += page

    async def list_all(self, project_id, user_id) -> List[Dict[str, Any]]:
        out = []
        for kind in KINDS:
            async for item in self._iter_items(self._ns(project_id, user_id, kind)):
                d = _to_dict(item, kind)
                d["expires_at"] = item.value.get("expires_at")
                out.append(d)
        return out

    # ----------------------------------------------------------------- write
    async def upsert(self, project_id, user_id, kind: str, key: str, value: dict) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            **value,
            "kind": kind,
            "updated_at": now.isoformat(),
            "expires_at": (now + timedelta(days=self._ttl_days)).isoformat(),
        }
        payload.setdefault("created_at", now.isoformat())
        await self._store.aput(self._ns(project_id, user_id, kind), key, payload)

    async def delete(self, project_id, user_id, kind: str, key: str) -> None:
        await self._store.adelete(self._ns(project_id, user_id, kind), key)

    async def delete_all(self, project_id, user_id) -> int:
        n = 0
        for kind in KINDS:
            ns = self._ns(project_id, user_id, kind)
            keys = [i.key async for i in self._iter_items(ns)]
            for k in keys:
                await self._store.adelete(ns, k)
                n += 1
        return n    