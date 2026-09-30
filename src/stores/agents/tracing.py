import functools
import inspect
import logging
import time

from langgraph.types import Command, Send

logger = logging.getLogger("uvicorn")

_INPUT_KEYS = (
    "question", "question_type", "active_agent", "retrieval_attempts",
    "iteration_count", "max_iterations", "grounding_passed",
)


def _short(v, n=200):
    if isinstance(v, str):
        return v if len(v) <= n else f"{v[:n]}...(+{len(v) - n} chars)"
    return v


def summarize(v):
    """Compact, log-safe view of a state value (never dumps full chunk text)."""
    if isinstance(v, list) and v and hasattr(v[0], "chunk_id"):
        return {
            "n": len(v),
            "chunks": [
                (d.chunk_id, d.doc_id, round(getattr(d, "score", 0) or 0, 3))
                for d in v[:15]
            ],
        }
    if isinstance(v, list):
        return [summarize(x) for x in v[:8]] + ([f"...(+{len(v) - 8})"] if len(v) > 8 else [])
    if isinstance(v, dict):
        return {k: summarize(x) for k, x in v.items()}
    return _short(v)


def _summarize_goto(goto):
    items = goto if isinstance(goto, (list, tuple)) else [goto]
    out = []
    for g in items:
        if isinstance(g, Send):
            out.append(f"Send({g.node}, q={_short(g.arg.get('question'), 80)!r})")
        else:
            out.append(str(g))
    return out


def _log_enter(label, state):
    snap = {k: _short(state.get(k)) for k in _INPUT_KEYS if k in state}
    snap["n_chunks"] = len(state.get("retrieved_chunks", []))
    snap["n_excluded"] = len(state.get("excluded_chunk_ids", []))
    logger.info("[agent] >> %s | in=%s", label, snap)


def _log_exit(label, out, t0):
    ms = (time.perf_counter() - t0) * 1000
    if isinstance(out, Command):
        logger.info("[agent] << %s (%.0fms) | goto=%s | update=%s",
                    label, ms, _summarize_goto(out.goto), summarize(out.update or {}))
    elif isinstance(out, dict):
        logger.info("[agent] << %s (%.0fms) | update=%s", label, ms, summarize(out))
    else:  # router functions return a plain string
        logger.info("[agent] -- %s (%.0fms) | route -> %s", label, ms, out)


def traced(name: str | None = None):
    def deco(fn):
        label = name or fn.__name__

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(state, *a, **kw):
                _log_enter(label, state)
                t0 = time.perf_counter()
                try:
                    out = await fn(state, *a, **kw)
                except Exception:
                    logger.exception("[agent] !! %s raised after %.0fms",
                                     label, (time.perf_counter() - t0) * 1000)
                    raise
                _log_exit(label, out, t0)
                return out
            return awrapper

        @functools.wraps(fn)
        def wrapper(state, *a, **kw):
            t0 = time.perf_counter()
            out = fn(state, *a, **kw)
            _log_exit(label, out, t0)
            return out
        return wrapper
    return deco