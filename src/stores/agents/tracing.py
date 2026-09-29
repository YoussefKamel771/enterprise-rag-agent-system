"""
stores/agents/tracing.py

Observability for the agent graph, in three layers -- none of which require
touching node logic:

1. `TracedStateGraph` -- a StateGraph subclass that wraps every node and every
   conditional-edge router as it is registered. For each node you get:
   what state it received, how long it took, what it returned (summarised:
   chunk counts + top scores instead of full texts), and where it routed next
   (`Command.goto`, including `Send` fan-outs). graph.py just swaps
   `StateGraph(RAGState)` for `TracedStateGraph(RAGState)`.

2. `AgentLogCallbackHandler` -- a LangChain callback attached to the
   classifier chat model. Logs every LLM call made by the Supervisor /
   Decomposer / Filter / Verifier / Reformulator nodes: latency, token usage
   and the (structured) output. It is attached at the *model* level
   (ChatOpenAI(callbacks=[...])) rather than via the invoke config, so it
   fires regardless of how LangGraph propagates run config into nodes.

3. `run_context` -- per-request correlation id (`[thread:abc123]` prefix on
   every line, so concurrent requests and parallel `Send` branches can be
   told apart) and an optional in-memory trace list that the endpoint can
   return when the request sets `debug: true`.

Verbosity is controlled by settings.AGENT_LOG_LEVEL (logger "uvicorn.agent",
a child of the "uvicorn" logger the rest of the app already uses):
    INFO  -> one line per node in/out, router decisions, LLM call summaries
    DEBUG -> additionally: LLM input messages, chunk text previews
"""

import functools
import json
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackHandler
from langgraph.graph import StateGraph

# ---------------------------------------------------------------------------
# Context (correlation id, current node, optional trace collector)
# ---------------------------------------------------------------------------
# ContextVars are copied into every asyncio task LangGraph spawns (including
# parallel Send branches), so the same run id / node name follow the work
# without being threaded through node signatures. The trace list is shared
# by reference, so appends from parallel branches all land in one list.

_run_id_var: ContextVar[str] = ContextVar("agent_run_id", default="-")
_node_var: ContextVar[str] = ContextVar("agent_node", default="-")
_trace_var: ContextVar[Optional[list]] = ContextVar("agent_trace", default=None)

_PREVIEW_CHARS = 300


class _RunAdapter(logging.LoggerAdapter):
    """Prefixes every line with `[run_id] (node)` from the current context."""

    def process(self, msg, kwargs):
        return f"[{_run_id_var.get()}] ({_node_var.get()}) {msg}", kwargs


# Use this from nodes too: `from .tracing import agent_logger as logger`
agent_logger = _RunAdapter(logging.getLogger("uvicorn.agent"), {})


def configure_tracing(level: str = "INFO", preview_chars: int = 300) -> None:
    """Called once from build_rag_graph with values from Settings."""
    global _PREVIEW_CHARS
    _PREVIEW_CHARS = preview_chars
    logging.getLogger("uvicorn.agent").setLevel(getattr(logging, level.upper(), logging.INFO))


@contextmanager
def run_context(thread_id: str, debug: bool = False):
    """Sets the correlation id (+ trace list when debug) for one graph run."""
    run_id = f"{thread_id}:{uuid.uuid4().hex[:6]}"
    trace: Optional[list] = [] if debug else None
    t_run, t_trace = _run_id_var.set(run_id), _trace_var.set(trace)
    try:
        yield run_id, trace
    finally:
        _run_id_var.reset(t_run)
        _trace_var.reset(t_trace)


# ---------------------------------------------------------------------------
# Summarising state / updates so logs stay readable
# ---------------------------------------------------------------------------

def _preview(text: Any, n: Optional[int] = None) -> str:
    n = n or _PREVIEW_CHARS
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[:n] + "…"


def _summarize(value: Any) -> Any:
    """JSON-safe, size-bounded view of any state value."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _preview(value)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        # retrieved chunks: count + top-5 (doc_id, score, source) instead of full text
        if items and hasattr(items[0], "doc_id") and hasattr(items[0], "score"):
            show_text = agent_logger.isEnabledFor(logging.DEBUG)
            top = []
            for d in items[:5]:
                entry = {
                    "doc": _preview(d.doc_id, 14),
                    "score": round(float(d.score), 4),
                    "src": getattr(d, "source_type", None),
                }
                if show_text:
                    entry["text"] = _preview(d.text, 120)
                top.append(entry)
            return {"n": len(items), "top": top}
        return {"n": len(items), "items": [_summarize(i) for i in items[:8]]}
    if isinstance(value, dict):
        return {str(k): _summarize(v) for k, v in list(value.items())[:12]}
    if hasattr(value, "model_dump"):
        return _summarize(value.model_dump())
    return _preview(repr(value))


def _summarize_state(state: dict) -> dict:
    """What a node *received* -- deliberately a fixed, compact subset."""
    return {
        "question": _preview(state.get("question", ""), 160),
        "type": state.get("question_type"),
        "chunks": len(state.get("retrieved_chunks", []) or []),
        "iter": state.get("iteration_count"),
        "attempts": state.get("retrieval_attempts"),
    }


def _goto_repr(goto: Any) -> Any:
    if goto is None:
        return None
    if isinstance(goto, (list, tuple)):
        return [_goto_repr(g) for g in goto]
    if hasattr(goto, "node") and hasattr(goto, "arg"):      # langgraph.types.Send
        return f"Send({goto.node})"
    return str(goto)


def _fmt(obj: Any) -> str:
    return json.dumps(obj, default=str, ensure_ascii=False)


def _record(event: dict) -> None:
    trace = _trace_var.get()
    if trace is not None:
        event["seq"] = len(trace) + 1
        trace.append(event)


# ---------------------------------------------------------------------------
# Node / router wrappers
# ---------------------------------------------------------------------------

def traced_node(name: str):
    """Wraps an async node. Handles both return styles: a partial-state dict
    and `Command(update=..., goto=...)`."""

    def decorator(fn):
        @functools.wraps(fn)          # keeps __wrapped__ / annotations so LangGraph
        async def wrapper(state, *args, **kwargs):   # still sees the original signature
            token = _node_var.set(name)
            started = time.perf_counter()
            input_view = _summarize_state(state)
            agent_logger.info("▶ in=%s", _fmt(input_view))
            try:
                result = await fn(state, *args, **kwargs)
            except Exception as exc:
                ms = round((time.perf_counter() - started) * 1000)
                agent_logger.exception("✖ node crashed after %dms", ms)
                _record({"node": name, "ms": ms, "input": input_view, "error": repr(exc)})
                _node_var.reset(token)
                raise

            ms = round((time.perf_counter() - started) * 1000)
            goto = None
            update = result
            if hasattr(result, "update") and hasattr(result, "goto"):   # Command
                update, goto = (result.update or {}), _goto_repr(result.goto)
            update_view = _summarize(update or {})

            level = logging.WARNING if (update or {}).get("error") else logging.INFO
            agent_logger.log(
                level, "◀ %dms%s out=%s", ms,
                f" goto={_fmt(goto)}" if goto is not None else "",
                _fmt(update_view),
            )
            _record({"node": name, "ms": ms, "input": input_view,
                     "goto": goto, "update": update_view})
            _node_var.reset(token)
            return result
        return wrapper
    return decorator


def traced_router(source: str):
    """Wraps a conditional-edge routing function (sync, returns a node name)."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            token = _node_var.set(f"router:{source}")
            decision = fn(state, *args, **kwargs)
            context = {
                "grounding_passed": state.get("grounding_passed"),
                "iter": state.get("iteration_count"),
                "max_iter": state.get("max_iterations"),
                "chunks": len(state.get("retrieved_chunks", []) or []),
                "attempts": state.get("retrieval_attempts"),
            }
            agent_logger.info("⑂ decision=%s | %s", decision, _fmt(context))
            _record({"node": f"router:{source}", "decision": decision, "context": context})
            _node_var.reset(token)
            return decision
        return wrapper
    return decorator


class TracedStateGraph(StateGraph):
    """Drop-in StateGraph that wraps nodes/routers with the tracers above."""

    def add_node(self, node, action=None, **kwargs):
        if isinstance(node, str) and callable(action):
            action = traced_node(node)(action)
        return super().add_node(node, action, **kwargs)

    def add_conditional_edges(self, source, path, path_map=None, **kwargs):
        if callable(path):
            path = traced_router(source)(path)
        return super().add_conditional_edges(source, path, path_map, **kwargs)


# ---------------------------------------------------------------------------
# LLM-call logging (LangChain callback)
# ---------------------------------------------------------------------------

class AgentLogCallbackHandler(AsyncCallbackHandler):
    """Logs each chat-model call made through the classifier LLM. The node
    name in the prefix comes from `_node_var`, set by `traced_node`."""

    def __init__(self):
        self._started: dict = {}

    async def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self._started[run_id] = time.perf_counter()
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or "?"
        flat = messages[0] if messages else []
        agent_logger.info("llm▶ model=%s messages=%d", model, len(flat))
        if agent_logger.isEnabledFor(logging.DEBUG):
            for m in flat:
                agent_logger.debug("llm▶ [%s] %s", getattr(m, "type", "?"), _preview(m.content, 600))

    async def on_llm_end(self, response, *, run_id, **kwargs):
        started = self._started.pop(run_id, None)
        ms = round((time.perf_counter() - started) * 1000) if started else -1
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        msg = getattr(gen, "message", None)
        tool_calls = getattr(msg, "tool_calls", None) or []
        usage = getattr(msg, "usage_metadata", None)
        # structured output arrives as tool-call args (or plain content for json modes)
        out = tool_calls[0].get("args") if tool_calls else (getattr(msg, "content", None) or getattr(gen, "text", ""))
        agent_logger.info("llm◀ %dms tokens=%s out=%s", ms, _fmt(usage) if usage else "n/a", _fmt(_summarize(out)))
        # Feed token usage into the same per-run trace list used by traced_node/
        # traced_router, so eval/telemetry.py's token_usage() can read it back
        # without a second instrumentation path. usage_metadata is a dict-like
        # with input_tokens/output_tokens on the providers that report it.
        _record({"node": f"llm:{_node_var.get()}", "ms": ms, "usage": dict(usage) if usage else {}})

    async def on_llm_error(self, error, *, run_id, **kwargs):
        started = self._started.pop(run_id, None)
        ms = round((time.perf_counter() - started) * 1000) if started else -1
        agent_logger.error("llm✖ %dms error=%r", ms, error)