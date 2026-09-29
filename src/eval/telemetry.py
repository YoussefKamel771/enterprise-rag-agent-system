"""
eval/telemetry.py -- Section 5.1

Everything here reads the `trace` list produced by
`stores.agents.tracing.run_context(..., debug=True)`. No new instrumentation
required beyond the one-line addition already made to
`AgentLogCallbackHandler.on_llm_end` (records token usage into the same
trace list under node name `llm:<node>`).
"""

from typing import Any


def extract_telemetry(trace: list[dict], final_state: dict) -> dict:
    node_entries = [e for e in trace if "node" in e and not e["node"].startswith(("router:", "llm:"))]
    router_entries = [e for e in trace if e.get("node", "").startswith("router:")]

    latencies: dict[str, list[int]] = {}
    for e in node_entries:
        latencies.setdefault(e["node"], []).append(e.get("ms", 0))

    return {
        "total_latency_ms": sum(e.get("ms", 0) for e in node_entries),
        "node_count": len(node_entries),
        "recursion_depth": len(trace),  # proxy for graph super-steps taken
        "per_node_latency_ms": {
            n: {"calls": len(v), "mean": sum(v) / len(v), "max": max(v)}
            for n, v in latencies.items()
        },
        "iteration_count": final_state.get("iteration_count", 0),
        "retrieval_attempts": final_state.get("retrieval_attempts", 0),
        "routing_decisions": [r.get("decision") for r in router_entries],
        "hit_recursion_cap": final_state.get("iteration_count", 0) >= final_state.get("max_iterations", 3),
    }


def token_usage(trace: list[dict]) -> dict:
    llm_entries = [e for e in trace if e.get("node", "").startswith("llm:")]
    totals = {"input_tokens": 0, "output_tokens": 0}
    for e in llm_entries:
        usage = e.get("usage") or {}
        totals["input_tokens"] += usage.get("input_tokens", 0) or 0
        totals["output_tokens"] += usage.get("output_tokens", 0) or 0
    return {**totals, "llm_calls": len(llm_entries)}


def extract_draft_answers(trace: list[dict]) -> list[str]:
    """Every draft_answer the generator produced, one per reformulation
    iteration -- not just the final one. Needed for the "false rejection by
    Verifier" failure mode (framework Section 5.2): judging only the final
    draft can't tell you whether an EARLIER draft was actually correct and
    got discarded by an overly strict grounding check.

    Relies on settings.EVAL_PREVIEW_CHARS being set wide enough (via
    `tracing.configure_tracing`) that these aren't truncated -- see
    run_agent.py, which sets this before any benchmark run starts.
    """
    drafts = []
    for e in trace:
        if e.get("node") == "generator":
            draft = (e.get("update") or {}).get("draft_answer")
            if draft:
                drafts.append(draft)
    return drafts


def summarize_telemetry_across_questions(per_question_telemetry: list[dict]) -> dict:
    if not per_question_telemetry:
        return {}
    n = len(per_question_telemetry)
    return {
        "mean_latency_ms": sum(t["total_latency_ms"] for t in per_question_telemetry) / n,
        "mean_llm_calls": sum(t.get("llm_calls", 0) for t in per_question_telemetry) / n,
        "mean_tokens": sum(
            t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in per_question_telemetry
        ) / n,
        "reformulation_rate": sum(1 for t in per_question_telemetry if t.get("iteration_count", 0) > 0) / n,
        "recursion_cap_hit_rate": sum(1 for t in per_question_telemetry if t.get("hit_recursion_cap")) / n,
    }