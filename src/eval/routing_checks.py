"""
eval/routing_checks.py

Per-category checks, computed from the target's `outputs["steps"]` (node
names recorded while streaming the graph) instead of a hand-rolled trace.
Each returned value is numeric/bool so evaluators.py can emit it directly as
LangSmith feedback.
"""


def _nodes(steps: list[dict], name: str) -> list[dict]:
    return [s for s in steps if s.get("node") == name]


def check_info_not_found_routing(outputs: dict, expected_doc_ids: list[str]) -> dict:
    steps = outputs.get("steps", [])
    reached = bool(_nodes(steps, "info_not_found"))
    exhausted = outputs.get("iteration_count", 0) >= outputs.get("max_iterations", 3)
    return {
        "reached_info_not_found": reached,
        "verifier_calls": len(_nodes(steps, "verifier")),
        "gave_up_at_cap": reached and exhausted,
        # corpus DID have the answer but grounding still failed
        "false_info_not_found": reached and bool(expected_doc_ids),
    }


def check_completeness_sweep(outputs: dict) -> dict:
    sweeps = _nodes(outputs.get("steps", []), "completeness_sweep")
    return {
        "sweep_iterations": len(sweeps),
        "sweep_triggered": len(sweeps) >= 1,
        "sweep_looped_more_than_once": len(sweeps) >= 2,
        "final_chunk_count": len(outputs.get("retrieved_docs", [])),
    }


def check_conflict_clustering(outputs: dict) -> dict:
    docs = outputs.get("retrieved_docs", [])
    with_recency = sum(1 for d in docs if (d.get("metadata") or {}).get("last_modified"))
    return {
        "distinct_docs_retrieved": len({d["doc_id"] for d in docs}),
        "recency_metadata_present": with_recency / max(len(docs), 1),
    }


def check_constraint_filter(outputs: dict) -> dict:
    entries = _nodes(outputs.get("steps", []), "constraint_filter")
    if not entries:
        return {"filter_triggered": False}
    filters = entries[-1].get("retrieval_filters") or {}
    return {
        "filter_triggered": True,
        "extracted_nonempty_filter": bool(filters.get("source_types") or filters.get("other_equals")),
        "fell_back_to_unfiltered": bool(filters.get("fallback_unfiltered", False)),
    }


def run_category_checks(question_type: str, outputs: dict, expected_doc_ids: list[str]) -> dict:
    if question_type == "info_not_found":
        return check_info_not_found_routing(outputs, expected_doc_ids)
    if question_type == "completeness":
        return check_completeness_sweep(outputs)
    if question_type == "conflicting_info":
        return check_conflict_clustering(outputs)
    if question_type == "constrained":
        return check_constraint_filter(outputs)
    return {}