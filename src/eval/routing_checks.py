"""
eval/routing_checks.py -- Section 3

Validates that questions reach the graph nodes ROUTE_TABLE says they should,
and runs the four per-category checks named in the evaluation framework:
Info Not Found -> info_not_found, Completeness -> completeness_sweep
convergence, Conflicting Info -> conflict_resolver clustering, Constrained ->
constraint_filter extraction quality.
"""

from stores.agents.supervisor import ROUTE_TABLE


def routing_accuracy(results: list[dict], ground_truth: dict[str, str]) -> dict:
    """results: [{question_id, question_type (predicted), active_agent}, ...]
    ground_truth: {question_id: question_type (from the parquet)}"""
    classification_correct = 0
    routing_correct = 0
    confusion: dict[tuple, int] = {}

    for r in results:
        true_type = ground_truth.get(r["question_id"])
        pred_type = r.get("question_type")
        confusion[(true_type, pred_type)] = confusion.get((true_type, pred_type), 0) + 1
        if pred_type == true_type:
            classification_correct += 1
        if r.get("active_agent") == ROUTE_TABLE.get(pred_type):
            routing_correct += 1

    n = len(results) or 1
    return {
        "classification_accuracy": classification_correct / n,
        "routing_correctness": routing_correct / n,  # expect ~1.0; below that is a code bug, not a model-quality issue
        "confusion_matrix": {f"{t}->{p}": c for (t, p), c in confusion.items()},
    }


def check_info_not_found_routing(trace: list[dict], final_state: dict, expected_doc_ids: list[str]) -> dict:
    reached_info_not_found = any(e.get("node") == "info_not_found" for e in trace)
    verifier_entries = [e for e in trace if e.get("node") == "verifier"]
    exhausted_iterations = final_state.get("iteration_count", 0) >= final_state.get("max_iterations", 3)
    return {
        "reached_info_not_found": reached_info_not_found,
        "verifier_calls": len(verifier_entries),
        "gave_up_at_cap": reached_info_not_found and exhausted_iterations,
        # a "false info_not_found" -- the corpus DID have the answer but grounding still failed
        "false_info_not_found": reached_info_not_found and bool(expected_doc_ids),
    }


def check_completeness_sweep(trace: list[dict], expected_doc_ids: list[str]) -> dict:
    sweep_entries = [e for e in trace if e.get("node") == "completeness_sweep"]
    router_entries = [e for e in trace if e.get("node") == "router:completeness_sweep"]
    final_chunk_count = (sweep_entries[-1].get("update") or {}).get("chunks", 0) if sweep_entries else 0
    return {
        "sweep_iterations": len(sweep_entries),
        "triggered": len(sweep_entries) >= 1,
        "looped_more_than_once": len(sweep_entries) >= 2,
        "final_recall_vs_expected": final_chunk_count / max(len(expected_doc_ids), 1),
        "stop_reason": router_entries[-1].get("decision") if router_entries else None,
    }


def check_conflict_clustering(retrieved_chunks: list) -> dict:
    doc_ids = {getattr(d, "doc_id", None) for d in retrieved_chunks}
    doc_ids.discard(None)
    last_modified = [
        (getattr(d, "metadata", {}) or {}).get("last_modified")
        for d in retrieved_chunks
        if (getattr(d, "metadata", {}) or {}).get("last_modified")
    ]
    n = len(retrieved_chunks) or 1
    return {
        "distinct_docs_retrieved": len(doc_ids),
        "recency_metadata_present": len(last_modified) / n,
    }


def check_constraint_filter(trace: list[dict]) -> dict:
    filter_entries = [e for e in trace if e.get("node") == "constraint_filter"]
    if not filter_entries:
        return {"triggered": False}
    update = filter_entries[-1].get("update") or {}
    filters = update.get("retrieval_filters") or {}
    return {
        "triggered": True,
        "extracted_nonempty_filter": bool(filters.get("source_types") or filters.get("other_equals")),
        "fell_back_to_unfiltered": bool(filters.get("fallback_unfiltered", False)),
    }


# ---------------------------------------------------------------------------
# Dispatch: run the category-specific check that applies to this question,
# so run_agent.py doesn't need a long if/elif chain of its own.
# ---------------------------------------------------------------------------

def run_category_checks(question_type: str, trace: list[dict], final_state: dict,
                        expected_doc_ids: list[str]) -> dict:
    checks: dict = {}
    if question_type == "info_not_found":
        checks.update(check_info_not_found_routing(trace, final_state, expected_doc_ids))
    if question_type == "completeness":
        checks.update(check_completeness_sweep(trace, expected_doc_ids))
    if question_type == "conflicting_info":
        checks.update(check_conflict_clustering(final_state.get("retrieved_chunks", [])))
    if question_type == "constrained":
        checks.update(check_constraint_filter(trace))
    return checks