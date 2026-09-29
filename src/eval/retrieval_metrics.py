"""
eval/retrieval_metrics.py -- Section 4.1

    Document Recall_i    = | Retrieved_i ∩ Expected_i | / | Expected_i |
    Invalid Extra Docs_i = | Retrieved_i \\ Expected_i |

Two definitions of Retrieved_i are computed and reported separately:
- "raw"  : every distinct doc_id across retrieved_chunks in the final state
           (retrieval subsystem's ceiling).
- "cited": result["citations"] (what actually grounded the final answer).
A large raw-vs-cited gap on a category means retrieval is fine but
generation/verification isn't using what it found.
"""

from typing import Optional


def document_recall(retrieved_doc_ids: set[str], expected_doc_ids: list[str]) -> Optional[float]:
    if not expected_doc_ids:
        return None  # no ground-truth docs for this question -- exclude from the mean, don't treat as 0
    expected = set(expected_doc_ids)
    return len(retrieved_doc_ids & expected) / len(expected)


def invalid_extra_docs(retrieved_doc_ids: set[str], expected_doc_ids: list[str]) -> int:
    return len(retrieved_doc_ids - set(expected_doc_ids))


def compute_retrieval_metrics(final_state: dict, expected_doc_ids: list[str]) -> dict:
    raw_ids = {getattr(d, "doc_id", None) for d in final_state.get("retrieved_chunks", [])}
    raw_ids.discard(None)
    cited_ids = set(final_state.get("citations", []) or [])

    return {
        "raw_recall": document_recall(raw_ids, expected_doc_ids),
        "raw_invalid_extra": invalid_extra_docs(raw_ids, expected_doc_ids),
        "cited_recall": document_recall(cited_ids, expected_doc_ids),
        "cited_invalid_extra": invalid_extra_docs(cited_ids, expected_doc_ids),
        "raw_doc_count": len(raw_ids),
        "cited_doc_count": len(cited_ids),
    }


def aggregate_retrieval_metrics(per_question: list[dict]) -> dict:
    def _mean(key):
        vals = [q[key] for q in per_question if q.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "mean_raw_recall": _mean("raw_recall"),
        "mean_cited_recall": _mean("cited_recall"),
        "mean_raw_invalid_extra": _mean("raw_invalid_extra"),
        "mean_cited_invalid_extra": _mean("cited_invalid_extra"),
        "n_excluded_no_ground_truth": sum(1 for q in per_question if q.get("raw_recall") is None),
    }