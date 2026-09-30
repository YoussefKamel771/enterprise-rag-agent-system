"""
eval/evaluators.py

LangSmith evaluators. Each returns {"results": [{"key", "score", "comment"?}]},
so one function can emit several feedback keys. LangSmith stores these per run
and aggregates them per experiment -- this replaces eval_runs, report.py and
telemetry.py. Latency, token counts and cost are recorded by LangSmith itself
on every run.

Evaluators skip (return no feedback) when the target crashed (outputs empty):
an infra failure is not a wrong answer.
"""

from stores.agents.supervisor import ROUTE_TABLE

from .judge import judge_one
from .retrieval_metrics import compute_retrieval_metrics
from .routing_checks import run_category_checks


def _num(v) -> float:
    return float(v)


def build_evaluators(judge_chain) -> list:
    async def judge_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
        if not outputs:
            return {"results": []}
        qtype = reference_outputs["question_type"]
        verdict = await judge_one(
            judge_chain,
            inputs["question"],
            reference_outputs["gold_answer"],
            reference_outputs["answer_facts"],
            outputs.get("answer", ""),
            qtype,
        )
        completeness = verdict.completeness_pct / 100.0
        results = [
            {"key": "correctness", "score": verdict.correctness,
             "comment": verdict.correctness_reasoning},
            {"key": "completeness", "score": completeness},
            # Overall Score = mean(correctness * completeness); LangSmith averages this key
            {"key": "overall", "score": verdict.correctness * completeness},
        ]
        if qtype == "conflicting_info" and verdict.acknowledges_contradiction is not None:
            results.append({"key": "acknowledges_contradiction",
                            "score": _num(verdict.acknowledges_contradiction)})
        return {"results": results}

    def retrieval_evaluator(outputs: dict, reference_outputs: dict):
        if not outputs:
            return {"results": []}
        m = compute_retrieval_metrics(outputs, reference_outputs["expected_doc_ids"])
        results = [
            {"key": "invalid_extra_docs_raw", "score": m["raw_invalid_extra"]},
            {"key": "invalid_extra_docs_cited", "score": m["cited_invalid_extra"]},
        ]
        if m["raw_recall"] is not None:
            results.append({"key": "document_recall_raw", "score": m["raw_recall"]})
            results.append({"key": "document_recall_cited", "score": m["cited_recall"]})
        return {"results": results}

    def routing_evaluator(outputs: dict, reference_outputs: dict):
        if not outputs:
            return {"results": []}
        true_type = reference_outputs["question_type"]
        pred_type = outputs.get("question_type")
        results = [
            {"key": "classification_correct", "score": _num(pred_type == true_type)},
            # ~1.0 by construction; below that is a code bug, not a model issue
            {"key": "routing_correct",
             "score": _num(outputs.get("active_agent") == ROUTE_TABLE.get(pred_type))},
        ]
        checks = run_category_checks(true_type, outputs, reference_outputs["expected_doc_ids"])
        results += [{"key": k, "score": _num(v)} for k, v in checks.items() if v is not None]
        return {"results": results}

    return [judge_evaluator, retrieval_evaluator, routing_evaluator]