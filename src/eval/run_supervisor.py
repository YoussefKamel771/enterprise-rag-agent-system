"""
eval/run_supervisor.py

Evaluate the supervisor's classification step alone: one cheap LLM call per
question, no retrieval, no generation, no judge. Cheap enough to run on all
500 questions every time you touch SUPERVISOR_SYSTEM_PROMPT or the model.

    python -m eval.run_supervisor --run-tag sup-v1
    python -m eval.run_supervisor --run-tag sup-v1 --repetitions 3   # consistency

Per-run feedback:
    classification_correct   exact question_type match
    agent_route_correct      predicted category routes to the same agent as the true one
                             (basic/semantic/miscellaneous confusion is harmless)
    classification_failed    structured output could not be parsed
    multi_doc_correct        requires_multi_doc vs (len(expected_doc_ids) > 1)
    source_estimate_recall   estimated_source_types vs the question's source_types
Experiment-level (summary) feedback:
    accuracy, macro_f1, precision/recall/f1 per category,
    mean_confidence_correct / mean_confidence_wrong
"""

import argparse
import asyncio
import logging
from collections import Counter
import os

from langsmith import aevaluate

from helpers.config import get_settings
from stores.agents.graph import _build_classifier_llm
from stores.agents.supervisor import ROUTE_TABLE, build_classification_chain
from .examples import load_examples
from .run_benchmark import _git_commit

logger = logging.getLogger("uvicorn.eval")

os.environ["LANGCHAIN_TRACING_V2"] = "false"

def make_target(chain):
    async def target(inputs: dict) -> dict:
        try:
            r = await chain.ainvoke({"question": inputs["question"]})
            return {**r.model_dump(), "destination": ROUTE_TABLE.get(r.question_type), "failed": False}
        except Exception as exc:
            # A parse failure IS a supervisor failure (in production it falls back to
            # direct_retriever), so score it instead of letting it vanish as an infra error.
            logger.warning("classification failed: %s", str(exc)[:200])
            return {"question_type": None, "destination": None, "failed": True, "error": str(exc)[:300]}
    return target


# ---------------------------------------------------------------------------
# Per-run evaluators
# ---------------------------------------------------------------------------

def supervisor_evaluator(outputs: dict, reference_outputs: dict, example):
    if not outputs:
        return {"results": []}
    true_type = reference_outputs["question_type"]
    pred = outputs.get("question_type")
    results = [
        {"key": "classification_correct", "score": float(pred == true_type)},
        {"key": "classification_failed", "score": float(bool(outputs.get("failed")))},
        {"key": "agent_route_correct",
         "score": float(pred is not None and ROUTE_TABLE.get(pred) == ROUTE_TABLE.get(true_type))},
    ]
    if not outputs.get("failed"):
        multi_true = len(reference_outputs.get("expected_doc_ids") or []) > 1
        if reference_outputs.get("expected_doc_ids"):
            results.append({"key": "multi_doc_correct",
                            "score": float(bool(outputs.get("requires_multi_doc")) == multi_true)})
        expected_src = {s.lower() for s in (example.metadata or {}).get("source_types") or []}
        est_src = {s.lower() for s in outputs.get("estimated_source_types") or []}
        if expected_src:
            results.append({"key": "source_estimate_recall",
                            "score": len(est_src & expected_src) / len(expected_src)})
    return {"results": results}


# ---------------------------------------------------------------------------
# Summary evaluator: per-class P/R/F1 + calibration
# ---------------------------------------------------------------------------

def classification_summary(runs, examples):
    pairs = []
    for r, e in zip(runs, examples):
        true = (e.outputs or {}).get("question_type")
        out = r.outputs or {}
        pairs.append((true, out.get("question_type"), out.get("confidence")))

    classes = sorted({t for t, _, _ in pairs if t})
    results = [{"key": "accuracy", "score": sum(t == p for t, p, _ in pairs) / max(len(pairs), 1)}]
    f1s = []
    for c in classes:
        tp = sum(t == c and p == c for t, p, _ in pairs)
        fp = sum(t != c and p == c for t, p, _ in pairs)
        fn = sum(t == c and p != c for t, p, _ in pairs)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        results += [{"key": f"precision/{c}", "score": prec},
                    {"key": f"recall/{c}", "score": rec},
                    {"key": f"f1/{c}", "score": f1}]
    results.append({"key": "macro_f1", "score": sum(f1s) / max(len(f1s), 1)})

    ok = [conf for t, p, conf in pairs if t == p and conf is not None]
    bad = [conf for t, p, conf in pairs if t != p and conf is not None]
    if ok:
        results.append({"key": "mean_confidence_correct", "score": sum(ok) / len(ok)})
    if bad:
        results.append({"key": "mean_confidence_wrong", "score": sum(bad) / len(bad)})
    return {"results": results}


async def main_async(args):
    settings = get_settings()
    examples = load_examples(settings, args.sample)
    chain = build_classification_chain(_build_classifier_llm(settings))

    results = await aevaluate(
        make_target(chain),
        data=examples,
        evaluators=[supervisor_evaluator],
        summary_evaluators=[classification_summary],
        experiment_prefix=f"supervisor-{args.run_tag}",
        max_concurrency=args.concurrency,
        num_repetitions=args.repetitions,
        metadata={"component": "supervisor", "run_tag": args.run_tag, "sample": args.sample,
                  "git_commit": _git_commit(), "classifier_model": settings.AGENT_CLASSIFIER_MODEL_ID},
    )
    confusion: Counter = Counter()
    async for row in results:
        true = (row["example"].outputs or {}).get("question_type")
        pred = (row["run"].outputs or {}).get("question_type")
        if true != pred:
            confusion[f"{true}->{pred}"] += 1
    print(f"Experiment: {results.experiment_name}")
    print("Top confusions:", confusion.most_common(10) or "none")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(description="Evaluate the supervisor classifier in isolation.")
    p.add_argument("--run-tag", required=True)
    p.add_argument("--sample", default="full", help="'full' or an integer (stratified)")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--repetitions", type=int, default=1, help=">1 measures run-to-run consistency")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()