"""
eval/run_components.py

Evaluate the retriever and the generator on their own, outside LangGraph.
Each run is a LangSmith experiment on the same dataset as the agent runs, so
everything can be compared side by side.

    python -m eval.run_components retrieval  --run-tag r1 --sample 100
    python -m eval.run_components generation --run-tag g-oracle --context oracle --sample 50
    python -m eval.run_components generation --run-tag g-naive  --context retrieved --sample 50

generation --context oracle    : gold-doc chunks as context -> isolates generator/prompt/model
generation --context retrieved : real hybrid+rerank context -> naive-RAG baseline (no agent)
"""

import argparse
import asyncio
import logging

from langsmith import Client, aevaluate

from helpers.config import get_settings
from models import ChunkModel, RetrievalResult, RetrievedDocument
from ..load_questions import stratified_sample
from .components_eval.component_evaluators import build_eval_dependencies
from ..run_benchmark import _git_commit
from ..judge import build_judge_llm, build_judge_chain
from ..evaluators import build_evaluators
from .components_eval.component_evaluators import (
    retrieval_evaluator, build_faithfulness_chain, build_faithfulness_evaluator, _dedupe,
)

logger = logging.getLogger("uvicorn.eval")


def _load_examples(settings, sample: str, oracle_only: bool = False):
    client = Client()
    examples = list(client.list_examples(dataset_name=settings.EVAL_LANGSMITH_DATASET))
    if not examples:
        raise SystemExit("Dataset is empty -- run `python -m eval.upload_dataset` first.")
    if oracle_only:  # oracle context needs gold docs (drops info_not_found)
        examples = [e for e in examples if e.outputs.get("expected_doc_ids")]
    if sample != "full":
        examples = stratified_sample(examples, n=int(sample), key=lambda e: e.metadata["question_type"])
    return examples


# ---------------------------------------------------------------------------
# Retrieval target
# ---------------------------------------------------------------------------

def make_retrieval_target(deps, project_id: int, candidate_k: int, top_k: int):
    nlp, reranker = deps["nlp_controller"], deps["reranker_client"]

    async def target(inputs: dict) -> dict:
        project = await deps["get_project"](project_id)
        q = inputs["question"]

        # one retrieval (embedding + dense + lexical + RRF), then rerank once
        fused = await nlp.search_vector_db_collection(
            project=project, text=q, candidate_k=candidate_k, top_k=None, debug=True)
        if not fused or not getattr(fused, "ok", False) or not fused.documents:
            return {"dense": [], "lexical": [], "fused": [], "reranked": []}

        reranked = await reranker.rerank(query=q, results=fused, top_n=top_k)
        return {
            "dense": _dedupe([r["doc_id"] for r in (fused.dense_debug or [])]),
            "lexical": _dedupe([r["doc_id"] for r in (fused.lexical_debug or [])]),
            "fused": _dedupe([d.doc_id for d in fused.documents]),
            "reranked": _dedupe([d.doc_id for d in reranked.documents]),
        }
    return target


# ---------------------------------------------------------------------------
# Generation target
# ---------------------------------------------------------------------------

async def _oracle_contexts(deps, chunk_model, project_id, question, expected, max_chunks):
    chunks = await chunk_model.get_project_chunks(
        project_id=project_id, page_no=1, page_size=5000, asset_ids=list(expected))
    if not chunks:
        raise RuntimeError(f"No chunks found for gold docs {expected[:3]}... (not chunked/indexed?)")
    docs = [
        RetrievedDocument(chunk_id=c.chunk_id, doc_id=c.chunk_asset_id, text=c.chunk_text,
                          score=0.0, rank=i + 1)
        for i, c in enumerate(chunks)
    ]
    ranked = (await deps["reranker_client"].rerank(
        query=question, results=RetrievalResult(documents=docs), top_n=len(docs))).documents

    picked, seen = [], set()
    for d in ranked:                       # best chunk of every gold doc first
        if d.doc_id not in seen:
            picked.append(d)
            seen.add(d.doc_id)
    for d in ranked:                       # then fill with the next-best chunks
        if len(picked) >= max_chunks:
            break
        if d not in picked:
            picked.append(d)
    return picked[:max_chunks]


def make_generation_target(deps, examples, chunk_model, project_id, context_mode,
                           candidate_k, top_k, max_chunks):
    gen, tp, nlp = deps["generation_client"], deps["template_parser"], deps["nlp_controller"]
    expected_by_qid = {e.inputs["question_id"]: e.outputs["expected_doc_ids"] for e in examples}

    async def target(inputs: dict) -> dict:
        q = inputs["question"]
        if context_mode == "oracle":
            docs = await _oracle_contexts(deps, chunk_model, project_id, q,
                                          expected_by_qid[inputs["question_id"]], max_chunks)
        else:
            project = await deps["get_project"](project_id)
            res = await nlp.search_vector_db_collection(
                project=project, text=q, candidate_k=candidate_k, top_k=top_k)
            docs = res.documents if res and getattr(res, "ok", False) else []

        # same prompt construction as the agent's generator node
        system_prompt = tp.get("rag", "system_prompt")
        docs_prompt = "\n".join(
            tp.get("rag", "document_prompt", {"doc_num": i + 1, "chunk_text": d.text})
            for i, d in enumerate(docs))
        footer = tp.get("rag", "footer_prompt", {"query": q})
        answer = await gen.generate_text(
            prompt="\n\n".join([docs_prompt, footer]),
            chat_history=[gen.construct_prompt(system_prompt, role="system")])
        if not answer:
            raise RuntimeError("generation_failed: empty completion")

        return {
            "answer": answer,
            "contexts": [{"doc_id": d.doc_id, "text": d.text} for d in docs],
            "context_mode": context_mode,
        }
    return target


# ---------------------------------------------------------------------------

async def main_async(args):
    settings = get_settings()
    component = args.component
    examples = _load_examples(settings, args.sample,
                              oracle_only=(component == "generation" and args.context == "oracle"))
    deps = await build_eval_dependencies(settings)
    project_id = args.project_id or settings.EVAL_PROJECT_ID

    metadata = {
        "component": component,
        "run_tag": args.run_tag,

        # Dataset
        "dataset": settings.EVAL_LANGSMITH_DATASET,
        "dataset_version": "v1-gold-docs",  # fixed for now, but could be a setting
        "n_examples": len(examples),
        "sample": args.sample,

        # Indexing
        "chunker": "recursive-splitter",  # fixed for now, but could be a setting

        # Retrieval
        "candidate_k": args.candidate_k,
        "top_k": args.top_k,

        # Models
        "embedding_model": settings.EMBEDDING_MODEL_ID,
        "embedding_model_size": settings.EMBEDDING_MODEL_SIZE,
        "reranker_model": settings.RERANKER_MODEL_ID,

        # Reproducibility
        "git_commit": _git_commit(),
    }

    try:
        if component == "retrieval":
            target = make_retrieval_target(deps, project_id, args.candidate_k, args.top_k)
            evaluators = [retrieval_evaluator]
            prefix = f"retrieval-{args.run_tag}"
        else:
            judge_llm = build_judge_llm(settings)
            judge_evaluator = build_evaluators(build_judge_chain(judge_llm))[0]  # correctness/completeness/overall
            faith_evaluator = build_faithfulness_evaluator(build_faithfulness_chain(judge_llm))
            chunk_model = await ChunkModel.create_instance(db_client=deps["db_client"])
            target = make_generation_target(deps, examples, chunk_model, project_id, args.context,
                                            args.candidate_k, args.top_k, args.max_chunks)
            evaluators = [judge_evaluator, faith_evaluator]
            prefix = f"generation-{args.context}-{args.run_tag}"
            metadata.update({"context_mode": args.context, "max_chunks": args.max_chunks,
                             "generation_model": settings.GENERATION_MODEL_ID,
                             "judge_model": settings.AGENT_JUDGE_MODEL_ID})

        results = await aevaluate(target, data=examples, evaluators=evaluators,
                                  experiment_prefix=prefix, max_concurrency=args.concurrency,
                                  metadata=metadata)
        n = sum([1 async for _ in results])
        print(f"Done: {n} runs -> experiment '{results.experiment_name}' (open it in LangSmith)")
    finally:
        await deps["engine"].dispose()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    p = argparse.ArgumentParser(description="Evaluate retriever / generator in isolation.")
    p.add_argument("component", choices=["retrieval", "generation"])
    p.add_argument("--run-tag", required=True)
    p.add_argument("--sample", default="100", help="'full' or an integer (stratified)")
    p.add_argument("--context", choices=["oracle", "retrieved"], default="oracle",
                   help="generation only: where the context comes from")
    p.add_argument("--candidate-k", type=int, default=20)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-chunks", type=int, default=5, help="oracle context size")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--project-id", type=int, default=1)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()