"""
eval/component_evaluators.py

Evaluators for testing the retriever and the generator in isolation
(no LangGraph). Reuses the main judge for correctness/completeness.
"""

import math
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from helpers.config import get_settings
from stores.llm import LLMProviderFactory
from stores.vectordb import VectorDBProviderFactory
from stores.reranker import RerankerProviderFactory
from stores.llm.templates.template_parser import TemplateParser
from controllers import NLPController
from models import ProjectModel


async def build_eval_dependencies(settings=None) -> dict:
    settings = settings or get_settings()

    postgres_conn = (
        f"postgresql+asyncpg://{settings.POSTGRES_USERNAME}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_MAIN_DATABASE}"
    )
    engine = create_async_engine(postgres_conn)
    db_client = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    llm_factory = LLMProviderFactory(settings)
    vectordb_factory = VectorDBProviderFactory(settings, db_client=db_client)
    reranker_factory = RerankerProviderFactory(settings)

    generation_client = llm_factory.create(settings.GENERATION_BACKEND)
    generation_client.set_generation_model(settings.GENERATION_MODEL_ID)

    embedding_client = llm_factory.create(settings.EMBEDDING_BACKEND)
    embedding_client.set_embedding_model(settings.EMBEDDING_MODEL_ID, settings.EMBEDDING_MODEL_SIZE)

    vectordb_client = vectordb_factory.create(settings.VECTOR_DB_BACKEND)
    await vectordb_client.connect()

    reranker_client = reranker_factory.create(settings.RERANKER_BACKEND)
    reranker_client.set_reranker_model(settings.RERANKER_MODEL_ID)

    template_parser = TemplateParser(language=settings.PRIMARY_LANG, default_language=settings.DEFAULT_LANG)

    nlp_controller = NLPController(
        vectordb_client=vectordb_client,
        generation_client=generation_client,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        template_parser=template_parser,
    )

    project_model_holder: dict = {}

    async def get_project(project_id: int):
        if "model" not in project_model_holder:
            project_model_holder["model"] = await ProjectModel.create_instance(db_client=db_client)
        return await project_model_holder["model"].get_project_or_create_one(project_id=project_id)


    return {"settings": settings, "engine": engine, "db_client": db_client, "vectordb_client": vectordb_client,
        "nlp_controller": nlp_controller, "get_project": get_project,
        "reranker_client": reranker_client, "generation_client": generation_client,
        "template_parser": template_parser, "db_client": db_client}




STAGES = ("dense", "lexical", "fused", "reranked")


def _dedupe(ids: list[str]) -> list[str]:
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def rank_metrics(ranked: list[str], expected: list[str], ks=(5, 10)) -> dict[str, float]:
    """Document-level ranking metrics. `ranked` is a best-first list of doc_ids
    (chunk-level duplicates collapse to the doc's best rank)."""
    exp = set(expected)
    ranked = _dedupe(ranked)
    m: dict[str, float] = {}
    for k in ks:
        top = ranked[:k]
        hits = len(set(top) & exp)
        dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(top) if d in exp)
        idcg = sum(1 / math.log2(i + 2) for i in range(min(len(exp), k)))
        m[f"recall@{k}"] = hits / len(exp)
        m[f"hit@{k}"] = float(hits > 0)
        m[f"precision@{k}"] = hits / k
        m[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    m["mrr"] = next((1 / (i + 1) for i, d in enumerate(ranked) if d in exp), 0.0)
    return m


def retrieval_evaluator(outputs: dict, reference_outputs: dict):
    expected = reference_outputs.get("expected_doc_ids") or []
    if not outputs or not expected:  # info_not_found etc.: nothing to score
        return {"results": []}

    results = []
    for stage in STAGES:
        m = rank_metrics(outputs.get(stage, []), expected)
        for key in ("recall@5", "recall@10", "mrr"):
            results.append({"key": f"{stage}/{key}", "score": m[key]})

    # ceiling for the reranker: is the answer anywhere in the candidate set?
    fused = set(outputs.get("fused", []))
    results.append({"key": "candidate_recall", "score": len(fused & set(expected)) / len(set(expected))})

    final = rank_metrics(outputs.get("reranked", []), expected, ks=(5,))
    for key in ("hit@5", "ndcg@5", "precision@5"):
        results.append({"key": f"reranked/{key}", "score": final[key]})
    return {"results": results}


# ---------------------------------------------------------------------------
# Faithfulness (generation grounded in the context it was given)
# ---------------------------------------------------------------------------

class ClaimSupport(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the context directly states or entails the claim.")


class FaithfulnessVerdict(BaseModel):
    claims: list[ClaimSupport] = Field(default_factory=list)


FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You check whether an answer is grounded in the provided context passages. "
     "List each discrete factual claim in the answer and mark it supported only if the "
     "context directly states or clearly entails it. Ignore your own knowledge. "
     "If the answer declines to answer or states the information is unavailable, return no claims."),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])


def build_faithfulness_chain(judge_llm):
    return FAITHFULNESS_PROMPT | judge_llm.with_structured_output(FaithfulnessVerdict)


def build_faithfulness_evaluator(faith_chain):
    async def faithfulness_evaluator(outputs: dict, reference_outputs: dict):
        if not outputs or not outputs.get("answer") or not outputs.get("contexts"):
            return {"results": []}
        context = "\n\n".join(f"[{i + 1}] {c['text']}" for i, c in enumerate(outputs["contexts"]))
        verdict = await faith_chain.ainvoke({"context": context, "answer": outputs["answer"]})
        if not verdict.claims:
            return {"results": []}
        unsupported = [c.claim for c in verdict.claims if not c.supported]
        return {"results": [{
            "key": "faithfulness",
            "score": 1 - len(unsupported) / len(verdict.claims),
            "comment": ("Unsupported: " + "; ".join(unsupported))[:500] if unsupported else "all claims supported",
        }]}
    return faithfulness_evaluator

