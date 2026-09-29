"""
eval/judge.py -- Section 4.2

`build_judge_llm(settings)` mirrors stores/agents/graph.py::_build_classifier_llm
-- same OPENAI-backed pattern, but reads AGENT_JUDGE_MODEL_ID, kept as a
SEPARATE setting on purpose: the judge must be your strongest model and must
stay fixed across phase comparisons, independent of whatever routing model
you're iterating on.
"""

import asyncio
import logging

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .judge_schemas import JudgeVerdict

logger = logging.getLogger("uvicorn.eval")

JUDGE_SYSTEM_PROMPT = """\
You are a strict evaluator for an enterprise RAG system. You will be given a
question, a gold (reference) answer, a list of facts the answer is expected
to contain, and the system's generated answer. Score the generated answer
against the gold answer and fact list -- NOT against your own knowledge of
the topic, since you cannot see the source documents.

Correctness rules:
- correctness = 1 only if the generated answer's core claims match the gold
  answer and contain no material hallucination.
- A generated answer that is correct but incomplete is still correctness = 1
  (completeness is scored separately via fact coverage).
- A generated answer that states a plausible-sounding but unsupported claim
  is correctness = 0, even if the rest of the answer is accurate.
- If the gold answer indicates the information is not available and the
  generated answer also declines to answer, correctness = 1.
- If the gold answer indicates the information is not available but the
  generated answer confidently answers anyway, correctness = 0.

For each item in `answer_facts`, mark it covered=true only if the generated
answer states that fact (paraphrase acceptable; omission or contradiction is
not "covered").
"""

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", JUDGE_SYSTEM_PROMPT),
    ("human",
     "Question: {question}\n\n"
     "Gold answer: {gold_answer}\n\n"
     "Expected facts:\n{answer_facts}\n\n"
     "Generated answer: {generated_answer}\n\n"
     "{conflict_instruction}"),
])

CONFLICT_INSTRUCTION = (
    "This is a Conflicting Info question. Additionally set "
    "`acknowledges_contradiction` based on whether the generated answer "
    "surfaces the disagreement between sources, rather than silently "
    "picking one."
)


def build_judge_llm(settings):
    if settings.AGENT_JUDGE_BACKEND == "OPENAI":
        return ChatOpenAI(
            model=settings.AGENT_JUDGE_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL or None,
            temperature=0,
        )
    raise ValueError(f"Unsupported AGENT_JUDGE_BACKEND: {settings.AGENT_JUDGE_BACKEND}")


def build_judge_chain(judge_llm):
    return JUDGE_PROMPT | judge_llm.with_structured_output(JudgeVerdict)


async def judge_one(judge_chain, question: str, gold_answer: str, answer_facts: list[str],
                     generated_answer: str, question_type: str) -> JudgeVerdict:
    return await judge_chain.ainvoke({
        "question": question,
        "gold_answer": gold_answer,
        "answer_facts": "\n".join(f"- {f}" for f in answer_facts) or "(none listed)",
        "generated_answer": generated_answer or "[NO ANSWER GENERATED]",
        "conflict_instruction": CONFLICT_INSTRUCTION if question_type == "conflicting_info" else "",
    })


async def judge_batch(judge_chain, items: list[dict], concurrency: int = 5) -> list[JudgeVerdict | None]:
    """items: [{question, gold_answer, answer_facts, generated_answer, question_type, question_id}, ...]
    Returns verdicts in the same order as `items`; a failed judge call yields
    None at that position rather than aborting the whole batch."""
    sem = asyncio.Semaphore(concurrency)

    async def _run(item):
        async with sem:
            try:
                return await judge_one(
                    judge_chain, item["question"], item["gold_answer"], item["answer_facts"],
                    item["generated_answer"], item["question_type"],
                )
            except Exception:
                logger.exception("Judge call failed for question_id=%s", item.get("question_id"))
                return None

    return await asyncio.gather(*[_run(item) for item in items])


def overall_score(judgments: list[JudgeVerdict]) -> float:
    """Overall Score = (1/N) * sum_i [ Correctness_i * Completeness_i ] -- ignores
    None entries (failed judge calls) rather than counting them as 0, since a
    judge-infra failure is a different thing from a wrong answer."""
    scored = [j for j in judgments if j is not None]
    if not scored:
        return 0.0
    return sum(j.correctness * (j.completeness_pct / 100.0) for j in scored) / len(scored)