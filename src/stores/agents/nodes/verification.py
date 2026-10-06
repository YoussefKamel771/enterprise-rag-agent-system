"""
stores/agents/nodes/verification.py

Phase-3 verifier gate + reformulation loop, and the two terminal nodes
(`finalize`, `info_not_found`).

The verifier now JUDGES grounding instead of trusting inline citations. The old
version only checked that doc_ids cited by the generator were among the
retrieved chunks, but the generator prompt never asks for citations, so every
claim came out "unsupported" and phase 3 always ended in info_not_found.

Now: one structured LLM call sees the same context the generator saw (via
`select_context_chunks`) plus the draft, lists each claim, and says which
passages entail it. We then validate the returned doc_ids against the context
so a hallucinated id can never count as support.

Strictness: a claim only counts as supported if the judge marks it supported
AND names at least one valid doc_id. A draft with no claims (a refusal such as
"the documents don't say") passes: there is nothing unsupported to retry for.
If the judge call itself fails we fail OPEN (accept the draft, keep the
generator's citations) so infra errors aren't scored as "info not found".
"""

import logging
from typing import Dict, List

from ..config import AgentConfig
from ..prompts import GROUNDING_PROMPT, REFORMULATION_PROMPT
from ..schemas import GroundingVerdict
from ..state import RAGState
from .generation import select_context_chunks

logger = logging.getLogger("uvicorn")


def _format_context(chunks: List) -> str:
    """Label each passage with its doc_id so the judge can cite it back."""
    return "\n\n".join(f"[doc_id={d.doc_id}]\n{d.text}" for d in chunks)


def build_verifier_node(verifier_llm, config: AgentConfig):
    chain = GROUNDING_PROMPT | verifier_llm.with_structured_output(
        GroundingVerdict, method="function_calling")

    async def verifier_node(state: RAGState) -> dict:
        draft = state.get("draft_answer")
        if not draft:
            return {"grounding_passed": False, "unsupported_claims": ["no_draft_answer"]}

        # Judge against exactly what the generator was shown.
        context = select_context_chunks(
            state.get("retrieved_chunks", []),
            min_score=config.min_generation_score,
            max_chunks=config.max_generation_chunks,
        )
        chunks_by_doc: Dict[str, List] = {}
        for d in context:
            chunks_by_doc.setdefault(d.doc_id, []).append(d)

        try:
            verdict: GroundingVerdict = await chain.ainvoke(
                {"context": _format_context(context), "draft_answer": draft})
        except Exception:
            logger.exception("[agent] verifier judge call failed; failing open")
            return {"verified_claims": [], "unsupported_claims": [], "grounding_passed": True}

        verified, unsupported = [], []
        for c in verdict.claims:
            # keep only ids that really are in the context, de-duplicated in order
            doc_ids = [d for d in dict.fromkeys(c.supporting_doc_ids) if d in chunks_by_doc]
            supported = c.supported and bool(doc_ids)
            verified.append({
                "claim": c.claim,
                "supported": supported,
                "supporting_doc_ids": doc_ids if supported else [],
                "chunk_ids": [ch.chunk_id for d in doc_ids for ch in chunks_by_doc[d]] if supported else [],
            })
            if not supported:
                unsupported.append(c.claim)

        logger.info("[agent] verifier: %d claims judged, %d unsupported", len(verified), len(unsupported))
        return {
            "verified_claims": verified,
            "unsupported_claims": unsupported,
            "grounding_passed": not unsupported,
        }
    return verifier_node


def build_reformulator_node(reform_llm):
    chain = REFORMULATION_PROMPT | reform_llm

    async def reformulator_node(state: RAGState) -> dict:
        new_question = await chain.ainvoke({
            "question": state["question"],
            "unsupported": "; ".join(state.get("unsupported_claims", [])),
        })
        return {
            "question": new_question.content if hasattr(new_question, "content") else str(new_question),
            "iteration_count": state.get("iteration_count", 0) + 1,
        }
    return reformulator_node


def build_info_not_found_node():
    async def info_not_found_node(state: RAGState) -> dict:
        return {
            "final_answer": (
                "I couldn't find information supporting an answer to this question "
                "in the available documents after an exhaustive search."
            ),
            "citations": [],
        }
    return info_not_found_node


def build_finalize_node():
    async def finalize_node(state: RAGState) -> dict:
        return {
            "final_answer": state.get("draft_answer"),
            "citations": sorted({
                doc_id for claim in state.get("verified_claims", [])
                for doc_id in claim["supporting_doc_ids"]
            }) or state.get("citations", []),
        }
    return finalize_node