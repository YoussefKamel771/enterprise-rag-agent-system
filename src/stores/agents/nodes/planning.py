"""
stores/agents/nodes/planning.py

Nodes that break a question into sub-queries and fan out with LangGraph's
`Send` primitive to the shared `retrieve_subquery` worker.

`decomposer` and `broad_synthesis` were byte-for-byte duplicates apart from
whether `facets` is recorded; they now share `_build_fanout_node`.
"""

from typing import Literal

from langgraph.types import Command, Send

from ..prompts import DECOMPOSE_PROMPT
from ..routing import RETRIEVE_SUBQUERY
from ..schemas import SubQuestions
from ..state import RAGState


def _build_fanout_node(llm, *, record_facets: bool):
    chain = DECOMPOSE_PROMPT | llm.with_structured_output(SubQuestions, method="function_calling")

    async def fanout_node(state: RAGState) -> Command[Literal["retrieve_subquery"]]:
        result: SubQuestions = await chain.ainvoke({"question": state["question"]})
        # An empty list would produce Command(goto=[]) and silently dead-end the
        # graph with no draft answer, so fall back to the original question.
        sub_queries = result.sub_queries or [state["question"]]

        update: dict = {"sub_queries": sub_queries}
        if record_facets:
            update["facets"] = result.facets

        sends = [
            Send(RETRIEVE_SUBQUERY, {**state, "question": q, "sub_queries": [q]})
            for q in sub_queries
        ]
        return Command(update=update, goto=sends)

    return fanout_node


def build_decomposer_node(decomposer_llm):
    return _build_fanout_node(decomposer_llm, record_facets=True)


def build_broad_synthesis_node(expansion_llm):
    return _build_fanout_node(expansion_llm, record_facets=False)
