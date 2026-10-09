"""
stores/agents/llm.py

Construction of the LangChain chat models the graph nodes use for structured
output. Kept separate from `generation_client` (your LLMInterface provider),
which stays dedicated to answer generation.
"""

from dataclasses import dataclass, fields

from langchain_openai import ChatOpenAI


def build_classifier_llm(settings) -> ChatOpenAI:
    """Cheap/fast model used for supervisor + all structured-output calls."""
    if settings.AGENT_CLASSIFIER_BACKEND == "OPENAI":
        return ChatOpenAI(
            model=settings.AGENT_CLASSIFIER_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL or None,
            temperature=0,
            max_retries=1,
            extra_body={"max_tokens": 512, "reasoning_effort": "none"},
        )
    raise ValueError(f"Unsupported AGENT_CLASSIFIER_BACKEND: {settings.AGENT_CLASSIFIER_BACKEND}")


def build_memory_llm(settings) -> ChatOpenAI:
    """Model for summaries / question rewrites / memory extraction. Same wiring
    as the classifier but a larger output cap (512 tokens is too tight for a
    rolling summary). AGENT_MEMORY_MODEL_ID falls back to the classifier model."""
    if settings.AGENT_CLASSIFIER_BACKEND == "OPENAI":
        return ChatOpenAI(
            model=settings.AGENT_MEMORY_MODEL_ID or settings.AGENT_CLASSIFIER_MODEL_ID,
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_API_URL or None,   
            temperature=0,
            max_retries=1,
            extra_body={"max_tokens": 1024, "reasoning_effort": "none"},
        )
    raise ValueError(f"Unsupported AGENT_CLASSIFIER_BACKEND: {settings.AGENT_CLASSIFIER_BACKEND}")


@dataclass(frozen=True)
class AgentLLMs:
    """One field per role in the graph."""

    classifier: object
    decomposer: object
    constraint: object
    expansion: object
    verifier: object
    reformulator: object
    contextualizer: object
    summarizer: object
    memory_extractor: object

    @classmethod
    def shared(cls, llm) -> "AgentLLMs":
        """Every role uses the same model instance. Independent of field count."""
        return cls(**{f.name: llm for f in fields(cls)})


def build_agent_llms(settings) -> AgentLLMs:
    base = build_classifier_llm(settings)
    memory = build_memory_llm(settings)
    return AgentLLMs(
        classifier=base, decomposer=base, constraint=base, expansion=base,
        verifier=base, reformulator=base,
        contextualizer=memory, summarizer=memory, memory_extractor=memory,
    )