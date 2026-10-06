"""
stores/agents/llm.py

Construction of the LangChain chat models the graph nodes use for structured
output. Kept separate from `generation_client` (your LLMInterface provider),
which stays dedicated to answer generation.

`AgentLLMs` names every role explicitly. Today all roles share one model, but
a node asking for `llms.verifier` instead of "the classifier" means you can
later give the verifier a stronger model without touching any node code.
"""

from dataclasses import dataclass

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


@dataclass(frozen=True)
class AgentLLMs:
    """One field per role in the graph."""

    classifier: object
    decomposer: object
    constraint: object
    expansion: object
    verifier: object
    reformulator: object

    @classmethod
    def shared(cls, llm) -> "AgentLLMs":
        """Every role uses the same model instance (current behaviour)."""
        return cls(llm, llm, llm, llm, llm, llm)


def build_agent_llms(settings) -> AgentLLMs:
    return AgentLLMs.shared(build_classifier_llm(settings))
