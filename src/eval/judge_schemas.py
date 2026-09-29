from typing import Literal, Optional
from pydantic import BaseModel, Field


class FactCoverage(BaseModel):
    fact: str
    covered: bool
    evidence_quote: Optional[str] = Field(
        default=None,
        description="Short quote from the generated answer supporting `covered`, or null if not covered.",
    )


class JudgeVerdict(BaseModel):
    correctness: Literal[0, 1] = Field(
        description="1 only if the answer is factually correct AND contains no hallucinated claims "
                    "not supported by the gold answer. Any confident-but-wrong statement -> 0, "
                    "even if other parts of the answer are right."
    )
    correctness_reasoning: str
    fact_coverage: list[FactCoverage] = Field(default_factory=list)
    acknowledges_contradiction: Optional[bool] = Field(
        default=None,
        description="Only meaningful for conflicting_info questions: does the answer surface "
                    "the conflict with source attribution, rather than silently picking one side?",
    )

    @property
    def completeness_pct(self) -> float:
        if not self.fact_coverage:
            return 0.0
        return 100.0 * sum(f.covered for f in self.fact_coverage) / len(self.fact_coverage)

    def to_record(self) -> dict:
        """JSON-safe dict for storage.py -- includes the derived completeness_pct
        since that's a @property and wouldn't survive model_dump() otherwise."""
        return {**self.model_dump(), "completeness_pct": self.completeness_pct}