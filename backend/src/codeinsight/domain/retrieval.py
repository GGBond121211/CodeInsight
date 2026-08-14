"""Ranked retrieval value types."""

from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk

SEMANTIC_MATCH_REASON = "semantic_match"


@dataclass(frozen=True)
class RankedChunk:
    """A SourceChunk with an explainable retrieval score and 1-based rank."""

    chunk: SourceChunk
    score: float
    rank: int
    retrieval_reason: str = "direct_match"


@dataclass(frozen=True)
class SubQuestionEvidence:
    """Evidence kept independently for one planned subquestion."""

    question: str
    retrieval_mode: str
    results: tuple[RankedChunk, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("subquestion evidence question must not be blank")
        if len(self.results) != len(self.evidence_ids):
            raise ValueError("subquestion evidence IDs must match results")

    @property
    def covered(self) -> bool:
        """Whether this subquestion currently has at least one evidence block."""
        return bool(self.results)
