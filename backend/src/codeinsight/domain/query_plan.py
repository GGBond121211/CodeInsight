"""Public, validated plans produced before Smart Answer execution."""

from dataclasses import dataclass

SUPPORTED_RETRIEVAL_MODES = frozenset(
    {
        "lexical",
        "bm25",
        "hybrid",
    }
)
SUPPORTED_EXECUTION_ROUTES = frozenset({"linear", "agent", "insufficient"})
SUPPORTED_LANGUAGES = frozenset({"zh", "en", "mixed", "unknown"})


@dataclass(frozen=True)
class SubQuestion:
    """One independently retrievable public question plan item."""

    question: str
    intent: str
    retrieval_mode: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("subquestion must not be blank")
        if not self.intent.strip():
            raise ValueError("subquestion intent must not be blank")
        if self.retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(f"unsupported retrieval mode: {self.retrieval_mode}")


@dataclass(frozen=True)
class QueryPlan:
    """Validated route plan; it contains no source paths or hidden reasoning."""

    original_question: str
    language: str
    normalized_question: str
    subquestions: tuple[SubQuestion, ...]
    retrieval_modes: tuple[str, ...]
    execution_route: str
    confidence: float
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.original_question.strip():
            raise ValueError("original question must not be blank")
        if not self.normalized_question.strip():
            raise ValueError("normalized question must not be blank")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"unsupported language: {self.language}")
        if self.execution_route not in SUPPORTED_EXECUTION_ROUTES:
            raise ValueError(f"unsupported execution route: {self.execution_route}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("query plan confidence must be between 0 and 1")
        if len(self.subquestions) > 8:
            raise ValueError("query plan supports at most 8 subquestions")
        if len(set(self.retrieval_modes)) != len(self.retrieval_modes):
            raise ValueError("query plan retrieval modes must be unique")
        if any(mode not in SUPPORTED_RETRIEVAL_MODES for mode in self.retrieval_modes):
            raise ValueError("query plan contains an unsupported retrieval mode")
        subquestion_modes = tuple(item.retrieval_mode for item in self.subquestions)
        if set(subquestion_modes) != set(self.retrieval_modes):
            raise ValueError("retrieval_modes must match subquestion retrieval modes")
        if self.execution_route != "insufficient" and not self.subquestions:
            raise ValueError("non-insufficient query plans require a subquestion")

    def to_dict(self) -> dict:
        """Return only public plan data suitable for API display or events."""
        return {
            "original_question": self.original_question,
            "language": self.language,
            "normalized_question": self.normalized_question,
            "subquestions": [
                {
                    "question": item.question,
                    "intent": item.intent,
                    "retrieval_mode": item.retrieval_mode,
                }
                for item in self.subquestions
            ],
            "retrieval_modes": list(self.retrieval_modes),
            "execution_route": self.execution_route,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }
