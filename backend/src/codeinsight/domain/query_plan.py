"""Smart Answer 执行前生成的公开、已校验查询计划。"""

from dataclasses import dataclass

SUPPORTED_RETRIEVAL_MODES = frozenset(
    {
        "hybrid",
        "dense",
        "sparse",
    }
)
SUPPORTED_EXECUTION_ROUTES = frozenset({"linear", "agent", "insufficient"})
SUPPORTED_LANGUAGES = frozenset({"zh", "en", "mixed", "unknown"})


@dataclass(frozen=True)
class SubQuestion:
    """一个可以独立检索的公开问题计划项。"""

    question: str
    intent: str
    retrieval_mode: str

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("subquestion 不能为空")
        if not self.intent.strip():
            raise ValueError("subquestion 的 intent 不能为空")
        if self.retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
            raise ValueError(f"retrieval_mode 不受支持：{self.retrieval_mode}")


@dataclass(frozen=True)
class QueryPlan:
    """已校验的路线计划；其中不包含源码路径或隐藏推理。"""

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
            raise ValueError("original_question 不能为空")
        if not self.normalized_question.strip():
            raise ValueError("normalized_question 不能为空")
        if self.language not in SUPPORTED_LANGUAGES:
            raise ValueError(f"language 不受支持：{self.language}")
        if self.execution_route not in SUPPORTED_EXECUTION_ROUTES:
            raise ValueError(f"execution_route 不受支持：{self.execution_route}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("QueryPlan 的 confidence 必须在 0 和 1 之间")
        if len(self.subquestions) > 8:
            raise ValueError("QueryPlan 最多支持 8 个 subquestion")
        if len(set(self.retrieval_modes)) != len(self.retrieval_modes):
            raise ValueError("QueryPlan 的 retrieval_mode 不能重复")
        for mode in self.retrieval_modes:
            if mode not in SUPPORTED_RETRIEVAL_MODES:
                raise ValueError("QueryPlan 包含不受支持的 retrieval_mode")
        subquestion_mode_list: list[str] = []
        for item in self.subquestions:
            subquestion_mode_list.append(item.retrieval_mode)
        subquestion_modes = tuple(subquestion_mode_list)
        if set(subquestion_modes) != set(self.retrieval_modes):
            raise ValueError("retrieval_modes 必须与 subquestion 的 retrieval_mode 匹配")
        if self.execution_route != "insufficient" and not self.subquestions:
            raise ValueError("非 insufficient 的 QueryPlan 必须包含至少一个 subquestion")

    def to_dict(self) -> dict:
        """只返回适合 API 展示或事件记录的公开计划数据。"""
        return {
            "original_question": self.original_question,
            "language": self.language,
            "normalized_question": self.normalized_question,
            "subquestions": self._subquestion_dicts(),
            "retrieval_modes": list(self.retrieval_modes),
            "execution_route": self.execution_route,
            "confidence": self.confidence,
            "fallback_reason": self.fallback_reason,
        }

    def _subquestion_dicts(self) -> list[dict[str, str]]:
        subquestions: list[dict[str, str]] = []
        for item in self.subquestions:
            subquestions.append(
                {
                    "question": item.question,
                    "intent": item.intent,
                    "retrieval_mode": item.retrieval_mode,
                }
            )
        return subquestions
