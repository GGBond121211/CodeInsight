"""不可变的回答和引用值对象。"""

from dataclasses import dataclass

from codeinsight.domain.query_plan import QueryPlan

ANSWERED = "answered"
INSUFFICIENT_EVIDENCE = "insufficient_evidence"
PARTIALLY_ANSWERED = "partially_answered"
SUPPORTED_OUTCOMES = frozenset({ANSWERED, INSUFFICIENT_EVIDENCE})


@dataclass(frozen=True)
class AnswerCitation:
    evidence_id: str
    relative_path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class ModelCompletion:
    content: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    estimated_input_tokens: int | None = None


@dataclass(frozen=True)
class ModelAnswer:
    outcome: str
    answer: str
    evidence_ids: tuple[str, ...]
    model: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class RepositoryAnswer:
    outcome: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    retrieval_mode: str
    model: str | None
    prompt_version: str
    input_tokens: int | None
    output_tokens: int | None


@dataclass(frozen=True)
class SubQuestionAnswer:
    """一个 Smart Answer 子问题的回答和全局引用。"""

    question: str
    intent: str
    retrieval_mode: str
    outcome: str
    answer: str
    citations: tuple[AnswerCitation, ...]


@dataclass(frozen=True)
class AutoAnswerEvent:
    """不包含模型隐藏推理的公开 Smart Answer 事件摘要。"""

    sequence: int
    step: str
    summary: str


@dataclass(frozen=True)
class AutoAnswer:
    """带有路线和成本元数据的公开多问题回答。"""

    outcome: str
    answer: str
    citations: tuple[AnswerCitation, ...]
    retrieval_mode: str
    model: str | None
    prompt_version: str
    input_tokens: int
    output_tokens: int
    embedding_input_tokens: int
    plan: QueryPlan
    subquestions: tuple[SubQuestionAnswer, ...]
    router_model: str | None
    router_input_tokens: int
    router_output_tokens: int
    router_elapsed_milliseconds: float
    fallback_reason: str | None
    events: tuple[AutoAnswerEvent, ...]
