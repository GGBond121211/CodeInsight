"""仓库回答 Agent 生成的不可变值对象。"""

from dataclasses import dataclass

from codeinsight.domain.answer import RepositoryAnswer, SubQuestionAnswer

REVIEW_PASS = "pass"
REVIEW_REVISE = "revise"
SUPPORTED_REVIEW_VERDICTS = frozenset({REVIEW_PASS, REVIEW_REVISE})


@dataclass(frozen=True)
class CitationReview:
    verdict: str
    supported_evidence_ids: tuple[str, ...]
    feedback: str


@dataclass(frozen=True)
class AgentEvent:
    sequence: int
    step: str
    summary: str


@dataclass(frozen=True)
class AgentRepositoryAnswer:
    result: RepositoryAnswer
    revisions: int
    events: tuple[AgentEvent, ...]
    input_tokens: int
    output_tokens: int
    embedding_input_tokens: int = 0
    subquestions: tuple[SubQuestionAnswer, ...] = ()
