"""排序检索值对象。"""

from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk

DENSE_MATCH_REASON = "dense_match"
# 2.0 response compatibility for callers that still construct semantic results.
SEMANTIC_MATCH_REASON = "semantic_match"


@dataclass(frozen=True)
class RankedChunk:
    """带有可解释检索分数和从 1 开始排名的 SourceChunk。"""

    chunk: SourceChunk
    score: float
    rank: int
    retrieval_reason: str = "direct_match"


@dataclass(frozen=True)
class SubQuestionEvidence:
    """为一个计划子问题独立保存的证据。"""

    question: str
    retrieval_mode: str
    results: tuple[RankedChunk, ...]
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("子问题证据的 question 不能为空")
        if len(self.results) != len(self.evidence_ids):
            raise ValueError("子问题 evidence ID 必须与结果匹配")

    @property
    def covered(self) -> bool:
        """判断这个子问题当前是否至少有一个证据块。"""
        return bool(self.results)
