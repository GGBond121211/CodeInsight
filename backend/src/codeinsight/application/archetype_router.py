"""问题原型路由（Q-012 U2）：用示例问题做粗筛，命中就走固定配方。

它和 RAG 的区别值得写清楚：检索是「找出与问题相关的片段」，这里是「判断这个
问题属于哪一类已知问法」，命中的产物是一条取证路线，不是答案。所以阈值要
卡在「明显属于某一类」，卡不到就回退——错命中比未命中更危险：未命中只是慢，
错命中会给出一个看起来正确但答非所问的答案。

为什么用 Embedding 而不是让模型分类：每一次路由都调模型，就又把快路径的成本
拉回原来的量级。示例问题向量按原型库版本缓存在进程内，稳定状态下每个问题只
多一次 Embedding 调用。

2026-09-13：每个原型除示例问题外再锚一条 **summary**。原因是实测发现只拿示例
问题当锚点时，改写过的问句（例如「哪里实现了……」）很难过阈值；补通用例句又会
把「这个功能应该怎么设计比较好？」这类开放问题误判进来（负例错命中 0.083→0.167）。
把原型自己的摘要（作者为这个原型写的定义句）也当锚点，在 60 条 holdout + 12 条
负例上同时改善：命中 0.433→0.517、判对 0.383→0.467，判错原型 0.050 与负例错命中
0.083 都不变。摘要不是新造的例句，它本来就是这个原型的定义，所以不算往评测集上贴答案。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from codeinsight.application.archetype_library import ARCHETYPES_V1, LIBRARY_VERSION
from codeinsight.domain.archetype import QuestionArchetype

Embedder = Callable[[Sequence[str]], "object"]


@dataclass(frozen=True)
class ArchetypeRouterConfig:
    """阈值来自 C2.2 的阈值扫描（`experiments/q12_archetype_eval.py`，60 条 holdout +
    12 条负例）：0.50→0.62 区间命中率高但判错原型也多（0.117→0.100），0.65 起
    判错原型降到 0.050、命中里判对升到 0.885，代价是命中率从 0.550 降到 0.433。
    设计原则是「错命中比未命中更危险」，所以取 0.65。样本只有 72 条，这个取值是
    方向性证据，不是 A/B 结论——它还被后置条件挡着：快路径默认关闭。
    """

    threshold: float = 0.65
    margin: float = 0.02

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold 必须落在 (0, 1] 之间")
        if self.margin < 0.0:
            raise ValueError("margin 不能为负")


@dataclass(frozen=True)
class ArchetypeMatch:
    archetype: QuestionArchetype
    score: float
    runner_up: float

    @property
    def margin(self) -> float:
        return self.score - self.runner_up


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("向量维度不一致")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for a, b in zip(left, right):
        dot += a * b
        left_norm += a * a
        right_norm += b * b
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


class ArchetypeRouter:
    """把问题映射到原型；不确定时返回 None，由调用方回退到 Tool Loop。"""

    def __init__(
        self,
        *,
        embed: Embedder,
        library: tuple[QuestionArchetype, ...] = ARCHETYPES_V1,
        config: ArchetypeRouterConfig | None = None,
        library_version: str = LIBRARY_VERSION,
    ) -> None:
        if not library:
            raise ValueError("原型库不能为空")
        self._embed = embed
        self._library = library
        self._config = config or ArchetypeRouterConfig()
        self._library_version = library_version
        self._example_vectors: tuple[tuple[float, ...], ...] | None = None
        # 每个原型两条锚：示例问题（逐条）+ 摘要（一条）。
        self._anchor_owners: tuple[int, ...] = tuple(
            index
            for index, archetype in enumerate(library)
            for _ in (*archetype.example_questions, "")
        )

    @property
    def config(self) -> ArchetypeRouterConfig:
        return self._config

    @property
    def library_version(self) -> str:
        return self._library_version

    def _anchor_texts(self) -> tuple[str, ...]:
        return tuple(
            question
            for archetype in self._library
            for question in (*archetype.example_questions, archetype.summary)
        )

    def _vectors(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        batch = self._embed(texts)
        vectors = tuple(tuple(float(value) for value in item) for item in batch.vectors)
        if len(vectors) != len(texts):
            raise ValueError("Embedding 返回数量与输入数量不一致")
        return vectors

    def _ensure_anchor_vectors(self) -> tuple[tuple[float, ...], ...]:
        if self._example_vectors is None:
            self._example_vectors = self._vectors(self._anchor_texts())
        return self._example_vectors

    def route(self, question: str) -> ArchetypeMatch | None:
        """返回命中原型；低于阈值或与第二名太近都返回 None。"""

        if not question.strip():
            raise ValueError("question 不能为空")
        anchor_vectors = self._ensure_anchor_vectors()
        question_vector = self._vectors([question])[0]
        best_per_owner: dict[int, float] = {}
        for owner, vector in zip(self._anchor_owners, anchor_vectors):
            score = _cosine(question_vector, vector)
            if owner not in best_per_owner or score > best_per_owner[owner]:
                best_per_owner[owner] = score
        if not best_per_owner:
            return None
        ranked = sorted(best_per_owner.items(), key=lambda item: item[1], reverse=True)
        best_owner, best_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        if best_score < self._config.threshold:
            return None
        if best_score - runner_up < self._config.margin:
            return None
        return ArchetypeMatch(
            archetype=self._library[best_owner],
            score=best_score,
            runner_up=runner_up,
        )
