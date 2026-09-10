"""Qdrant 运行时向量后端的最小契约。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from codeinsight.domain.semantic import SparseEmbedding


@dataclass(frozen=True)
class VectorPoint:
    """一条 Dense/Sparse 向量及其可回到源码 Evidence 的 payload。"""

    point_id: str
    vector: tuple[float, ...]
    payload: Mapping[str, object]
    sparse_vector: SparseEmbedding | None = None

    @property
    def dense_vector(self) -> tuple[float, ...]:
        """Dense 命名向量。"""
        return self.vector


@dataclass(frozen=True)
class VectorSearchHit:
    point_id: str
    score: float
    payload: Mapping[str, object]


class VectorStore(Protocol):
    """2.1 正式运行时的 Dense/Sparse 向量后端契约。"""

    def upsert(self, points: Iterable[VectorPoint]) -> None: ...

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]: ...

    def search_sparse(
        self,
        vector: SparseEmbedding,
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]: ...

    def delete(self, point_ids: Iterable[str]) -> None: ...

    def health(self) -> bool: ...
