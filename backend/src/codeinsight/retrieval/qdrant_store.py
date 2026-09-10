"""Qdrant/HNSW 向量后端适配器。"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from codeinsight.domain.semantic import SparseEmbedding
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.retrieval.vector_store import VectorPoint, VectorSearchHit


class QdrantVectorStore:
    """把统一 VectorStore 契约映射到一个 Qdrant collection。"""

    def __init__(
        self,
        *,
        client: QdrantClient,
        collection_name: str,
        dimensions: int,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 128,
        hnsw_ef_search: int = 64,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions 必须是正整数")
        self.client = client
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.hnsw_ef_search = hnsw_ef_search
        self._ensure_collection(hnsw_m, hnsw_ef_construction)

    def upsert(self, points: Iterable[VectorPoint]) -> None:
        values = tuple(points)
        for point in values:
            if len(point.vector) != self.dimensions:
                raise ValueError("向量维度与 Qdrant collection 不匹配")
            if point.sparse_vector is None:
                raise ValueError("Dense/Sparse Qdrant 点缺少 Provider Sparse 向量")
        with get_telemetry().span("vector", "upsert"):
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=_qdrant_id(point.point_id),
                        vector={
                            "dense": list(point.vector),
                            "sparse": _qdrant_sparse(point.sparse_vector),
                        },
                        payload={"_point_id": point.point_id, **dict(point.payload)},
                    )
                    for point in values
                ],
                wait=True,
            )

    def search_dense(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        if len(vector) != self.dimensions:
            raise ValueError("查询向量维度与 Qdrant collection 不匹配")
        if limit <= 0:
            raise ValueError("limit 必须是正整数")
        with get_telemetry().span("vector", "search"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=list(vector),
                using="dense",
                query_filter=_filter(query_filter),
                limit=limit,
                with_payload=True,
                search_params={"hnsw_ef": self.hnsw_ef_search},
            )
        hits: list[VectorSearchHit] = []
        for item in response.points:
            payload = dict(item.payload or {})
            point_id = str(payload.pop("_point_id", item.id))
            hits.append(VectorSearchHit(point_id, float(item.score), payload))
        return tuple(hits)

    def search_sparse(
        self,
        vector: SparseEmbedding,
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        if not vector.indices:
            return ()
        if limit <= 0:
            raise ValueError("limit 必须是正整数")
        with get_telemetry().span("vector", "search_sparse"):
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=_qdrant_sparse(vector),
                using="sparse",
                query_filter=_filter(query_filter),
                limit=limit,
                with_payload=True,
            )
        hits: list[VectorSearchHit] = []
        for item in response.points:
            payload = dict(item.payload or {})
            point_id = str(payload.pop("_point_id", item.id))
            hits.append(VectorSearchHit(point_id, float(item.score), payload))
        return tuple(hits)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        """2.0 兼容别名；默认明确查询命名 Dense 向量。"""
        return self.search_dense(vector, limit=limit, query_filter=query_filter)

    def delete(self, point_ids: Iterable[str]) -> None:
        ids = tuple(point_ids)
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=[_qdrant_id(point_id) for point_id in ids]),
            wait=True,
        )

    def health(self) -> bool:
        self.client.get_collections()
        return True

    def _ensure_collection(self, hnsw_m: int, hnsw_ef_construction: int) -> None:
        names = {item.name for item in self.client.get_collections().collections}
        if self.collection_name in names:
            info = self.client.get_collection(self.collection_name)
            vectors = getattr(info.config.params, "vectors", None)
            sparse_vectors = getattr(info.config.params, "sparse_vectors", None)
            if not isinstance(vectors, dict) or set(vectors) != {"dense"}:
                raise ValueError(
                    "已有 Qdrant collection 不是 Dense/Sparse Named Vectors；需要新 collection 迁移"
                )
            if not isinstance(sparse_vectors, dict) or set(sparse_vectors) != {"sparse"}:
                raise ValueError(
                    "已有 Qdrant collection 不是 Dense/Sparse Named Vectors；需要新 collection 迁移"
                )
            dense = vectors.get("dense")
            if getattr(dense, "size", None) != self.dimensions:
                raise ValueError("已有 Qdrant collection 的 Dense 向量维度不匹配")
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "dense": VectorParams(
                    size=self.dimensions,
                    distance=Distance.COSINE,
                    hnsw_config=HnswConfigDiff(
                        m=hnsw_m,
                        ef_construct=hnsw_ef_construction,
                    ),
                ),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(full_scan_threshold=10_000),
                )
            },
        )


def _qdrant_id(point_id: str) -> str:
    """把任意稳定业务 ID 映射成 Qdrant 接受的 UUID，同时保留原 ID 于 payload。"""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, point_id))


def _qdrant_sparse(vector: SparseEmbedding) -> SparseVector:
    return SparseVector(indices=list(vector.indices), values=list(vector.values))


def _filter(query_filter: Mapping[str, object] | None):
    if not query_filter:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    return Filter(
        must=[
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in query_filter.items()
        ]
    )
