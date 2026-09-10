"""Qdrant staging/validate/publish/rollback 生命周期。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from codeinsight.domain.errors import QdrantUnavailableError
from codeinsight.domain.semantic import SemanticIndex
from codeinsight.retrieval.index_manifest import IndexManifest, IndexPublisher
from codeinsight.retrieval.index_pipeline import (
    REQUIRED_PAYLOAD_FIELDS,
    build_index_manifest,
    publish_semantic_index,
    source_fingerprint,
    vector_points_from_semantic_index,
)
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.vector_store import VectorPoint

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
CONTROL_DIR = ".codeinsight/qdrant-index"


@dataclass(frozen=True)
class QdrantValidation:
    valid: bool
    point_count: int
    unique_point_count: int
    reasons: tuple[str, ...] = ()
    sample_query_ok: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "point_count": self.point_count,
            "unique_point_count": self.unique_point_count,
            "reasons": list(self.reasons),
            "sample_query_ok": self.sample_query_ok,
        }


@dataclass(frozen=True)
class QdrantStagedIndex:
    manifest: IndexManifest
    store: QdrantVectorStore
    validation: QdrantValidation


class QdrantIndexPublisher:
    """用小型非向量 manifest 记录 active collection，向量始终留在 Qdrant。"""

    def __init__(
        self,
        *,
        client: QdrantClient,
        repository_root: str | Path,
        repo_id: str,
        collection_prefix: str = "codeinsight_2_1_dense_sparse",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 128,
        hnsw_ef_search: int = 64,
    ) -> None:
        root = Path(repository_root).resolve()
        if not root.is_dir():
            raise ValueError("repository_root 必须是目录")
        if not repo_id.strip():
            raise ValueError("repo_id 不能为空")
        safe_prefix = _SAFE_NAME.sub("_", collection_prefix).strip("_")
        if not safe_prefix:
            raise ValueError("collection_prefix 不能为空")
        self.client = client
        self.repository_root = root
        self.repo_id = repo_id
        self.collection_prefix = safe_prefix
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self._control = IndexPublisher(root / CONTROL_DIR)

    def stage(
        self,
        index: SemanticIndex,
        *,
        source_fingerprints: Mapping[str, str],
        chunk_version: str,
    ) -> QdrantStagedIndex:
        """写入新 collection 并完成发布前对账；不触碰当前 active。"""
        collection = self._staging_collection(index)
        store = QdrantVectorStore(
            client=self.client,
            collection_name=collection,
            dimensions=index.metadata.dimensions,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
        )
        points = vector_points_from_semantic_index(
            index,
            repo_id=self.repo_id,
            source_fingerprints=source_fingerprints,
            chunk_version=chunk_version,
        )
        source_hash = _source_hash(source_fingerprints)
        manifest = build_index_manifest(
            index_version=index.metadata.index_id,
            repo_id=self.repo_id,
            backend="qdrant",
            collection=collection,
            index=index,
            chunk_version=chunk_version,
            source_hash=source_hash,
        )
        # publish_semantic_index performs payload/schema checks before upsert and
        # requires Provider Sparse for this 2.1 path.
        publish_semantic_index(
            index,
            store,
            repo_id=self.repo_id,
            source_fingerprints=source_fingerprints,
            chunk_version=chunk_version,
            require_sparse=True,
        )
        validation = self.validate(
            manifest,
            store,
            expected_points=points,
            sample_vector=points[0].vector if points else None,
        )
        if not validation.valid:
            raise ValueError(f"Qdrant staging 校验失败：{'; '.join(validation.reasons)}")
        self._control.stage(manifest)
        return QdrantStagedIndex(manifest, store, validation)

    def validate(
        self,
        manifest: IndexManifest,
        store: QdrantVectorStore,
        *,
        expected_points: Iterable[VectorPoint],
        sample_vector: tuple[float, ...] | None = None,
    ) -> QdrantValidation:
        expected = tuple(expected_points)
        reasons: list[str] = []
        expected_ids = [point.point_id for point in expected]
        if len(expected_ids) != len(set(expected_ids)):
            reasons.append("expected point_id 不唯一")
        if any(set(point.payload) != set(REQUIRED_PAYLOAD_FIELDS) for point in expected):
            reasons.append("expected payload 字段不完整")
        try:
            count = int(self.client.count(collection_name=manifest.collection, exact=True).count)
            records = _scroll_all(self.client, manifest.collection)
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant collection 无法读取") from error
        record_ids = [
            str(dict(record.payload or {}).get("_point_id", record.id))
            for record in records
        ]
        if count != len(expected):
            reasons.append("point count 与预期不一致")
        if len(record_ids) != len(set(record_ids)):
            reasons.append("Qdrant point_id 不唯一")
        if set(record_ids) != set(expected_ids):
            reasons.append("Qdrant point_id 集合与预期不一致")
        expected_by_id = {point.point_id: point for point in expected}
        for record in records:
            payload = dict(record.payload or {})
            point_id = str(payload.get("_point_id", record.id))
            expected_point = expected_by_id.get(point_id)
            if expected_point is None:
                continue
            for name in REQUIRED_PAYLOAD_FIELDS:
                if payload.get(name) != expected_point.payload.get(name):
                    reasons.append(f"payload {name} 与预期不一致")
                    break
        sample_query_ok = False
        if sample_vector is not None:
            try:
                hits = store.search_dense(
                    sample_vector,
                    limit=1,
                    query_filter={
                        "repoId": manifest.repo_id,
                        "indexVersion": manifest.index_version,
                        "visibility": "active",
                    },
                )
                sample_query_ok = bool(hits)
            except Exception:
                sample_query_ok = False
            if not sample_query_ok:
                reasons.append("sample query 无法返回 active Evidence")
        return QdrantValidation(
            valid=not reasons,
            point_count=count,
            unique_point_count=len(set(record_ids)),
            reasons=tuple(dict.fromkeys(reasons)),
            sample_query_ok=sample_query_ok,
        )

    def publish(self, index_version: str) -> IndexManifest:
        """只切换小型 active manifest；失败不会覆盖旧 active。"""
        return self._control.publish(index_version)

    def active(self) -> IndexManifest | None:
        return self._control.active()

    def active_store(self) -> QdrantVectorStore | None:
        manifest = self.active()
        if manifest is None:
            return None
        if manifest.collection not in {
            item.name for item in self.client.get_collections().collections
        }:
            return None
        return QdrantVectorStore(
            client=self.client,
            collection_name=manifest.collection,
            dimensions=manifest.dimension,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            hnsw_ef_search=self.hnsw_ef_search,
        )

    def rollback(self) -> IndexManifest:
        return self._control.rollback()

    def _staging_collection(self, index: SemanticIndex) -> str:
        version = _SAFE_NAME.sub("_", index.metadata.index_id).strip("_")[:48]
        return f"{self.collection_prefix}_staging_{self.repo_id}_{version}"


def _source_hash(source_fingerprints: Mapping[str, str]) -> str:
    material = "\n".join(
        f"{path}\0{fingerprint}" for path, fingerprint in sorted(source_fingerprints.items())
    )
    return source_fingerprint(material)


def _scroll_all(client: QdrantClient, collection: str) -> tuple[object, ...]:
    records: list[object] = []
    offset = None
    while True:
        page, offset = client.scroll(
            collection_name=collection,
            limit=1000,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        records.extend(page)
        if offset is None:
            return tuple(records)
