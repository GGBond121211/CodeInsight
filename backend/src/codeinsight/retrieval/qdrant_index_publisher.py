"""Qdrant staging/validate/publish/rollback 生命周期。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from codeinsight.domain.errors import QdrantUnavailableError
from codeinsight.domain.semantic import SemanticIndex
from codeinsight.retrieval.index_manifest import IndexManifest
from codeinsight.retrieval.index_pipeline import (
    REQUIRED_PAYLOAD_FIELDS,
    build_index_manifest,
    publish_semantic_index,
    source_hash_from_fingerprints,
    vector_points_from_semantic_index,
)
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.vector_store import VectorPoint

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


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
    """用 Qdrant collection 和 alias 持有索引状态，不写本地 JSON。"""

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
        self.active_alias = f"{safe_prefix}_active_{repo_id}"
        self.previous_alias = f"{safe_prefix}_previous_{repo_id}"
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search
        self._staged: dict[str, IndexManifest] = {}

    def stage(
        self,
        index: SemanticIndex,
        *,
        source_fingerprints: Mapping[str, str],
        chunk_version: str,
    ) -> QdrantStagedIndex:
        """写入新 collection 并完成发布前对账；不触碰当前 active。"""
        collection = self._staging_collection(index.metadata.index_id)
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
        manifest = build_index_manifest(
            index_version=index.metadata.index_id,
            repo_id=self.repo_id,
            backend="qdrant",
            collection=collection,
            index=index,
            chunk_version=chunk_version,
            source_hash=source_hash_from_fingerprints(source_fingerprints),
        )
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
        self._staged[manifest.index_version] = manifest
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
        """原子切换 Qdrant active alias；失败不会覆盖旧 active。"""
        manifest = self._staged.get(index_version)
        if manifest is None:
            collection = self._staging_collection(index_version)
            manifest = self._manifest_from_collection(collection)
        if manifest is None:
            raise ValueError(f"Qdrant staging collection 不存在或缺少元数据：{index_version}")
        self._publish_alias(manifest.collection)
        self._staged.pop(index_version, None)
        return manifest

    def active(self) -> IndexManifest | None:
        collection = self._alias_collection(self.active_alias)
        if collection is None:
            return None
        return self._manifest_from_collection(collection)

    def active_store(self) -> QdrantVectorStore | None:
        manifest = self.active()
        if manifest is None or not self._collection_exists(manifest.collection):
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
        current = self._alias_collection(self.active_alias)
        previous = self._alias_collection(self.previous_alias)
        if current is None or previous is None:
            raise ValueError("没有可回滚的上一份 active index")
        try:
            self.client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=self.previous_alias)
                    ),
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=self.active_alias)
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=previous,
                            alias_name=self.active_alias,
                        )
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=current,
                            alias_name=self.previous_alias,
                        )
                    ),
                ]
            )
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant active alias 回滚失败") from error
        manifest = self._manifest_from_collection(previous)
        if manifest is None:
            raise ValueError("回滚后的 Qdrant collection 缺少索引元数据")
        return manifest

    def _publish_alias(self, collection: str) -> None:
        if not self._collection_exists(collection):
            raise ValueError(f"Qdrant collection 不存在：{collection}")
        current = self._alias_collection(self.active_alias)
        actions: list[object] = []
        previous = self._alias_collection(self.previous_alias)
        if previous is not None:
            actions.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.previous_alias)
                )
            )
        if current is not None:
            actions.append(
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=current,
                        alias_name=self.previous_alias,
                    )
                )
            )
        if current is not None:
            actions.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.active_alias)
                )
            )
        actions.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=collection,
                    alias_name=self.active_alias,
                )
            )
        )
        try:
            self.client.update_collection_aliases(actions)
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant active alias 发布失败") from error

    def _alias_collection(self, alias_name: str) -> str | None:
        try:
            aliases = self.client.get_aliases().aliases
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant alias 无法读取") from error
        for item in aliases:
            if item.alias_name == alias_name:
                return item.collection_name
        return None

    def _manifest_from_collection(self, collection: str) -> IndexManifest | None:
        if not self._collection_exists(collection):
            return None
        try:
            info = self.client.get_collection(collection)
            records = self.client.scroll(
                collection_name=collection,
                limit=1,
                with_payload=True,
                with_vectors=False,
            )[0]
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant collection 元数据无法读取") from error
        if not records:
            return None
        payload = dict(records[0].payload or {})
        try:
            repo_id = _required_text(payload, "repoId")
            index_version = _required_text(payload, "indexVersion")
            embedding_model = _required_text(payload, "embeddingModel")
            chunk_version = _required_text(payload, "chunkVersion")
            vector_schema_version = _required_text(payload, "vectorSchemaVersion")
            source_hash = _required_text(payload, "sourceHash")
            vectors = getattr(info.config.params, "vectors", None)
            dense = vectors.get("dense") if isinstance(vectors, dict) else None
            dimension = int(getattr(dense, "size", 0))
            distance = str(getattr(getattr(dense, "distance", None), "value", "cosine"))
        except (TypeError, ValueError):
            return None
        if repo_id != self.repo_id or dimension <= 0:
            return None
        return IndexManifest.create(
            index_version=index_version,
            repo_id=repo_id,
            backend="qdrant",
            collection=collection,
            embedding_model=embedding_model,
            dimension=dimension,
            distance=distance,
            chunk_version=chunk_version,
            source_hash=source_hash,
            vector_schema_version=vector_schema_version,
            sparse_model="provider-sparse",
        )

    def _collection_exists(self, collection: str) -> bool:
        try:
            return collection in {item.name for item in self.client.get_collections().collections}
        except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
            raise QdrantUnavailableError("Qdrant collection 列表无法读取") from error

    def _staging_collection(self, index_version: str) -> str:
        version = _SAFE_NAME.sub("_", index_version).strip("_")[:48]
        return f"{self.collection_prefix}_staging_{self.repo_id}_{version}"


def _required_text(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Qdrant payload 缺少 {name}")
    return value


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
