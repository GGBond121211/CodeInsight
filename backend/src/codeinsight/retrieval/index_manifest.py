"""Qdrant 索引身份值对象；索引状态由 Qdrant collection/alias 持有。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True)
class IndexManifest:
    """描述一份可被 Qdrant 安全复用的索引。"""

    index_version: str
    repo_id: str
    backend: str
    collection: str
    embedding_model: str
    dimension: int
    distance: str
    chunk_version: str
    source_hash: str
    updated_at: str
    vector_schema_version: str = "dense-v1"
    sparse_model: str = ""

    @classmethod
    def create(
        cls,
        *,
        index_version: str,
        repo_id: str,
        backend: str,
        collection: str,
        embedding_model: str,
        dimension: int,
        distance: str,
        chunk_version: str,
        source_hash: str,
        vector_schema_version: str = "dense-v1",
        sparse_model: str = "",
    ) -> IndexManifest:
        return cls(
            index_version=index_version,
            repo_id=repo_id,
            backend=backend,
            collection=collection,
            embedding_model=embedding_model,
            dimension=dimension,
            distance=distance,
            chunk_version=chunk_version,
            source_hash=source_hash,
            updated_at=datetime.now(UTC).isoformat(),
            vector_schema_version=vector_schema_version,
            sparse_model=sparse_model,
        )

    def __post_init__(self) -> None:
        if self.backend != "qdrant":
            raise ValueError("IndexManifest 只支持 Qdrant backend")
        for name in (
            "index_version",
            "repo_id",
            "backend",
            "collection",
            "embedding_model",
            "distance",
            "chunk_version",
            "source_hash",
            "updated_at",
            "vector_schema_version",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} 不能为空")
        if self.dimension <= 0:
            raise ValueError("dimension 必须是正整数")
