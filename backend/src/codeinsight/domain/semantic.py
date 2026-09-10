"""保留证据映射的 Dense/Sparse 检索不可变值对象。"""

import math
from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk


@dataclass(frozen=True)
class SparseEmbedding:
    """一个稀疏向量；索引升序且不重复，便于映射到 Qdrant。"""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse embedding 的 indices/values 长度必须一致")
        previous = -1
        for index, value in zip(self.indices, self.values, strict=True):
            if isinstance(index, bool) or index < 0 or index <= previous:
                raise ValueError("Sparse embedding 的 indices 必须是升序非负整数")
            if not math.isfinite(value) or value == 0.0:
                raise ValueError("Sparse embedding 的 values 必须是非零有限数")
            previous = index


@dataclass(frozen=True)
class EmbeddingBatch:
    """一次提供方调用返回的 Dense/Sparse 向量。

    ``input_tokens`` 保持在第三个位置，兼容 2.0 的测试适配器。没有提供方
    稀疏向量时，批次明确保持 ``None``；生产 Dense/Sparse 路径必须拒绝该批次，
    不能用本地词袋或哈希编码冒充 Learned Sparse。
    """

    model: str
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int | None
    sparse_vectors: tuple[SparseEmbedding, ...] | None = None
    response_schema_version: str = "dense-v1"

    def __post_init__(self) -> None:
        if self.sparse_vectors is not None and len(self.sparse_vectors) != len(self.vectors):
            raise ValueError("Dense/Sparse embedding 的数量必须一致")
        if not self.response_schema_version.strip():
            raise ValueError("Embedding response schema version 不能为空")

    @property
    def has_sparse(self) -> bool:
        return self.sparse_vectors is not None

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0

    @property
    def dense_vectors(self) -> tuple[tuple[float, ...], ...]:
        """更明确的 Dense 别名；``vectors`` 保留旧调用方兼容性。"""
        return self.vectors


@dataclass(frozen=True)
class SemanticModelMetadata:
    """用于标识语义索引的公开元数据。"""

    model: str
    dimensions: int
    language_coverage: str
    service: str
    index_id: str
    sparse_model: str = "codeinsight-sparse-v1"
    vector_schema_version: str = "dense-sparse-v1"


@dataclass(frozen=True)
class SemanticIndexEntry:
    """一个关联到稳定源码块和指纹的向量。"""

    chunk_id: str
    chunk: SourceChunk
    embedding: tuple[float, ...]
    source_fingerprint: str
    metadata: SemanticModelMetadata
    sparse_embedding: SparseEmbedding | None = None


@dataclass(frozen=True)
class SemanticIndex:
    """进程内语义索引；不代表存在持久化或数据库。"""

    metadata: SemanticModelMetadata
    entries: tuple[SemanticIndexEntry, ...]
