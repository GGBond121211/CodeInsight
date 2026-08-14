"""保留证据映射的语义检索不可变值对象。"""

from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk


@dataclass(frozen=True)
class EmbeddingBatch:
    """一次提供方调用返回的 Embedding 向量。"""

    model: str
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int | None

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True)
class SemanticModelMetadata:
    """用于标识语义索引的公开元数据。"""

    model: str
    dimensions: int
    language_coverage: str
    service: str
    index_id: str


@dataclass(frozen=True)
class SemanticIndexEntry:
    """一个关联到稳定源码块和指纹的向量。"""

    chunk_id: str
    chunk: SourceChunk
    embedding: tuple[float, ...]
    source_fingerprint: str
    metadata: SemanticModelMetadata


@dataclass(frozen=True)
class SemanticIndex:
    """进程内语义索引；不代表存在持久化或数据库。"""

    metadata: SemanticModelMetadata
    entries: tuple[SemanticIndexEntry, ...]
