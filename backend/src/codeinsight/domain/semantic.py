"""Immutable values for evidence-preserving semantic retrieval."""

from dataclasses import dataclass

from codeinsight.domain.source import SourceChunk


@dataclass(frozen=True)
class EmbeddingBatch:
    """Embedding vectors returned by one provider call."""

    model: str
    vectors: tuple[tuple[float, ...], ...]
    input_tokens: int | None

    @property
    def dimensions(self) -> int:
        return len(self.vectors[0]) if self.vectors else 0


@dataclass(frozen=True)
class SemanticModelMetadata:
    """Public metadata needed to identify a semantic index."""

    model: str
    dimensions: int
    language_coverage: str
    service: str
    index_id: str


@dataclass(frozen=True)
class SemanticIndexEntry:
    """One vector linked back to a stable source chunk and fingerprint."""

    chunk_id: str
    chunk: SourceChunk
    embedding: tuple[float, ...]
    source_fingerprint: str
    metadata: SemanticModelMetadata


@dataclass(frozen=True)
class SemanticIndex:
    """An in-process semantic index; no persistence or database is implied."""

    metadata: SemanticModelMetadata
    entries: tuple[SemanticIndexEntry, ...]
