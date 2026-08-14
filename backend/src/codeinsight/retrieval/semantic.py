"""In-process cosine semantic retrieval with evidence-preserving results."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence

from codeinsight.domain.retrieval import SEMANTIC_MATCH_REASON, RankedChunk
from codeinsight.domain.semantic import (
    EmbeddingBatch,
    SemanticIndex,
    SemanticIndexEntry,
    SemanticModelMetadata,
)
from codeinsight.domain.source import SourceChunk

EmbeddingFunction = Callable[[Sequence[str]], EmbeddingBatch]


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk_id(chunk: SourceChunk) -> str:
    identity = f"{chunk.relative_path}:{chunk.start_line}:{chunk.end_line}:{chunk.text}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _validate_batch(batch: EmbeddingBatch, expected_count: int) -> int:
    if len(batch.vectors) != expected_count:
        raise ValueError("embedding provider returned an unexpected vector count")
    if not batch.vectors:
        raise ValueError("semantic index requires at least one embedding")
    dimensions = len(batch.vectors[0])
    if dimensions == 0:
        raise ValueError("embedding vectors must not be empty")
    if any(len(vector) != dimensions for vector in batch.vectors):
        raise ValueError("embedding vectors must have equal dimensions")
    if any(not math.isfinite(value) for vector in batch.vectors for value in vector):
        raise ValueError("embedding vectors must contain finite values")
    return dimensions


def build_semantic_index(
    chunks: Sequence[SourceChunk],
    embed: EmbeddingFunction,
    *,
    language_coverage: str = "multilingual",
    service: str = "openai-compatible",
) -> SemanticIndex:
    """Embed *chunks* once and build an evidence-preserving in-memory index."""
    if not chunks:
        raise ValueError("semantic index requires at least one source chunk")
    batch = embed(tuple(chunk.text for chunk in chunks))
    dimensions = _validate_batch(batch, len(chunks))
    entry_ids = tuple(_chunk_id(chunk) for chunk in chunks)
    index_material = "|".join((batch.model, *entry_ids))
    index_id = hashlib.sha256(index_material.encode("utf-8")).hexdigest()[:20]
    metadata = SemanticModelMetadata(
        model=batch.model,
        dimensions=dimensions,
        language_coverage=language_coverage,
        service=service,
        index_id=index_id,
    )
    entries = tuple(
        SemanticIndexEntry(
            chunk_id=chunk_id,
            chunk=chunk,
            embedding=vector,
            source_fingerprint=_fingerprint(chunk.text),
            metadata=metadata,
        )
        for chunk_id, chunk, vector in zip(entry_ids, chunks, batch.vectors, strict=True)
    )
    return SemanticIndex(metadata=metadata, entries=entries)


def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("query and indexed embedding dimensions do not match")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


def search_chunks_semantic(
    query: str,
    index: SemanticIndex,
    embed: EmbeddingFunction,
    *,
    limit: int = 5,
    min_score: float = 0.20,
) -> tuple[RankedChunk, ...]:
    """Return high-enough cosine matches while preserving source evidence."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not query.strip():
        return ()
    query_batch = embed((query,))
    _validate_batch(query_batch, 1)
    if query_batch.dimensions != index.metadata.dimensions:
        raise ValueError("query embedding dimensions do not match semantic index")
    scored = [
        (entry, _cosine_similarity(query_batch.vectors[0], entry.embedding))
        for entry in index.entries
    ]
    scored = [item for item in scored if item[1] >= min_score]
    scored.sort(
        key=lambda item: (
            -item[1],
            item[0].chunk.relative_path,
            item[0].chunk.start_line,
            item[0].chunk_id,
        )
    )
    return tuple(
        RankedChunk(
            chunk=entry.chunk,
            score=score,
            rank=rank,
            retrieval_reason=SEMANTIC_MATCH_REASON,
        )
        for rank, (entry, score) in enumerate(scored[:limit], start=1)
    )
