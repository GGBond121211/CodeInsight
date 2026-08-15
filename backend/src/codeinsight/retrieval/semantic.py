"""进程内的余弦相似度语义检索，并保留证据映射。"""

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
        raise ValueError("Embedding 提供方返回的向量数量不符合预期")
    if not batch.vectors:
        raise ValueError("语义索引至少需要一个 Embedding")
    dimensions = len(batch.vectors[0])
    if dimensions == 0:
        raise ValueError("Embedding 向量不能为空")
    for vector in batch.vectors:
        if len(vector) != dimensions:
            raise ValueError("Embedding 向量的维度必须一致")
        for value in vector:
            if not math.isfinite(value):
                raise ValueError("Embedding 向量必须包含有限值")
    return dimensions


def build_semantic_index(
    chunks: Sequence[SourceChunk],
    embed: EmbeddingFunction,
    *,
    language_coverage: str = "multilingual",
    service: str = "openai-compatible",
) -> SemanticIndex:
    """对 *chunks* 一次性生成 Embedding，并构建保留证据映射的内存索引。"""
    if not chunks:
        raise ValueError("语义索引至少需要一个源码块")
    chunk_texts: list[str] = []
    for chunk in chunks:
        chunk_texts.append(chunk.text)
    batch = embed(tuple(chunk_texts))
    dimensions = _validate_batch(batch, len(chunks))
    entry_id_list: list[str] = []
    for chunk in chunks:
        entry_id_list.append(_chunk_id(chunk))
    entry_ids = tuple(entry_id_list)
    index_material = "|".join((batch.model, *entry_ids))
    index_id = hashlib.sha256(index_material.encode("utf-8")).hexdigest()[:20]
    metadata = SemanticModelMetadata(
        model=batch.model,
        dimensions=dimensions,
        language_coverage=language_coverage,
        service=service,
        index_id=index_id,
    )
    entry_list: list[SemanticIndexEntry] = []
    for chunk_id, chunk, vector in zip(
        entry_ids,
        chunks,
        batch.vectors,
        strict=True,
    ):
        entry_list.append(
            SemanticIndexEntry(
                chunk_id=chunk_id,
                chunk=chunk,
                embedding=vector,
                source_fingerprint=_fingerprint(chunk.text),
                metadata=metadata,
            )
        )
    entries = tuple(entry_list)
    return SemanticIndex(metadata=metadata, entries=entries)


def _cosine_similarity(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("查询向量与索引向量的维度不匹配")
    first_squared_sum = 0.0
    for value in first:
        first_squared_sum += value * value
    second_squared_sum = 0.0
    for value in second:
        second_squared_sum += value * value
    first_norm = math.sqrt(first_squared_sum)
    second_norm = math.sqrt(second_squared_sum)
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    dot_product = 0.0
    for left, right in zip(first, second, strict=True):
        dot_product += left * right
    return dot_product / (first_norm * second_norm)


def search_chunks_semantic(
    query: str,
    index: SemanticIndex,
    embed: EmbeddingFunction,
    *,
    limit: int = 5,
    min_score: float = 0.20,
) -> tuple[RankedChunk, ...]:
    """返回余弦相似度足够高的匹配结果，同时保留源码证据。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if not query.strip():
        return ()
    query_batch = embed((query,))
    _validate_batch(query_batch, 1)
    if query_batch.dimensions != index.metadata.dimensions:
        raise ValueError("查询 Embedding 的维度与语义索引不匹配")
    scored: list[tuple[SemanticIndexEntry, float]] = []
    for entry in index.entries:
        score = _cosine_similarity(
            query_batch.vectors[0],
            entry.embedding,
        )
        scored.append((entry, score))

    filtered_scores: list[tuple[SemanticIndexEntry, float]] = []
    for item in scored:
        if item[1] >= min_score:
            filtered_scores.append(item)
    scored = filtered_scores

    def sort_key(item: tuple[SemanticIndexEntry, float]) -> tuple[float, str, int, str]:
        entry, score = item
        return (
            -score,
            entry.chunk.relative_path,
            entry.chunk.start_line,
            entry.chunk_id,
        )

    scored.sort(key=sort_key)

    ranked_chunks: list[RankedChunk] = []
    for rank, (entry, score) in enumerate(scored[:limit], start=1):
        ranked_chunks.append(
            RankedChunk(
                chunk=entry.chunk,
                score=score,
                rank=rank,
                retrieval_reason=SEMANTIC_MATCH_REASON,
            )
        )
    return tuple(ranked_chunks)
