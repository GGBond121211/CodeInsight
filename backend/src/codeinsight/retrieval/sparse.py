"""Provider Learned Sparse 向量检索。

2.1.0 不把本地词袋、TF-IDF 或哈希编码命名为 Learned Sparse。Provider 只返回
Dense 时，调用方必须收到受控配置错误。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex, SparseEmbedding
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.vector_store import VectorStore

EmbeddingFunction = Callable[[Sequence[str]], EmbeddingBatch]
def require_sparse_batch(
    batch: EmbeddingBatch, expected_count: int
) -> tuple[SparseEmbedding, ...]:
    """严格要求 Provider 返回与 Dense 对齐的 Sparse 批次。"""
    if len(batch.vectors) != expected_count:
        raise ModelResponseError("Embedding 提供方返回的 Dense 数量不符合预期")
    if batch.sparse_vectors is None:
        raise ModelResponseError(
            "Embedding Provider 未返回 Sparse；2.1 Dense/Sparse 检索无法继续"
        )
    if len(batch.sparse_vectors) != expected_count:
        raise ModelResponseError("Embedding 提供方返回的 Sparse 数量不符合预期")
    return batch.sparse_vectors


def sparse_similarity(first: SparseEmbedding, second: SparseEmbedding) -> float:
    """计算两个稀疏向量的 cosine 相似度。"""
    if not first.indices or not second.indices:
        return 0.0
    first_values = dict(zip(first.indices, first.values, strict=True))
    second_values = dict(zip(second.indices, second.values, strict=True))
    dot = sum(
        first_values[index]
        * second_values[index]
        for index in first_values.keys() & second_values.keys()
    )
    first_norm = math.sqrt(sum(value * value for value in first.values))
    second_norm = math.sqrt(sum(value * value for value in second.values))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return dot / (first_norm * second_norm)


def search_chunks_sparse(
    query: str,
    index: SemanticIndex,
    embed: EmbeddingFunction,
    *,
    limit: int = 40,
    min_score: float = 0.0,
    vector_store: VectorStore | None = None,
    query_filter: dict[str, object] | None = None,
) -> tuple[RankedChunk, ...]:
    """按 sparse embedding 检索，并保留可信的源码证据映射。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if not query.strip():
        return ()
    query_batch = embed((query,))
    query_sparse = require_sparse_batch(query_batch, 1)[0]
    entries_by_id = {entry.chunk_id: entry for entry in index.entries}
    if vector_store is not None:
        hits = vector_store.search_sparse(
            query_sparse,
            limit=limit,
            query_filter=query_filter,
        )
        results: list[RankedChunk] = []
        for hit in hits:
            entry = entries_by_id.get(hit.point_id)
            if entry is None or hit.score < min_score:
                continue
            results.append(
                RankedChunk(
                    chunk=entry.chunk,
                    score=hit.score,
                    rank=len(results) + 1,
                    retrieval_reason="sparse_match",
                )
            )
        return tuple(results)

    scored: list[tuple[float, SourceChunk, str]] = []
    for entry in index.entries:
        if entry.sparse_embedding is None:
            raise ModelResponseError("索引缺少 Provider Sparse 向量，不能执行 Sparse 检索")
        score = sparse_similarity(query_sparse, entry.sparse_embedding)
        if score >= min_score:
            scored.append((score, entry.chunk, entry.chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1].relative_path, item[1].start_line, item[2]))
    return tuple(
        RankedChunk(chunk, score, rank, "sparse_match")
        for rank, (score, chunk, _) in enumerate(scored[:limit], start=1)
    )
