"""对仓库证据执行 RRF 候选融合和模型重排序。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.infrastructure.reranker import Reranker

_RRF_K = 60
_SOURCE_WEIGHTS = {
    "dense": 1.0,
    "sparse": 1.0,
    # 2.0 historical names remain readable for old evaluation artifacts only.
    "bm25": 1.0,
    "semantic": 1.0,
}


def _chunk_key(result: RankedChunk) -> tuple[str, int, int]:
    return (
        result.chunk.relative_path,
        result.chunk.start_line,
        result.chunk.end_line,
    )


def _fuse(
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int,
    rrf_k: int = _RRF_K,
    source_weights: Mapping[str, float] | None = None,
) -> tuple[RankedChunk, ...]:
    """只做候选融合，不执行最终精排。"""
    if rrf_k <= 0:
        raise ValueError("rrf_k 必须是正整数")
    weights = _SOURCE_WEIGHTS if source_weights is None else source_weights
    scores: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    chunks: dict[tuple[str, int, int], RankedChunk] = {}
    reasons: defaultdict[tuple[str, int, int], set[str]] = defaultdict(set)
    source_names: defaultdict[tuple[str, int, int], set[str]] = defaultdict(set)
    for source_name, results in sources:
        weight = weights.get(source_name, 1.0)
        for result in results:
            key = _chunk_key(result)
            scores[key] += weight / (rrf_k + result.rank)
            chunks[key] = result
            reasons[key].add(result.retrieval_reason)
            source_names[key].add(source_name)

    fused: list[tuple[float, tuple[str, int, int]]] = []
    for key, base_score in scores.items():
        agreement = min(1.0, (len(source_names[key]) - 1) / 2)
        # agreement 仍属于多路召回融合信号，不是最终精排。
        score = base_score * 100.0 + agreement * 0.12
        fused.append((score, key))

    def sort_key(item: tuple[float, tuple[str, int, int]]) -> tuple[float, str, int, int]:
        score, key = item
        return (-score, key[0], key[1], key[2])

    ordered = sorted(fused, key=sort_key)[:limit]

    ranked_chunks: list[RankedChunk] = []
    for rank, (score, key) in enumerate(ordered, start=1):
        retrieval_reason = "hybrid_match"
        if len(reasons[key]) <= 1:
            retrieval_reason = next(iter(reasons[key]))
        ranked_chunks.append(
            RankedChunk(
                chunk=chunks[key].chunk,
                score=score,
                rank=rank,
                retrieval_reason=retrieval_reason,
            )
        )
    return tuple(ranked_chunks)


def _rerank_documents(results: Sequence[RankedChunk]) -> tuple[str, ...]:
    """把代码身份和正文一起交给 Rerank，避免同名符号失去文件上下文。"""
    documents: list[str] = []
    for result in results:
        chunk = result.chunk
        symbol = chunk.symbol_path or "(无符号信息)"
        documents.append(
            f"Path: {chunk.relative_path}\n"
            f"Lines: {chunk.start_line}-{chunk.end_line}\n"
            f"Symbol: {symbol}\n"
            f"Code:\n{chunk.text}"
        )
    return tuple(documents)


def rerank_ranked_chunks(
    question: str,
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    reranker: Reranker,
    limit: int = 5,
    candidate_limit: int = 100,
    rrf_k: int = _RRF_K,
    source_weights: Mapping[str, float] | None = None,
) -> tuple[RankedChunk, ...]:
    """将 RRF 候选交给模型 Rerank，并恢复为可信的 RankedChunk。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if candidate_limit <= 0:
        raise ValueError("candidate_limit 必须是正整数")
    if not sources:
        return ()
    candidates = _fuse(
        sources,
        limit=min(candidate_limit, sum(len(results) for _, results in sources)),
        rrf_k=rrf_k,
        source_weights=source_weights,
    )
    if not candidates:
        return ()
    # Rerank 服务的上限由适配器再次校验；这里也限制候选数，避免任何调用方
    # 意外构造超过供应商上限的请求。
    if len(candidates) > 500:
        candidates = candidates[:500]
    ranked = reranker.rerank(
        question,
        _rerank_documents(candidates),
        top_n=min(limit, len(candidates)),
    )
    selected: list[RankedChunk] = []
    for rank, item in enumerate(ranked, start=1):
        candidate = candidates[item.index]
        selected.append(
            RankedChunk(
                chunk=candidate.chunk,
                score=item.relevance_score,
                rank=rank,
                retrieval_reason=candidate.retrieval_reason,
            )
        )
    return tuple(selected)


def fuse_ranked_chunks(
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int = 5,
    rrf_k: int = _RRF_K,
    source_weights: Mapping[str, float] | None = None,
) -> tuple[RankedChunk, ...]:
    """只执行加权 RRF 融合，供模型精排前的候选阶段使用。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if not sources:
        return ()
    return _fuse(
        sources,
        limit=limit,
        rrf_k=rrf_k,
        source_weights=source_weights,
    )
