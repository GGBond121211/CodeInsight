"""对仓库证据执行稳定的候选融合和代码感知重排。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.retrieval.tokenizer import tokenize_path, tokenize_query, tokenize_terms

_RRF_K = 60
_SOURCE_WEIGHTS = {
    "bm25": 1.0,
    "semantic": 0.8,
}


def _chunk_key(result: RankedChunk) -> tuple[str, int, int]:
    return (
        result.chunk.relative_path,
        result.chunk.start_line,
        result.chunk.end_line,
    )


def _code_relevance(
    question: str,
    result: RankedChunk,
    retrieval_reasons: set[str],
) -> float:
    """不请求另一个语言模型，直接为精确代码身份相关性评分。

    符号和路径匹配比偶然的正文匹配更强。这些信号在 RRF 之后工作，绝不会制造新证据。
    """
    query_tokens = set(tokenize_query(question))
    if not query_tokens:
        return 0.0
    symbol_tokens = set(tokenize_terms(result.chunk.symbol_path or ""))
    path_tokens = set(tokenize_path(result.chunk.relative_path))
    body_tokens = set(tokenize_terms(result.chunk.text))
    denominator = max(len(query_tokens), 1)
    symbol_overlap = len(query_tokens & symbol_tokens) / denominator
    path_overlap = len(query_tokens & path_tokens) / denominator
    body_overlap = len(query_tokens & body_tokens) / denominator
    return symbol_overlap * 0.12 + path_overlap * 0.08 + body_overlap * 0.05


def _rerank(
    question: str | None,
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int,
) -> tuple[RankedChunk, ...]:
    """融合候选，并应用少量可解释的代码相关性特征。

    这里有意保持确定性且不依赖模型。RRF 为每个检索器提供可比较的排名信号；
    精确查询词重叠和多来源一致性会进一步打破平局，让既可定位又有独立支持的
    仓库证据优先。
    """
    scores: defaultdict[tuple[str, int, int], float] = defaultdict(float)
    chunks: dict[tuple[str, int, int], RankedChunk] = {}
    reasons: defaultdict[tuple[str, int, int], set[str]] = defaultdict(set)
    source_names: defaultdict[tuple[str, int, int], set[str]] = defaultdict(set)
    for source_name, results in sources:
        weight = _SOURCE_WEIGHTS.get(source_name, 1.0)
        for result in results:
            key = _chunk_key(result)
            scores[key] += weight / (_RRF_K + result.rank)
            chunks[key] = result
            reasons[key].add(result.retrieval_reason)
            source_names[key].add(source_name)

    reranked: list[tuple[float, tuple[str, int, int]]] = []
    for key, base_score in scores.items():
        result = chunks[key]
        code_relevance = _code_relevance(question, result, reasons[key]) if question else 0.0
        agreement = min(1.0, (len(source_names[key]) - 1) / 2)
        # RRF 分数较小，因此把词法/结构特征作为打破平局的信号，避免它们占据主导。
        score = base_score * 100.0 + code_relevance + agreement * 0.12
        reranked.append((score, key))

    def sort_key(item: tuple[float, tuple[str, int, int]]) -> tuple[float, str, int, int]:
        score, key = item
        return (-score, key[0], key[1], key[2])

    ordered = sorted(reranked, key=sort_key)[:limit]

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


def rerank_ranked_chunks(
    question: str,
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int = 5,
) -> tuple[RankedChunk, ...]:
    """重排一个子问题独立生成的候选结果。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if not sources:
        return ()
    return _rerank(question, sources, limit=limit)


def fuse_ranked_chunks(
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int = 5,
) -> tuple[RankedChunk, ...]:
    """不使用查询特征，提供向后兼容的加权 RRF 融合。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    if not sources:
        return ()
    return _rerank(None, sources, limit=limit)
