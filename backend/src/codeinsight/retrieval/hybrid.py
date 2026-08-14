"""Stable candidate fusion and code-aware reranking for repository evidence."""

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
    """Score exact code identity without asking another language model.

    Symbol and path matches are stronger than incidental body matches. These
    signals operate after RRF and never manufacture new evidence.
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
    """Fuse candidates and apply small explainable code-relevance features.

    This is intentionally deterministic and model-free.  RRF gives each
    retriever a comparable rank signal; exact query-token overlap and
    multi-source agreement then break ties in favor of repository evidence
    that is both identifiable and independently supported.
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
        # RRF scores are small, so keep the lexical/structural features as
        # tie-break-strength signals rather than allowing them to dominate.
        score = base_score * 100.0 + code_relevance + agreement * 0.12
        reranked.append((score, key))

    ordered = sorted(
        reranked,
        key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2]),
    )[:limit]
    return tuple(
        RankedChunk(
            chunk=chunks[key].chunk,
            score=score,
            rank=rank,
            retrieval_reason=(
                "hybrid_match" if len(reasons[key]) > 1 else next(iter(reasons[key]))
            ),
        )
        for rank, (score, key) in enumerate(ordered, start=1)
    )


def rerank_ranked_chunks(
    question: str,
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int = 5,
) -> tuple[RankedChunk, ...]:
    """Rerank independently generated candidates for one subquestion."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not sources:
        return ()
    return _rerank(question, sources, limit=limit)


def fuse_ranked_chunks(
    sources: Sequence[tuple[str, Sequence[RankedChunk]]],
    *,
    limit: int = 5,
) -> tuple[RankedChunk, ...]:
    """Backward-compatible weighted RRF fusion without query features."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    if not sources:
        return ()
    return _rerank(None, sources, limit=limit)
