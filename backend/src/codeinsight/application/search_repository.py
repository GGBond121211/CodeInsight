"""End-to-end repository search use case."""

from collections.abc import Callable, Sequence
from pathlib import Path

from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.bm25 import search_chunks_bm25
from codeinsight.retrieval.hybrid import rerank_ranked_chunks
from codeinsight.retrieval.lexical import search_chunks as search_chunks_lexical
from codeinsight.retrieval.persistent_semantic import (
    build_persistent_semantic_index,
    embedding_model_id,
)
from codeinsight.retrieval.semantic import build_semantic_index, search_chunks_semantic

SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]
HYBRID_CANDIDATE_LIMIT = 20


def candidate_retrieval_modes(primary_mode: str) -> tuple[str, ...]:
    """Return deterministic candidate modes for one planned subquestion."""
    if primary_mode not in {"lexical", "bm25", "hybrid"}:
        raise ValueError(f"unsupported retrieval mode: {primary_mode}")
    if primary_mode == "hybrid":
        return ("bm25",)
    modes = [primary_mode, "bm25"]
    return tuple(dict.fromkeys(modes))


def build_repository_semantic_index(
    root: str | Path,
    *,
    chunk_max_lines: int = 80,
    semantic_embed: SemanticEmbed,
    cache_root: str | Path | None = None,
    semantic_model: str | None = None,
) -> SemanticIndex:
    """Load or incrementally build one evidence-preserving semantic index."""
    scan_result = scan_repository(root)
    model = semantic_model or embedding_model_id(semantic_embed)
    if model is not None:
        return build_persistent_semantic_index(
            root,
            scan_result,
            semantic_embed,
            model=model,
            chunk_max_lines=chunk_max_lines,
            cache_root=cache_root,
        )
    fixed_chunks = chunk_scan_result(scan_result, max_lines=chunk_max_lines)
    return build_semantic_index(fixed_chunks, semantic_embed)


def retrieve_subquestion_evidence(
    root: str | Path,
    question: str,
    *,
    primary_mode: str,
    limit: int = 5,
    chunk_max_lines: int = 80,
    semantic_embed: SemanticEmbed | None = None,
    semantic_index: SemanticIndex | None = None,
    search=None,
) -> tuple[RankedChunk, ...]:
    """Retrieve and rerank evidence independently for one subquestion.

    The Router-selected sparse mode remains primary. When a shared semantic
    index is available, semantic candidates are added without rebuilding the
    index. The optional ``search`` hook keeps Agent tests and application
    injection boundaries intact.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    search_fn = search or search_repository
    candidate_limit = max(limit, min(HYBRID_CANDIDATE_LIMIT, limit * 4))
    modes = candidate_retrieval_modes(primary_mode)
    sources = [
        (
            mode,
            search_fn(
                root,
                question,
                limit=candidate_limit,
                chunk_max_lines=chunk_max_lines,
                retrieval_mode=mode,
                semantic_embed=None,
                semantic_index=None,
            ),
        )
        for mode in modes
    ]
    if semantic_embed is not None and semantic_index is not None:
        sources.append(
            (
                "semantic",
                search_chunks_semantic(
                    question,
                    semantic_index,
                    semantic_embed,
                    limit=candidate_limit,
                ),
            )
        )
    return rerank_ranked_chunks(question, sources, limit=limit)


def _search_scanned_repository(
    scan_result,
    question: str,
    *,
    limit: int,
    chunk_max_lines: int,
    retrieval_mode: str,
    semantic_embed: SemanticEmbed | None,
    semantic_index: SemanticIndex | None,
) -> tuple[RankedChunk, ...]:
    fixed_chunks = chunk_scan_result(scan_result, max_lines=chunk_max_lines)
    if retrieval_mode == "hybrid":
        if semantic_embed is None:
            raise ValueError("hybrid retrieval requires an embedding model")
        index = semantic_index or build_semantic_index(fixed_chunks, semantic_embed)
        candidate_limit = max(limit, min(HYBRID_CANDIDATE_LIMIT, limit * 4))
        semantic_results = search_chunks_semantic(
            question,
            index,
            semantic_embed,
            limit=candidate_limit,
        )
        bm25_results = _search_scanned_repository(
            scan_result,
            question,
            limit=candidate_limit,
            chunk_max_lines=chunk_max_lines,
            retrieval_mode="bm25",
            semantic_embed=None,
            semantic_index=None,
        )
        return rerank_ranked_chunks(
            question,
            (
                ("bm25", bm25_results),
                ("semantic", semantic_results),
            ),
            limit=limit,
        )
    if retrieval_mode == "lexical":
        return search_chunks_lexical(question, fixed_chunks, limit=limit)
    if retrieval_mode == "bm25":
        return search_chunks_bm25(question, fixed_chunks, limit=limit)
    raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")


def search_repository(
    root: str | Path,
    question: str,
    *,
    limit: int = 5,
    chunk_max_lines: int = 80,
    retrieval_mode: str = "hybrid",
    semantic_embed: SemanticEmbed | None = None,
    semantic_index: SemanticIndex | None = None,
) -> tuple[RankedChunk, ...]:
    """Scan *root*, chunk it, and return the top *limit* matches for *question*.

    This is the read-only application use case: scan -> chunk -> search.
    Exceptions from the underlying layers are intentionally not caught, and no
    caching, retry, configuration, or extra interface is introduced.
    """
    scan_result = scan_repository(root)
    return _search_scanned_repository(
        scan_result,
        question,
        limit=limit,
        chunk_max_lines=chunk_max_lines,
        retrieval_mode=retrieval_mode,
        semantic_embed=semantic_embed,
        semantic_index=semantic_index,
    )
