"""端到端的仓库搜索用例。"""

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
    """为一个已规划的子问题返回确定性的候选检索模式。"""
    if primary_mode not in {"lexical", "bm25", "hybrid"}:
        raise ValueError(f"不支持的检索模式：{primary_mode}")
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
    """加载或增量构建一个保留证据映射的语义索引。"""
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
    """独立检索并重排一个子问题的证据。

    Router 选择的稀疏模式仍然是主模式。如果共享语义索引可用，则加入语义候选，
    不重复构建索引。可选的 ``search`` hook 用于保持 Agent 测试和应用层注入边界不变。
    """
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    search_fn = search or search_repository
    candidate_limit = max(limit, min(HYBRID_CANDIDATE_LIMIT, limit * 4))
    modes = candidate_retrieval_modes(primary_mode)
    sources: list[tuple[str, tuple[RankedChunk, ...]]] = []
    for mode in modes:
        results = search_fn(
            root,
            question,
            limit=candidate_limit,
            chunk_max_lines=chunk_max_lines,
            retrieval_mode=mode,
            semantic_embed=None,
            semantic_index=None,
        )
        sources.append((mode, results))
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
            raise ValueError("hybrid 检索需要 Embedding 模型")
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
    raise ValueError(f"不支持的检索模式：{retrieval_mode}")


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
    """扫描 *root*、切分文件，并返回 *question* 的前 *limit* 个匹配结果。

    这是只读应用用例：扫描 -> 切分 -> 搜索。
    下层异常会有意向上抛出；这里不额外引入缓存、重试、配置或接口。
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
