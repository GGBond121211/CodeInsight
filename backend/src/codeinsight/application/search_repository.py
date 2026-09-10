"""端到端的 Dense/Sparse 仓库搜索用例。"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import (
    ApiException,
    ResponseHandlingException,
    UnexpectedResponse,
)

from codeinsight.domain.errors import QdrantNotConfiguredError, QdrantUnavailableError
from codeinsight.domain.retrieval import RankedChunk
from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.infrastructure.reranker import Reranker
from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.hybrid import rerank_ranked_chunks
from codeinsight.retrieval.index_pipeline import publish_semantic_index
from codeinsight.retrieval.qdrant_index_publisher import QdrantIndexPublisher
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.semantic import build_semantic_index, search_chunks_dense
from codeinsight.retrieval.sparse import search_chunks_sparse
from codeinsight.retrieval.vector_store import VectorStore

SemanticEmbed = Callable[[Sequence[str]], EmbeddingBatch]

# Q-003：两路各自粗召回 40，融合候选上限 100，默认最终返回 10。
DENSE_CANDIDATE_LIMIT = 40
SPARSE_CANDIDATE_LIMIT = 40
HYBRID_CANDIDATE_LIMIT = 100
DEFAULT_FINAL_TOP_K = 10
DEFAULT_RRF_K = 60
DEFAULT_QDRANT_COLLECTION = "codeinsight_2_1_dense_sparse"
_COLLECTION_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def candidate_retrieval_modes(primary_mode: str) -> tuple[str, ...]:
    """返回规划模式对应的 Dense/Sparse 候选来源。"""
    mode = _normalize_mode(primary_mode)
    if mode == "hybrid":
        return ("dense", "sparse")
    if mode in {"dense", "sparse"}:
        return (mode,)
    raise ValueError(f"不支持的检索模式：{primary_mode}")


def build_repository_semantic_index(
    root: str | Path,
    *,
    chunk_max_lines: int = 80,
    chunk_overlap_ratio: float = 0.0,
    chunk_max_tokens: int | None = None,
    semantic_embed: SemanticEmbed,
    require_sparse: bool = True,
) -> SemanticIndex:
    """扫描、切块并构建内存中的证据映射；向量持久化由 Qdrant 完成。"""
    scan_result = scan_repository(root)
    chunks = chunk_scan_result(
        scan_result,
        max_lines=chunk_max_lines,
        overlap_ratio=chunk_overlap_ratio,
        max_tokens=chunk_max_tokens,
    )
    return build_semantic_index(
        chunks,
        semantic_embed,
        require_sparse=require_sparse,
    )


def repository_id(root: str | Path) -> str:
    """生成不含源码内容的稳定仓库身份。"""
    return hashlib.sha256(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:20]


def create_runtime_vector_store(
    root: str | Path,
    *,
    dimensions: int,
    model: str,
) -> QdrantVectorStore:
    """创建 2.1 运行时唯一向量后端：Qdrant Named Dense/Sparse。

    未配置 URL 时使用 Qdrant client 的进程内模式，便于离线契约测试；正式部署
    应显式配置 ``CODEINSIGHT_QDRANT_URL``，不会切换到本地 JSON。
    """
    client = _runtime_qdrant_client()
    prefix = _runtime_collection_prefix()
    model_id = _COLLECTION_SAFE.sub("_", model).strip("_")[:48] or "embedding"
    collection = f"{prefix}_{repository_id(root)}_{model_id}"
    try:
        return QdrantVectorStore(
            client=client,
            collection_name=collection,
            dimensions=dimensions,
        )
    except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
        raise QdrantUnavailableError("Qdrant collection 不可用") from error


def prepare_runtime_vector_store(
    root: str | Path,
    index: SemanticIndex,
    *,
    chunk_version: str = "fixed-lines-v1",
) -> QdrantVectorStore:
    """复用 active Qdrant，否则创建 staging 并原子切换 active manifest。"""
    root_path = Path(root).resolve()
    repo_id = repository_id(root_path)
    source_fingerprints = _repository_source_fingerprints(root_path)
    try:
        publisher = QdrantIndexPublisher(
            client=_runtime_qdrant_client(),
            repository_root=root_path,
            repo_id=repo_id,
            collection_prefix=_runtime_collection_prefix(),
        )
        active = publisher.active()
        if (
            active is not None
            and active.repo_id == repo_id
            and active.index_version == index.metadata.index_id
            and active.chunk_version == chunk_version
            and active.vector_schema_version == index.metadata.vector_schema_version
            and _qdrant_collection_exists(publisher.client, active.collection)
        ):
            existing = publisher.active_store()
            if existing is not None:
                return existing
        staged = publisher.stage(
            index,
            source_fingerprints=source_fingerprints,
            chunk_version=chunk_version,
        )
        publisher.publish(staged.manifest.index_version)
        return staged.store
    except (ApiException, ResponseHandlingException, UnexpectedResponse, OSError) as error:
        raise QdrantUnavailableError("Qdrant active/staging collection 不可用") from error


def publish_repository_semantic_index(
    root: str | Path,
    index: SemanticIndex,
    store: VectorStore,
    *,
    chunk_version: str = "fixed-lines-v1",
) -> None:
    """把当前源码版本发布到已创建的 Qdrant/测试向量后端。"""
    scan_result = scan_repository(root)
    source_fingerprints = {
        source.relative_path: hashlib.sha256(source.text.encode("utf-8")).hexdigest()
        for source in scan_result.files
    }
    publish_semantic_index(
        index,
        store,
        repo_id=repository_id(root),
        source_fingerprints=source_fingerprints,
        chunk_version=chunk_version,
        require_sparse=True,
    )


def retrieve_subquestion_evidence(
    root: str | Path,
    question: str,
    *,
    primary_mode: str,
    limit: int = DEFAULT_FINAL_TOP_K,
    chunk_max_lines: int = 80,
    chunk_overlap_ratio: float = 0.0,
    semantic_embed: SemanticEmbed | None = None,
    semantic_index: SemanticIndex | None = None,
    semantic_store: VectorStore | None = None,
    reranker: Reranker | None = None,
    search=None,
) -> tuple[RankedChunk, ...]:
    """为一个子问题独立召回 Dense/Sparse 候选并执行最终 Rerank。"""
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    mode = _normalize_mode(primary_mode)
    candidate_retrieval_modes(mode)
    if search is not None and search is not search_repository:
        return search(
            root,
            question,
            limit=limit,
            chunk_max_lines=chunk_max_lines,
            chunk_overlap_ratio=chunk_overlap_ratio,
            retrieval_mode=mode,
            semantic_embed=semantic_embed,
            semantic_index=semantic_index,
            semantic_store=semantic_store,
            reranker=reranker,
        )
    if semantic_embed is None or semantic_index is None:
        return search_repository(
            root,
            question,
            limit=limit,
            chunk_max_lines=chunk_max_lines,
            chunk_overlap_ratio=chunk_overlap_ratio,
            retrieval_mode=mode,
            semantic_embed=semantic_embed,
            reranker=reranker,
        )
    store = semantic_store
    if store is None:
        store = prepare_runtime_vector_store(root, semantic_index)
    return _retrieve_from_index(
        root,
        question,
        mode=mode,
        limit=limit,
        semantic_index=semantic_index,
        semantic_embed=semantic_embed,
        semantic_store=store,
        reranker=reranker,
    )


def search_repository(
    root: str | Path,
    question: str,
    *,
    limit: int = DEFAULT_FINAL_TOP_K,
    chunk_max_lines: int = 80,
    chunk_overlap_ratio: float = 0.0,
    retrieval_mode: str = "hybrid",
    semantic_embed: SemanticEmbed | None = None,
    semantic_index: SemanticIndex | None = None,
    semantic_store: VectorStore | None = None,
    reranker: Reranker | None = None,
) -> tuple[RankedChunk, ...]:
    """扫描仓库并返回可回到源码的 Dense/Sparse 证据。

    2.1 正常 ``hybrid`` 路径只使用 Provider Dense + Provider Sparse + RRF +
    Rerank；向量数据和索引生命周期由 Qdrant 管理。
    """
    if limit <= 0:
        raise ValueError("limit 必须是正整数")
    mode = _normalize_mode(retrieval_mode)
    candidate_retrieval_modes(mode)
    scan_result = scan_repository(root)
    chunks = chunk_scan_result(
        scan_result,
        max_lines=chunk_max_lines,
        overlap_ratio=chunk_overlap_ratio,
    )
    if semantic_embed is None:
        raise ValueError("Dense/Sparse 检索需要 Embedding 模型")
    index = semantic_index or build_semantic_index(
        chunks,
        semantic_embed,
        require_sparse=True,
    )
    store = semantic_store
    if store is None:
        store = prepare_runtime_vector_store(root, index)
    return _retrieve_from_index(
        root,
        question,
        mode=mode,
        limit=limit,
        semantic_index=index,
        semantic_embed=semantic_embed,
        semantic_store=store,
        reranker=reranker,
    )


def _retrieve_from_index(
    root: str | Path,
    question: str,
    *,
    mode: str,
    limit: int,
    semantic_index: SemanticIndex,
    semantic_embed: SemanticEmbed,
    semantic_store: VectorStore,
    reranker: Reranker | None,
) -> tuple[RankedChunk, ...]:
    query_filter: Mapping[str, object] = {
        "repoId": repository_id(root),
        "indexVersion": semantic_index.metadata.index_id,
        "visibility": "active",
    }
    sources: list[tuple[str, tuple[RankedChunk, ...]]] = []
    if mode in {"hybrid", "dense"}:
        sources.append(
            (
                "dense",
                search_chunks_dense(
                    question,
                    semantic_index,
                    semantic_embed,
                    limit=DENSE_CANDIDATE_LIMIT,
                    vector_store=semantic_store,
                    query_filter=dict(query_filter),
                ),
            )
        )
    if mode in {"hybrid", "sparse"}:
        sources.append(
            (
                "sparse",
                search_chunks_sparse(
                    question,
                    semantic_index,
                    semantic_embed,
                    limit=SPARSE_CANDIDATE_LIMIT,
                    vector_store=semantic_store,
                    query_filter=dict(query_filter),
                ),
            )
        )
    if mode != "hybrid":
        return tuple(item for _, results in sources for item in results[:limit])
    if reranker is None:
        raise ValueError("Dense/Sparse hybrid 检索必须配置模型 Rerank")
    return rerank_ranked_chunks(
        question,
        sources,
        reranker=reranker,
        limit=limit,
        candidate_limit=HYBRID_CANDIDATE_LIMIT,
        rrf_k=DEFAULT_RRF_K,
        source_weights={"dense": 1.0, "sparse": 1.0},
    )


def _normalize_mode(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "auto":
        return "hybrid"
    return normalized


# 2.0 评测脚本仍可能导入这个内部函数；保留显式历史入口，但不把它用于新主路。
def _search_scanned_repository(
    scan_result,
    question: str,
    *,
    limit: int,
    chunk_max_lines: int,
    retrieval_mode: str,
    semantic_embed: SemanticEmbed | None,
    semantic_index: SemanticIndex | None,
    semantic_store: VectorStore | None,
    reranker: Reranker | None,
) -> tuple[RankedChunk, ...]:
    chunks = chunk_scan_result(scan_result, max_lines=chunk_max_lines)
    mode = _normalize_mode(retrieval_mode)
    candidate_retrieval_modes(mode)
    if semantic_embed is None:
        raise ValueError("Dense/Sparse 检索需要 Embedding 模型")
    index = semantic_index or build_semantic_index(
        chunks,
        semantic_embed,
        require_sparse=True,
    )
    store = semantic_store
    if store is None:
        store = create_runtime_vector_store(
            "__scanned_repository__",
            dimensions=index.metadata.dimensions,
            model=index.metadata.model,
        )
    return _retrieve_from_index(
        "__scanned_repository__",
        question,
        mode=mode,
        limit=limit,
        semantic_index=index,
        semantic_embed=semantic_embed,
        semantic_store=store,
        reranker=reranker,
    )


def _runtime_collection_prefix() -> str:
    prefix = os.environ.get("CODEINSIGHT_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
    return _COLLECTION_SAFE.sub("_", prefix).strip("_") or DEFAULT_QDRANT_COLLECTION


def _runtime_qdrant_client() -> QdrantClient:
    source = os.environ
    url = source.get("CODEINSIGHT_QDRANT_URL", "").strip()
    api_key = source.get("CODEINSIGHT_QDRANT_API_KEY", "").strip() or None
    timeout = int(source.get("CODEINSIGHT_QDRANT_TIMEOUT", "10"))
    if url:
        return QdrantClient(url=url, api_key=api_key, timeout=timeout)
    environment = source.get("CODEINSIGHT_ENV", "development").strip().lower()
    if environment in {"prod", "production", "release"}:
        raise QdrantNotConfiguredError(
            "生产运行时必须配置 CODEINSIGHT_QDRANT_URL；不回退本地 JSON 或进程内向量"
        )
    # Development/test only: this is Qdrant's in-process backend. Persistent
    # deployments must set the URL.
    return QdrantClient(location=":memory:")


def _repository_source_fingerprints(root: str | Path) -> dict[str, str]:
    scan_result = scan_repository(root)
    return {
        source.relative_path: hashlib.sha256(source.text.encode("utf-8")).hexdigest()
        for source in scan_result.files
    }


def _qdrant_collection_exists(client: QdrantClient, collection: str) -> bool:
    return collection in {item.name for item in client.get_collections().collections}
