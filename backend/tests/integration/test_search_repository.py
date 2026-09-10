"""Auto Answer 使用的共享检索引擎集成测试。"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from codeinsight.application.search_repository import repository_id, search_repository
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult
from codeinsight.ingestion.chunker import chunk_source_file
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.index_pipeline import publish_semantic_index, source_fingerprint
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.semantic import build_semantic_index

BACKEND_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = BACKEND_ROOT / "tests" / "fixtures" / "sample_repo"


def _fake_embed(texts):
    vectors = []
    for _ in texts:
        vectors.append((1.0, 0.0))
    return EmbeddingBatch(
        "fake",
        tuple(vectors),
        len(texts),
        tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
        "dense-sparse-v1",
    )


class _FakeReranker:
    def rerank(self, query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


def test_search_requires_embedding_provider() -> None:
    with pytest.raises(ValueError, match="Embedding"):
        search_repository(FIXTURE_ROOT, "Where is checkout defined?", retrieval_mode="hybrid")


def test_internal_hybrid_combines_dense_and_sparse_candidates() -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        limit=5,
        retrieval_mode="hybrid",
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
    )

    assert results
    expected_reasons = {"dense_match", "sparse_match", "hybrid_match"}
    for item in results:
        assert item.retrieval_reason in expected_reasons


def test_search_repository_can_use_an_explicit_qdrant_store() -> None:
    scan_result = scan_repository(FIXTURE_ROOT)
    index = build_semantic_index(
        tuple(
            # Use the same fixed-size source chunks as the application path.
            item
            for source in scan_result.files
            for item in chunk_source_file(source)
        ),
        _fake_embed,
    )
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="search_repository_test",
        dimensions=2,
    )
    publish_semantic_index(
        index,
        store,
        repo_id=repository_id(FIXTURE_ROOT),
        source_fingerprints={
            source.relative_path: source_fingerprint(source.text)
            for source in scan_result.files
        },
        chunk_version="fixed-lines-v1",
    )

    results = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        limit=5,
        retrieval_mode="hybrid",
        semantic_embed=_fake_embed,
        reranker=_FakeReranker(),
        semantic_index=index,
        semantic_store=store,
    )

    assert results
    assert all(item.chunk.relative_path for item in results)


@pytest.mark.parametrize("removed_mode", ["keyword", "unsupported"])
def test_unsupported_retrieval_modes_are_rejected(removed_mode: str) -> None:
    with pytest.raises(ValueError, match="不支持的检索模式"):
        search_repository(FIXTURE_ROOT, "Where is checkout?", retrieval_mode=removed_mode)
