"""Auto Answer 使用的共享检索引擎集成测试。"""

from pathlib import Path

import pytest

from codeinsight.application.search_repository import repository_id, search_repository
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult
from codeinsight.ingestion.chunker import chunk_source_file
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval.index_pipeline import publish_semantic_index, source_fingerprint
from codeinsight.retrieval.semantic import build_semantic_index
from codeinsight.retrieval.vector_store import LocalJsonVectorStore

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


@pytest.mark.parametrize("retrieval_mode", ["lexical", "bm25"])
def test_internal_sparse_retrieval_resolves_within_fixture(retrieval_mode: str) -> None:
    results = search_repository(
        FIXTURE_ROOT,
        "Where is checkout defined?",
        limit=5,
        retrieval_mode=retrieval_mode,
    )

    result_paths = []
    for item in results:
        result_paths.append(item.chunk.relative_path)
    assert "src/shop/service.py" in result_paths


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


def test_search_repository_can_use_an_explicit_vector_store(tmp_path) -> None:
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
    store = LocalJsonVectorStore(tmp_path / "vectors.json")
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


@pytest.mark.parametrize("removed_mode", ["ast-bm25", "graph-bm25"])
def test_removed_retrieval_modes_are_rejected(removed_mode: str) -> None:
    with pytest.raises(ValueError, match="不支持的检索模式"):
        search_repository(FIXTURE_ROOT, "Where is checkout?", retrieval_mode=removed_mode)
