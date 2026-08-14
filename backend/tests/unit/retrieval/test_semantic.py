"""确定性且保留证据映射的语义检索测试。"""

import pytest

from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.semantic import build_semantic_index, search_chunks_semantic


def _chunks() -> tuple[SourceChunk, ...]:
    return (
        SourceChunk("src/shop/shipping/workflow.py", 11, 13, "fragile courier lane"),
        SourceChunk("src/shop/pricing.py", 11, 14, "price times quantity"),
    )


class _FakeEmbedder:
    def __call__(self, texts):
        vectors = []
        for text in texts:
            if "fragile" in text or "专门的配送通道" in text:
                vectors.append((1.0, 0.0))
            elif "price" in text or "价格" in text:
                vectors.append((0.0, 1.0))
            else:
                vectors.append((0.0, 0.0))
        return EmbeddingBatch("fake-multilingual", tuple(vectors), 3)


def test_build_index_preserves_source_identity_and_metadata() -> None:
    index = build_semantic_index(_chunks(), _FakeEmbedder())

    assert len(index.entries) == 2
    assert index.metadata.model == "fake-multilingual"
    assert index.metadata.dimensions == 2
    assert index.metadata.language_coverage == "multilingual"
    assert index.metadata.index_id
    assert index.entries[0].chunk_id
    assert len(index.entries[0].source_fingerprint) == 64
    assert index.entries[0].chunk.relative_path == "src/shop/shipping/workflow.py"


def test_search_returns_evidence_with_semantic_reason() -> None:
    embedder = _FakeEmbedder()
    index = build_semantic_index(_chunks(), embedder)

    results = search_chunks_semantic("易碎件要走专门的配送通道", index, embedder, limit=1)

    assert len(results) == 1
    assert results[0].chunk.relative_path == "src/shop/shipping/workflow.py"
    assert results[0].retrieval_reason == "semantic_match"
    assert results[0].rank == 1


def test_low_similarity_returns_no_evidence() -> None:
    embedder = _FakeEmbedder()
    index = build_semantic_index(_chunks(), embedder)

    assert search_chunks_semantic("unrelated question", index, embedder) == ()


def test_index_rejects_inconsistent_embedding_count() -> None:
    def bad_embedder(texts):
        return EmbeddingBatch("fake", ((1.0, 0.0),), 1)

    with pytest.raises(ValueError, match="向量数量"):
        build_semantic_index(_chunks(), bad_embedder)


def test_query_dimension_mismatch_is_explicit() -> None:
    index = build_semantic_index(_chunks(), _FakeEmbedder())

    def wrong_dimension(_texts):
        return EmbeddingBatch("fake", ((1.0, 0.0, 0.0),), 1)

    with pytest.raises(ValueError, match="维度"):
        search_chunks_semantic("fragile", index, wrong_dimension)
