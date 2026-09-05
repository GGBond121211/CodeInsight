"""确定性且保留证据映射的语义检索测试。"""

import pytest

from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.semantic import (
    build_semantic_index,
    filter_indexable_chunks,
    search_chunks_semantic,
)
from codeinsight.retrieval.vector_store import LocalJsonVectorStore


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


def test_index_skips_whitespace_only_chunks_before_embedding() -> None:
    chunks = (
        SourceChunk("app.py", 1, 1, "\n  \t"),
        SourceChunk("app.py", 2, 2, "return value"),
    )
    received: list[tuple[str, ...]] = []

    def embedder(texts):
        values = tuple(texts)
        received.append(values)
        return EmbeddingBatch("fake", ((1.0, 0.0),), len(values))

    index = build_semantic_index(chunks, embedder)

    assert received == [("return value",)]
    assert [(entry.chunk.start_line, entry.chunk.end_line) for entry in index.entries] == [
        (2, 2)
    ]


def test_filter_indexable_chunks_keeps_order_and_original_ranges() -> None:
    chunks = (
        SourceChunk("app.py", 1, 1, "first"),
        SourceChunk("app.py", 2, 2, "\n"),
        SourceChunk("app.py", 3, 3, "third"),
    )

    assert filter_indexable_chunks(chunks) == (chunks[0], chunks[2])


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


def test_search_can_use_vector_store_without_changing_evidence_mapping(tmp_path) -> None:
    embedder = _FakeEmbedder()
    index = build_semantic_index(_chunks(), embedder)
    store = LocalJsonVectorStore(tmp_path / "vectors.json")
    from codeinsight.retrieval.index_pipeline import publish_semantic_index, source_fingerprint

    publish_semantic_index(
        index,
        store,
        repo_id="demo",
        source_fingerprints={
            "src/shop/shipping/workflow.py": source_fingerprint("shipping"),
            "src/shop/pricing.py": source_fingerprint("pricing"),
        },
        chunk_version="fixed-lines-v1",
    )

    results = search_chunks_semantic(
        "易碎件要走专门的配送通道",
        index,
        embedder,
        limit=1,
        vector_store=store,
        query_filter={"repoId": "demo", "visibility": "active"},
    )

    assert len(results) == 1
    assert results[0].chunk.relative_path == "src/shop/shipping/workflow.py"
    assert results[0].chunk.start_line == 11
