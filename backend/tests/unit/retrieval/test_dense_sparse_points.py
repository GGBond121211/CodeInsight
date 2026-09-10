"""Qdrant Named Dense/Sparse 向量契约测试。"""

import pytest
from qdrant_client import QdrantClient

from codeinsight.domain.semantic import SparseEmbedding
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.vector_store import VectorPoint


def test_qdrant_named_vectors_support_dense_sparse_queries_and_filters() -> None:
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="dense_sparse_contract",
        dimensions=2,
    )
    store.upsert(
        [
            VectorPoint(
                "a",
                (1.0, 0.0),
                {"repoId": "repo-a", "visibility": "active"},
                SparseEmbedding((4,), (1.0,)),
            ),
            VectorPoint(
                "b",
                (0.0, 1.0),
                {"repoId": "repo-b", "visibility": "active"},
                SparseEmbedding((5,), (1.0,)),
            ),
        ]
    )

    assert store.search_dense((0.9, 0.1), limit=1)[0].point_id == "a"
    assert store.search_sparse(SparseEmbedding((4,), (1.0,)), limit=1)[0].point_id == "a"
    assert store.search_dense(
        (0.9, 0.1), query_filter={"repoId": "repo-b"}
    )[0].point_id == "b"


def test_qdrant_named_vectors_reject_dense_only_points() -> None:
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="dense_sparse_required",
        dimensions=2,
    )

    with pytest.raises(ValueError, match="Sparse"):
        store.upsert([VectorPoint("a", (1.0, 0.0), {})])
