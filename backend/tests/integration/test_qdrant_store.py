"""Qdrant 适配器集成测试；未启动本地 Qdrant 时跳过。"""

import os

import pytest
from qdrant_client import QdrantClient

from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.vector_store import VectorPoint


def _store() -> QdrantVectorStore:
    url = os.environ.get("CODEINSIGHT_QDRANT_URL", "http://127.0.0.1:6335")
    client = QdrantClient(url=url, timeout=3)
    try:
        client.get_collections()
    except Exception as error:
        pytest.skip(f"本地 Qdrant 不可用：{type(error).__name__}")
    return QdrantVectorStore(
        client=client,
        collection_name="codeinsight_test_vectors",
        dimensions=2,
        hnsw_m=8,
        hnsw_ef_construction=64,
        hnsw_ef_search=32,
    )


def test_qdrant_upsert_search_filter_delete_and_health() -> None:
    store = _store()
    store.delete(["point-a", "point-b"])
    store.upsert(
        [
            VectorPoint("point-a", (1.0, 0.0), {"path": "a.py", "language": "py"}),
            VectorPoint("point-b", (0.0, 1.0), {"path": "b.py", "language": "py"}),
        ]
    )

    assert store.health()
    assert store.search((0.95, 0.05), limit=1)[0].point_id == "point-a"
    assert store.search((0.95, 0.05), query_filter={"path": "missing.py"}) == ()
    store.delete(["point-a"])
    assert all(item.point_id != "point-a" for item in store.search((1.0, 0.0), limit=2))
