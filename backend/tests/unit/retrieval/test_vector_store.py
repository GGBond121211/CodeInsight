"""向量后端契约和 local_json exact 测试。"""

from codeinsight.retrieval.vector_store import (
    LocalJsonVectorStore,
    VectorPoint,
)


def test_local_json_upsert_search_filter_delete_and_reload(tmp_path) -> None:
    path = tmp_path / "vectors.json"
    store = LocalJsonVectorStore(path)
    store.upsert(
        [
            VectorPoint("a", (1.0, 0.0), {"path": "a.py", "language": "py"}),
            VectorPoint("b", (0.0, 1.0), {"path": "b.md", "language": "md"}),
        ]
    )

    assert store.health()
    assert store.search((0.9, 0.1), limit=1)[0].point_id == "a"
    filtered = store.search((0.0, 1.0), query_filter={"language": "py"})
    assert [item.point_id for item in filtered] == ["a"]
    reloaded = LocalJsonVectorStore(path)
    assert reloaded.search((1.0, 0.0), limit=2)[0].payload["path"] == "a.py"
    reloaded.delete(["a"])
    assert [item.point_id for item in reloaded.search((1.0, 0.0), limit=2)] == ["b"]
