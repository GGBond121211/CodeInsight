"""检索状态复用的单元测试：调用方已经持有索引时，不许再重建整仓库索引。"""

from pathlib import Path

from codeinsight.application import search_repository as module
from codeinsight.application.search_repository import prepare_search_state, search_repository


def test_search_repository_skips_rebuild_when_state_is_provided(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "src.py").write_text("value = 1\n", encoding="utf-8")
    captured: list[dict[str, object]] = []

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("给了索引和向量库时不允许重建整仓库索引")

    def fake_retrieve(root, question, **kwargs):
        captured.append(kwargs)
        return ()

    monkeypatch.setattr(module, "build_repository_semantic_index", forbidden)
    monkeypatch.setattr(module, "prepare_runtime_vector_store", forbidden)
    monkeypatch.setattr(module, "_retrieve_from_index", fake_retrieve)

    results = search_repository(
        tmp_path,
        "定义在哪里？",
        semantic_embed=lambda texts: None,
        semantic_index="index",
        semantic_store="store",
        reranker=object(),
    )

    assert results == ()
    assert captured[0]["semantic_index"] == "index"
    assert captured[0]["semantic_store"] == "store"


def test_prepare_search_state_publishes_the_index_it_built(
    tmp_path: Path, monkeypatch
) -> None:
    built = object()
    published: list[object] = []

    monkeypatch.setattr(
        module, "build_repository_semantic_index", lambda root, **kwargs: built
    )

    def fake_publish(root, index):
        published.append(index)
        return "store"

    monkeypatch.setattr(module, "prepare_runtime_vector_store", fake_publish)

    index, store = prepare_search_state(tmp_path, semantic_embed=lambda texts: None)

    assert index is built
    assert store == "store"
    assert published == [built]
