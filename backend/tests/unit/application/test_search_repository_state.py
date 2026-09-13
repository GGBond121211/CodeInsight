"""检索状态复用的单元测试：调用方已经持有索引时，不许再重建整仓库索引。"""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from codeinsight.application import search_repository as module
from codeinsight.application.search_repository import (
    CHUNK_VERSION,
    build_repository_semantic_index,
    prepare_search_state,
    repository_id,
    reuse_active_index,
    search_repository,
)
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.retrieval.index_pipeline import source_fingerprint
from codeinsight.retrieval.qdrant_index_publisher import QdrantIndexPublisher
from codeinsight.retrieval.semantic import search_chunks_dense
from codeinsight.retrieval.sparse import search_chunks_sparse


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


class _FakeEmbedder:
    """只按文本长度给可区分的向量；这里测身份与复用，不测检索质量。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        return EmbeddingBatch(
            "fake-embedding",
            tuple((1.0, float(len(text) % 7)) for text in texts),
            len(texts),
            tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
            "dense-sparse-v1",
        )


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _publish(root: Path, client, embedder, *, chunk_version: str = CHUNK_VERSION):
    """在测试用的进程内 Qdrant 上发布一份索引，模拟上一次运行留下的 active。"""

    index = build_repository_semantic_index(root, semantic_embed=embedder)
    publisher = QdrantIndexPublisher(
        client=client, repository_root=root, repo_id=repository_id(root)
    )
    staged = publisher.stage(
        index,
        source_fingerprints={"src/app.py": source_fingerprint("def run():\n    return 1\n")},
        chunk_version=chunk_version,
    )
    publisher.publish(staged.manifest.index_version)
    return index


def test_reuse_active_index_skips_embedding_and_keeps_identity(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    client = QdrantClient(location=":memory:")
    embedder = _FakeEmbedder()
    published = _publish(root, client, embedder)
    assert embedder.calls == 1

    reused = reuse_active_index(root, client=client)

    assert reused is not None
    index, store = reused
    assert store is not None
    assert index.vectors_persisted is True
    assert index.metadata.index_id == published.metadata.index_id
    assert index.metadata.dimensions == published.metadata.dimensions
    assert [entry.chunk_id for entry in index.entries] == [
        entry.chunk_id for entry in published.entries
    ]
    assert all(entry.embedding == () for entry in index.entries)
    # 复用路径一次 Embedding 都不该发出去。
    assert embedder.calls == 1


def test_reuse_active_index_rejects_changed_source(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    client = QdrantClient(location=":memory:")
    _publish(root, client, _FakeEmbedder())
    (root / "src" / "app.py").write_text("def run():\n    return 2\n", encoding="utf-8")

    assert reuse_active_index(root, client=client) is None


def test_reuse_active_index_rejects_mismatched_chunk_version(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    client = QdrantClient(location=":memory:")
    _publish(root, client, _FakeEmbedder(), chunk_version="other-chunking-v9")

    assert reuse_active_index(root, client=client) is None


def test_reuse_active_index_needs_an_explicit_qdrant_url(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("CODEINSIGHT_QDRANT_URL", raising=False)

    assert reuse_active_index(_repo(tmp_path)) is None


def test_prepare_search_state_prefers_the_published_index(tmp_path: Path, monkeypatch) -> None:
    reused_state = ("published-index", "published-store")
    monkeypatch.setattr(module, "reuse_active_index", lambda root, **kwargs: reused_state)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("有可复用的 active 索引时不允许重新 Embedding 整仓库")

    monkeypatch.setattr(module, "build_repository_semantic_index", forbidden)

    assert prepare_search_state(tmp_path, semantic_embed=lambda texts: None) == reused_state


def test_vectorless_index_refuses_local_exact_search(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    client = QdrantClient(location=":memory:")
    _publish(root, client, _FakeEmbedder())
    index, _store = reuse_active_index(root, client=client)

    with pytest.raises(ValueError, match="持久化在 Qdrant"):
        search_chunks_dense("run 定义在哪？", index, _FakeEmbedder())

    with pytest.raises(ValueError, match="持久化在 Qdrant"):
        search_chunks_sparse("run 定义在哪？", index, _FakeEmbedder())
