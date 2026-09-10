"""Qdrant 索引身份值对象测试。"""

import pytest

from codeinsight.retrieval.index_manifest import IndexManifest


def _manifest(backend: str = "qdrant") -> IndexManifest:
    return IndexManifest.create(
        index_version="v1",
        repo_id="demo",
        backend=backend,
        collection="collection-v1",
        embedding_model="embedding-a",
        dimension=2,
        distance="cosine",
        chunk_version="fixed-lines-v1",
        source_hash="hash-v1",
    )


def test_manifest_is_an_in_memory_qdrant_identity() -> None:
    manifest = _manifest()

    assert manifest.backend == "qdrant"
    assert manifest.collection == "collection-v1"


def test_manifest_requires_qdrant_backend() -> None:
    with pytest.raises(ValueError, match="只支持 Qdrant"):
        _manifest("file")
