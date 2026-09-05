"""IndexManifest 和发布回滚测试。"""

import pytest

from codeinsight.retrieval.index_manifest import IndexManifest, IndexPublisher


def _manifest(version: str) -> IndexManifest:
    return IndexManifest.create(
        index_version=version,
        repo_id="demo",
        backend="local_json",
        collection=f"collection-{version}",
        embedding_model="embedding-a",
        dimension=2,
        distance="cosine",
        chunk_version="fixed-lines-v1",
        source_hash=f"hash-{version}",
    )


def test_stage_publish_and_rollback_keep_active_index_isolated(tmp_path) -> None:
    publisher = IndexPublisher(tmp_path / "indexes")
    first = _manifest("v1")
    second = _manifest("v2")

    publisher.stage(first)
    publisher.publish("v1")
    publisher.stage(second)
    assert publisher.active().index_version == "v1"
    publisher.publish("v2")
    assert publisher.active().index_version == "v2"
    assert publisher.rollback().index_version == "v1"


def test_invalid_staging_is_rejected_without_touching_active(tmp_path) -> None:
    publisher = IndexPublisher(tmp_path / "indexes")
    publisher.stage(_manifest("v1"))
    publisher.publish("v1")
    (publisher.staging / "bad.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError):
        publisher.publish("bad")
    assert publisher.active().index_version == "v1"
