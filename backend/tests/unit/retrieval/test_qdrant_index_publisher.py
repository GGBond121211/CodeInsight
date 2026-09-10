"""Qdrant staging、校验、active 指针和回滚测试。"""

import pytest
from qdrant_client import QdrantClient

from codeinsight.application.search_repository import create_runtime_vector_store
from codeinsight.domain.errors import ModelConfigurationError
from codeinsight.domain.semantic import (
    SemanticIndex,
    SemanticIndexEntry,
    SemanticModelMetadata,
    SparseEmbedding,
)
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.index_pipeline import source_fingerprint
from codeinsight.retrieval.qdrant_index_publisher import QdrantIndexPublisher


def _index(text: str) -> SemanticIndex:
    metadata = SemanticModelMetadata(
        "embedding-a", 2, "multilingual", "test", f"index-{text}"
    )
    chunk = SourceChunk("src/app.py", 1, 1, text, "run")
    entry = SemanticIndexEntry(
        f"point-{text}",
        chunk,
        (1.0, 0.0),
        source_fingerprint(text),
        metadata,
        SparseEmbedding((1,), (1.0,)),
    )
    return SemanticIndex(metadata, (entry,))


def test_qdrant_staging_does_not_replace_active_until_publish_and_can_rollback(tmp_path) -> None:
    publisher = QdrantIndexPublisher(
        client=QdrantClient(location=":memory:"),
        repository_root=tmp_path,
        repo_id="repo-a",
    )
    first = _index("return 1")
    fingerprints = {"src/app.py": source_fingerprint("return 1")}

    staged_first = publisher.stage(
        first,
        source_fingerprints=fingerprints,
        chunk_version="fixed-lines-v1",
    )

    assert staged_first.validation.valid
    assert publisher.active() is None
    published_first = publisher.publish(staged_first.manifest.index_version)
    assert published_first.collection == staged_first.manifest.collection
    assert publisher.active_store() is not None

    second = _index("return 2")
    staged_second = publisher.stage(
        second,
        source_fingerprints={"src/app.py": source_fingerprint("return 2")},
        chunk_version="fixed-lines-v1",
    )
    publisher.publish(staged_second.manifest.index_version)

    assert publisher.active().index_version == staged_second.manifest.index_version
    assert publisher.rollback().index_version == staged_first.manifest.index_version


def test_production_requires_explicit_qdrant_url(monkeypatch) -> None:
    monkeypatch.setenv("CODEINSIGHT_ENV", "production")
    monkeypatch.delenv("CODEINSIGHT_QDRANT_URL", raising=False)

    with pytest.raises(ModelConfigurationError, match="QDRANT_URL"):
        create_runtime_vector_store(".", dimensions=2, model="embedding-a")
