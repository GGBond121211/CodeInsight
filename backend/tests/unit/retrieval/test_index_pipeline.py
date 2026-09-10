"""Qdrant 索引业务层元数据和增量计划测试。"""

from qdrant_client import QdrantClient

from codeinsight.domain.semantic import (
    SemanticIndex,
    SemanticIndexEntry,
    SemanticModelMetadata,
    SparseEmbedding,
)
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.index_pipeline import (
    plan_incremental_update,
    publish_semantic_index,
    source_fingerprint,
    validate_vector_points,
    vector_points_from_semantic_index,
)
from codeinsight.retrieval.qdrant_store import QdrantVectorStore
from codeinsight.retrieval.vector_store import VectorPoint


def _index() -> SemanticIndex:
    metadata = SemanticModelMetadata("embedding-a", 2, "multilingual", "test", "index")
    chunk = SourceChunk("src/app.py", 3, 5, "def run():\n    return 1", "run")
    entry = SemanticIndexEntry(
        "point-a",
        chunk,
        (1.0, 0.0),
        "chunk-hash",
        metadata,
        SparseEmbedding((1,), (0.5,)),
    )
    return SemanticIndex(metadata, (entry,))


def test_vector_points_carry_evidence_and_repository_metadata() -> None:
    points = vector_points_from_semantic_index(
        _index(),
        repo_id="repo-a",
        source_fingerprints={"src/app.py": source_fingerprint("source")},
        chunk_version="structured-v1",
    )

    validate_vector_points(points)
    validate_vector_points(points, require_sparse=True)
    assert points[0].point_id == "point-a"
    assert points[0].payload["path"] == "src/app.py"
    assert points[0].payload["startLine"] == 3
    assert points[0].payload["language"] == "python"
    assert points[0].payload["module"] == "src.app"
    assert points[0].payload["indexVersion"] == "index"
    assert points[0].payload["embeddingModel"] == "embedding-a"
    assert points[0].payload["vectorSchemaVersion"] == "dense-sparse-v1"
    assert points[0].sparse_vector == SparseEmbedding((1,), (0.5,))


def test_incremental_plan_deletes_changed_file_and_keeps_unchanged_file() -> None:
    old = (
        VectorPoint("old-a", (1.0, 0.0), {"path": "a.py", "sourceFingerprint": "old"}),
        VectorPoint("old-b", (0.0, 1.0), {"path": "b.py", "sourceFingerprint": "same"}),
    )
    current = (
        VectorPoint("new-a", (0.0, 1.0), {"path": "a.py", "sourceFingerprint": "new"}),
        VectorPoint("new-b", (0.0, 1.0), {"path": "b.py", "sourceFingerprint": "same"}),
    )

    plan = plan_incremental_update(old, current)

    assert plan.delete == ("old-a",)
    assert [point.point_id for point in plan.upsert] == ["new-a"]
    assert plan.unchanged_paths == ("b.py",)


def test_publish_validates_payload_before_writing_qdrant() -> None:
    store = QdrantVectorStore(
        client=QdrantClient(location=":memory:"),
        collection_name="index_pipeline_test",
        dimensions=2,
    )
    points = publish_semantic_index(
        _index(),
        store,
        repo_id="repo-a",
        source_fingerprints={"src/app.py": "source-hash"},
        chunk_version="structured-v1",
        require_sparse=True,
    )

    assert points[0].payload["visibility"] == "active"
    assert store.search_dense((1.0, 0.0), limit=1)[0].point_id == "point-a"
