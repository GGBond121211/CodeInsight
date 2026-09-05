"""索引业务层元数据和增量计划测试。"""

from codeinsight.domain.semantic import (
    SemanticIndex,
    SemanticIndexEntry,
    SemanticModelMetadata,
)
from codeinsight.domain.source import SourceChunk
from codeinsight.retrieval.index_pipeline import (
    plan_incremental_update,
    publish_semantic_index,
    source_fingerprint,
    validate_vector_points,
    vector_points_from_semantic_index,
)
from codeinsight.retrieval.vector_store import LocalJsonVectorStore, VectorPoint


def _index() -> SemanticIndex:
    metadata = SemanticModelMetadata("embedding-a", 2, "multilingual", "test", "index")
    chunk = SourceChunk("src/app.py", 3, 5, "def run():\n    return 1", "run")
    entry = SemanticIndexEntry("point-a", chunk, (1.0, 0.0), "chunk-hash", metadata)
    return SemanticIndex(metadata, (entry,))


def test_vector_points_carry_evidence_and_repository_metadata() -> None:
    points = vector_points_from_semantic_index(
        _index(),
        repo_id="repo-a",
        source_fingerprints={"src/app.py": source_fingerprint("source")},
        chunk_version="structured-v1",
    )

    validate_vector_points(points)
    assert points[0].point_id == "point-a"
    assert points[0].payload["path"] == "src/app.py"
    assert points[0].payload["startLine"] == 3
    assert points[0].payload["language"] == "python"
    assert points[0].payload["module"] == "src.app"


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


def test_publish_validates_payload_before_writing_local_store(tmp_path) -> None:
    store = LocalJsonVectorStore(tmp_path / "vectors.json")
    points = publish_semantic_index(
        _index(),
        store,
        repo_id="repo-a",
        source_fingerprints={"src/app.py": "source-hash"},
        chunk_version="structured-v1",
    )

    assert points[0].payload["visibility"] == "active"
    assert store.search((1.0, 0.0), limit=1)[0].point_id == "point-a"
