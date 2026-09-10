"""按文件复用持久化语义索引测试。"""

import json
from pathlib import Path

from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.ingestion.scanner import scan_repository
from codeinsight.retrieval import persistent_semantic


class _CountingEmbedder:
    model = "persistent-test-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts):
        values = tuple(texts)
        self.calls.append(values)
        vectors: list[tuple[float, float]] = []
        for text in values:
            code_point_total = 0
            for char in text:
                code_point_total += ord(char)
            vectors.append(
                (float(len(text)), float(code_point_total % 101))
            )
        return EmbeddingBatch(self.model, tuple(vectors), len(values))


def _build_cached(repository, embedder, *, cache_root=None, chunk_max_lines=80):
    return persistent_semantic.build_persistent_semantic_index(
        repository,
        scan_repository(repository),
        embedder.embed,
        model=embedder.model,
        chunk_max_lines=chunk_max_lines,
        cache_root=cache_root,
    )


def test_unchanged_repository_reuses_all_persisted_vectors(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "first.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    (repository / "second.py").write_text("def second():\n    return 2\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()

    first = _build_cached(repository, embedder, cache_root=cache)
    calls_after_first_build = len(embedder.calls)
    second = _build_cached(repository, embedder, cache_root=cache)

    assert calls_after_first_build == 1
    assert len(embedder.calls) == calls_after_first_build
    first_vectors = []
    second_vectors = []
    for entry in first.entries:
        first_vectors.append(entry.embedding)
    for entry in second.entries:
        second_vectors.append(entry.embedding)
    assert first_vectors == second_vectors
    assert first.metadata.index_id == second.metadata.index_id


def test_default_cache_permission_uses_user_temp_fallback(monkeypatch, tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "module.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
    default_root = tmp_path / "default-cache"
    fallback_root = tmp_path / "fallback-cache"
    embedder = _CountingEmbedder()
    writes: list[Path] = []
    original_write = persistent_semantic._write_manifest

    monkeypatch.delenv("CODEINSIGHT_SEMANTIC_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        persistent_semantic,
        "default_semantic_cache_root",
        lambda: default_root,
    )
    monkeypatch.setattr(
        persistent_semantic,
        "_fallback_semantic_cache_root",
        lambda: fallback_root,
    )

    def deny_default_once(path: Path, payload: dict) -> None:
        writes.append(path)
        if len(writes) == 1:
            raise PermissionError("default cache is not writable")
        original_write(path, payload)

    monkeypatch.setattr(persistent_semantic, "_write_manifest", deny_default_once)

    index = _build_cached(repository, embedder)

    assert index.entries
    assert writes[0].is_relative_to(default_root)
    assert writes[1].is_relative_to(fallback_root)


def test_only_changed_file_is_reembedded(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    first_path = repository / "first.py"
    second_path = repository / "second.py"
    first_path.write_text("def first():\n    return 1\n", encoding="utf-8")
    second_path.write_text("def second():\n    return 2\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()
    _build_cached(repository, embedder, cache_root=cache)
    embedder.calls.clear()

    second_path.write_text("def second():\n    return 3\n", encoding="utf-8")
    index = _build_cached(repository, embedder, cache_root=cache)

    assert len(embedder.calls) == 1
    assert "return 3" in embedder.calls[0][0]
    assert len(index.entries) == 2


def test_model_and_chunk_configuration_use_separate_cache_identity(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    module_lines = []
    for index in range(6):
        module_lines.append(f"line_{index} = {index}")
    (repository / "module.py").write_text(
        "\n".join(module_lines), encoding="utf-8"
    )
    cache = tmp_path / "cache"
    first = _CountingEmbedder()
    _build_cached(
        repository,
        first,
        chunk_max_lines=3,
        cache_root=cache,
    )

    second = _CountingEmbedder()
    second.model = "different-model"
    _build_cached(
        repository,
        second,
        chunk_max_lines=3,
        cache_root=cache,
    )
    third = _CountingEmbedder()
    _build_cached(
        repository,
        third,
        chunk_max_lines=2,
        cache_root=cache,
    )

    manifests = list(cache.rglob("*.json"))
    assert len(manifests) == 3
    assert second.calls
    assert third.calls
    chunk_sizes = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunk_sizes.add(payload["chunk_max_lines"])
    assert chunk_sizes == {
        2,
        3,
    }


def test_deleted_file_is_removed_from_persisted_index(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    keep = repository / "keep.py"
    remove = repository / "remove.py"
    keep.write_text("keep = True\n", encoding="utf-8")
    remove.write_text("remove = True\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()
    _build_cached(repository, embedder, cache_root=cache)
    embedder.calls.clear()

    remove.unlink()
    index = _build_cached(repository, embedder, cache_root=cache)

    assert embedder.calls == []
    indexed_paths = []
    for entry in index.entries:
        indexed_paths.append(entry.chunk.relative_path)
    assert indexed_paths == ["keep.py"]


def test_whitespace_only_file_is_not_sent_to_embedding(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "blank.py").write_text("\n\t\n", encoding="utf-8")
    (repository / "keep.py").write_text("keep = True\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()

    index = _build_cached(repository, embedder, cache_root=cache)

    assert len(embedder.calls) == 1
    assert all(text.strip() for text in embedder.calls[0])
    assert [entry.chunk.relative_path for entry in index.entries] == ["keep.py"]
