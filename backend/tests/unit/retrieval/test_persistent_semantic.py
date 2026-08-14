"""Tests for file-level persistent semantic index reuse."""

import json

from codeinsight.application.search_repository import build_repository_semantic_index
from codeinsight.domain.semantic import EmbeddingBatch


class _CountingEmbedder:
    model = "persistent-test-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts):
        values = tuple(texts)
        self.calls.append(values)
        vectors = tuple(
            (float(len(text)), float(sum(ord(char) for char in text) % 101)) for text in values
        )
        return EmbeddingBatch(self.model, vectors, len(values))


def test_unchanged_repository_reuses_all_persisted_vectors(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "first.py").write_text("def first():\n    return 1\n", encoding="utf-8")
    (repository / "second.py").write_text("def second():\n    return 2\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()

    first = build_repository_semantic_index(
        repository, semantic_embed=embedder.embed, cache_root=cache
    )
    calls_after_first_build = len(embedder.calls)
    second = build_repository_semantic_index(
        repository, semantic_embed=embedder.embed, cache_root=cache
    )

    assert calls_after_first_build == 1
    assert len(embedder.calls) == calls_after_first_build
    assert [entry.embedding for entry in first.entries] == [
        entry.embedding for entry in second.entries
    ]
    assert first.metadata.index_id == second.metadata.index_id


def test_only_changed_file_is_reembedded(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    first_path = repository / "first.py"
    second_path = repository / "second.py"
    first_path.write_text("def first():\n    return 1\n", encoding="utf-8")
    second_path.write_text("def second():\n    return 2\n", encoding="utf-8")
    cache = tmp_path / "cache"
    embedder = _CountingEmbedder()
    build_repository_semantic_index(repository, semantic_embed=embedder.embed, cache_root=cache)
    embedder.calls.clear()

    second_path.write_text("def second():\n    return 3\n", encoding="utf-8")
    index = build_repository_semantic_index(
        repository, semantic_embed=embedder.embed, cache_root=cache
    )

    assert len(embedder.calls) == 1
    assert "return 3" in embedder.calls[0][0]
    assert len(index.entries) == 2


def test_model_and_chunk_configuration_use_separate_cache_identity(tmp_path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "module.py").write_text(
        "\n".join(f"line_{index} = {index}" for index in range(6)), encoding="utf-8"
    )
    cache = tmp_path / "cache"
    first = _CountingEmbedder()
    build_repository_semantic_index(
        repository,
        semantic_embed=first.embed,
        chunk_max_lines=3,
        cache_root=cache,
    )

    second = _CountingEmbedder()
    second.model = "different-model"
    build_repository_semantic_index(
        repository,
        semantic_embed=second.embed,
        chunk_max_lines=3,
        cache_root=cache,
    )
    third = _CountingEmbedder()
    build_repository_semantic_index(
        repository,
        semantic_embed=third.embed,
        chunk_max_lines=2,
        cache_root=cache,
    )

    manifests = list(cache.rglob("*.json"))
    assert len(manifests) == 3
    assert second.calls
    assert third.calls
    chunk_sizes = {
        json.loads(path.read_text(encoding="utf-8"))["chunk_max_lines"] for path in manifests
    }
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
    build_repository_semantic_index(repository, semantic_embed=embedder.embed, cache_root=cache)
    embedder.calls.clear()

    remove.unlink()
    index = build_repository_semantic_index(
        repository, semantic_embed=embedder.embed, cache_root=cache
    )

    assert embedder.calls == []
    assert [entry.chunk.relative_path for entry in index.entries] == ["keep.py"]
