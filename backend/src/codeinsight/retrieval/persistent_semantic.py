"""Local persistent semantic indexes with file-level incremental rebuilds."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from codeinsight.domain.semantic import EmbeddingBatch, SemanticIndex
from codeinsight.domain.source import ScanResult, SourceChunk
from codeinsight.ingestion.chunker import chunk_source_file
from codeinsight.retrieval.semantic import build_semantic_index

EmbeddingFunction = Callable[[Sequence[str]], EmbeddingBatch]
CACHE_SCHEMA_VERSION = 1
CHUNKING_VERSION = "fixed-lines-v1"


def default_semantic_cache_root() -> Path:
    """Return an OS-local cache directory outside analyzed repositories."""
    configured = os.environ.get("CODEINSIGHT_SEMANTIC_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        return Path(os.environ["LOCALAPPDATA"]) / "CodeInsight" / "semantic-indexes"
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "codeinsight" / "semantic-indexes"


def embedding_model_id(embed: EmbeddingFunction) -> str | None:
    """Read a stable public model ID from a bound embedding adapter method."""
    owner = getattr(embed, "__self__", None)
    model = getattr(owner, "model", None) or getattr(embed, "model", None)
    return model if isinstance(model, str) and model.strip() else None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _cache_path(
    root: Path,
    *,
    model: str,
    chunk_max_lines: int,
    cache_root: Path,
) -> Path:
    repository_id = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:20]
    configuration = json.dumps(
        {
            "model": model,
            "chunk_max_lines": chunk_max_lines,
            "chunking_version": CHUNKING_VERSION,
            "schema_version": CACHE_SCHEMA_VERSION,
        },
        sort_keys=True,
    )
    configuration_id = hashlib.sha256(configuration.encode("utf-8")).hexdigest()[:20]
    return cache_root / repository_id / f"{configuration_id}.json"


def _chunk_payload(chunk: SourceChunk, vector: Sequence[float]) -> dict[str, Any]:
    return {
        "relative_path": chunk.relative_path,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "symbol_path": chunk.symbol_path,
        "embedding": list(vector),
    }


def _chunk_from_payload(payload: dict[str, Any]) -> SourceChunk:
    return SourceChunk(
        relative_path=payload["relative_path"],
        start_line=payload["start_line"],
        end_line=payload["end_line"],
        text=payload["text"],
        symbol_path=payload.get("symbol_path"),
    )


def _read_manifest(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or payload.get("schema_version") != CACHE_SCHEMA_VERSION:
        return None
    return payload


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _embed_chunks(
    chunks: Sequence[SourceChunk], embed: EmbeddingFunction
) -> tuple[str, int, tuple[tuple[float, ...], ...]]:
    batch = embed(tuple(chunk.text for chunk in chunks))
    if len(batch.vectors) != len(chunks):
        raise ValueError("embedding provider returned an unexpected vector count")
    return batch.model, batch.dimensions, batch.vectors


def build_persistent_semantic_index(
    root: str | Path,
    scan_result: ScanResult,
    embed: EmbeddingFunction,
    *,
    model: str,
    chunk_max_lines: int,
    cache_root: str | Path | None = None,
) -> SemanticIndex:
    """Load unchanged vectors and embed only files whose content changed."""
    root_path = Path(root).resolve()
    storage_root = Path(cache_root) if cache_root is not None else default_semantic_cache_root()
    path = _cache_path(
        root_path,
        model=model,
        chunk_max_lines=chunk_max_lines,
        cache_root=storage_root,
    )
    manifest = _read_manifest(path) or {}
    cached_files = manifest.get("files", {}) if manifest.get("model") == model else {}
    files_payload: dict[str, Any] = {}
    dimensions = manifest.get("dimensions") if cached_files else None
    file_chunks: dict[str, tuple[SourceChunk, ...]] = {}
    file_vectors: dict[str, tuple[tuple[float, ...], ...]] = {}
    changed_paths: list[str] = []
    changed_chunks: list[SourceChunk] = []

    for source in scan_result.files:
        source_fingerprint = _sha256(source.text)
        cached = cached_files.get(source.relative_path)
        if cached and cached.get("source_fingerprint") == source_fingerprint:
            chunks = tuple(_chunk_from_payload(item) for item in cached.get("chunks", ()))
            vectors = tuple(
                tuple(float(value) for value in item["embedding"])
                for item in cached.get("chunks", ())
            )
        else:
            chunks = chunk_source_file(source, max_lines=chunk_max_lines)
            vectors = ()
            if chunks:
                changed_paths.append(source.relative_path)
                changed_chunks.extend(chunks)
        file_chunks[source.relative_path] = chunks
        file_vectors[source.relative_path] = vectors
        files_payload[source.relative_path] = {
            "source_fingerprint": source_fingerprint,
            "chunks": [],
        }

    if changed_chunks:
        returned_model, returned_dimensions, changed_vectors = _embed_chunks(changed_chunks, embed)
        if returned_model != model:
            raise ValueError("embedding provider model changed during index build")
        if dimensions is not None and returned_dimensions != dimensions:
            raise ValueError("embedding dimensions changed during incremental index build")
        dimensions = returned_dimensions
        offset = 0
        for relative_path in changed_paths:
            count = len(file_chunks[relative_path])
            file_vectors[relative_path] = changed_vectors[offset : offset + count]
            offset += count

    ordered_chunks: list[SourceChunk] = []
    ordered_vectors: list[tuple[float, ...]] = []
    for source in scan_result.files:
        chunks = file_chunks[source.relative_path]
        vectors = file_vectors[source.relative_path]
        files_payload[source.relative_path]["chunks"] = [
            _chunk_payload(chunk, vector) for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        ordered_chunks.extend(chunks)
        ordered_vectors.extend(vectors)

    if not ordered_chunks:
        raise ValueError("semantic index requires at least one source chunk")
    manifest_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "repository_root": str(root_path),
        "model": model,
        "dimensions": dimensions,
        "chunk_max_lines": chunk_max_lines,
        "chunking_version": CHUNKING_VERSION,
        "files": files_payload,
    }
    _write_manifest(path, manifest_payload)
    vectors = tuple(ordered_vectors)
    return build_semantic_index(
        tuple(ordered_chunks),
        lambda _texts: EmbeddingBatch(model, vectors, 0),
    )
