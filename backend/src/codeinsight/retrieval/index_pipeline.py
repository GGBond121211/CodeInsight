"""把源码语义索引转换为向量后端记录，并计算增量更新计划。"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from codeinsight.domain.semantic import SemanticIndex
from codeinsight.retrieval.index_manifest import IndexManifest
from codeinsight.retrieval.vector_store import VectorPoint

REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "repoId",
        "path",
        "language",
        "startLine",
        "endLine",
        "symbol",
        "module",
        "sourceFingerprint",
        "chunkVersion",
        "sourceHash",
        "visibility",
        "indexVersion",
        "embeddingModel",
        "vectorSchemaVersion",
    }
)


@dataclass(frozen=True)
class IncrementalUpdatePlan:
    upsert: tuple[VectorPoint, ...]
    delete: tuple[str, ...]
    unchanged_paths: tuple[str, ...]


def vector_points_from_semantic_index(
    index: SemanticIndex,
    *,
    repo_id: str,
    source_fingerprints: Mapping[str, str],
    chunk_version: str,
    visibility: str = "active",
) -> tuple[VectorPoint, ...]:
    """将语义索引的每个条目映射为可检索且可回到 Evidence 的向量点。"""
    points: list[VectorPoint] = []
    source_hash = source_hash_from_fingerprints(source_fingerprints)
    for entry in index.entries:
        chunk = entry.chunk
        source_fingerprint = source_fingerprints.get(chunk.relative_path)
        if source_fingerprint is None:
            raise ValueError(f"缺少源码指纹：{chunk.relative_path}")
        payload = {
            "repoId": repo_id,
            "path": chunk.relative_path,
            "language": _language(chunk.relative_path),
            "startLine": chunk.start_line,
            "endLine": chunk.end_line,
            "symbol": chunk.symbol_path or "",
            "module": _module(chunk.relative_path),
            "sourceFingerprint": source_fingerprint,
            "chunkVersion": chunk_version,
            "sourceHash": source_hash,
            "visibility": visibility,
            "indexVersion": index.metadata.index_id,
            "embeddingModel": index.metadata.model,
            "vectorSchemaVersion": index.metadata.vector_schema_version,
        }
        points.append(
            VectorPoint(
                entry.chunk_id,
                entry.embedding,
                payload,
                sparse_vector=entry.sparse_embedding,
            )
        )
    return tuple(points)


def build_index_manifest(
    *,
    index_version: str,
    repo_id: str,
    backend: str,
    collection: str,
    index: SemanticIndex,
    chunk_version: str,
    source_hash: str,
) -> IndexManifest:
    return IndexManifest.create(
        index_version=index_version,
        repo_id=repo_id,
        backend=backend,
        collection=collection,
        embedding_model=index.metadata.model,
        dimension=index.metadata.dimensions,
        distance="cosine",
        chunk_version=chunk_version,
        source_hash=source_hash,
        vector_schema_version=index.metadata.vector_schema_version,
        sparse_model=index.metadata.sparse_model,
    )


def plan_incremental_update(
    existing: Iterable[VectorPoint],
    current: Iterable[VectorPoint],
) -> IncrementalUpdatePlan:
    """按 path/sourceFingerprint 计算删除、upsert 和未变化文件。"""
    old_points = tuple(existing)
    new_points = tuple(current)
    old_by_path: dict[str, list[VectorPoint]] = {}
    new_by_path: dict[str, list[VectorPoint]] = {}
    for point in old_points:
        old_by_path.setdefault(str(point.payload.get("path", "")), []).append(point)
    for point in new_points:
        new_by_path.setdefault(str(point.payload.get("path", "")), []).append(point)

    delete: list[str] = []
    upsert: list[VectorPoint] = []
    unchanged: list[str] = []
    for path in sorted(set(old_by_path) | set(new_by_path)):
        old = old_by_path.get(path, [])
        new = new_by_path.get(path, [])
        old_fingerprint = {item.payload.get("sourceFingerprint") for item in old}
        new_fingerprint = {item.payload.get("sourceFingerprint") for item in new}
        if old_fingerprint == new_fingerprint and old and new:
            unchanged.append(path)
            continue
        delete.extend(item.point_id for item in old)
        upsert.extend(new)
    return IncrementalUpdatePlan(tuple(upsert), tuple(sorted(delete)), tuple(unchanged))


def validate_vector_points(
    points: Sequence[VectorPoint], *, require_sparse: bool = False
) -> None:
    """发布前检查 payload 字段和 Evidence 行号，不改变后端状态。"""
    for point in points:
        missing = REQUIRED_PAYLOAD_FIELDS - set(point.payload)
        if missing:
            raise ValueError(f"向量点缺少 payload 字段：{sorted(missing)}")
        if (
            not isinstance(point.payload["startLine"], int)
            or not isinstance(point.payload["endLine"], int)
            or point.payload["startLine"] < 1
            or point.payload["endLine"] < point.payload["startLine"]
        ):
            raise ValueError("向量点行号必须是合法的 1-based 闭区间")
        if require_sparse and point.sparse_vector is None:
            raise ValueError("Dense/Sparse 向量点缺少 Provider Sparse 向量")


def publish_semantic_index(
    index: SemanticIndex,
    store,
    *,
    repo_id: str,
    source_fingerprints: Mapping[str, str],
    chunk_version: str,
    require_sparse: bool = False,
) -> tuple[VectorPoint, ...]:
    """校验并写入任意 VectorStore；只有 payload 合法后才触碰后端。"""
    points = vector_points_from_semantic_index(
        index,
        repo_id=repo_id,
        source_fingerprints=source_fingerprints,
        chunk_version=chunk_version,
    )
    validate_vector_points(points, require_sparse=require_sparse)
    store.upsert(points)
    if not store.health():
        raise RuntimeError("向量后端写入后健康检查失败")
    return points


def source_fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def source_hash_from_fingerprints(source_fingerprints: Mapping[str, str]) -> str:
    material = "\n".join(
        f"{path}\0{fingerprint}" for path, fingerprint in sorted(source_fingerprints.items())
    )
    return source_fingerprint(material)


def _language(relative_path: str) -> str:
    suffix = PurePosixPath(relative_path).suffix.lower()
    return {".py": "python", ".md": "markdown", ".toml": "toml", ".json": "json"}.get(
        suffix,
        suffix.lstrip(".") or "unknown",
    )


def _module(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    return ".".join(path.with_suffix("").parts)
