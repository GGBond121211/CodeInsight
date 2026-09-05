"""向量后端的最小契约，以及可回退的本地 exact 实现。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class VectorPoint:
    """一条向量及其可回到源码 Evidence 的 payload。"""

    point_id: str
    vector: tuple[float, ...]
    payload: Mapping[str, object]


@dataclass(frozen=True)
class VectorSearchHit:
    point_id: str
    score: float
    payload: Mapping[str, object]


class VectorStore(Protocol):
    """Step 4 只需要的向量后端操作。"""

    def upsert(self, points: Iterable[VectorPoint]) -> None: ...

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]: ...

    def delete(self, point_ids: Iterable[str]) -> None: ...

    def health(self) -> bool: ...


class FallbackVectorStore:
    """显式的主后端/回退后端选择器，并保留本次选择原因。

    只有调用方明确组合两个后端时才启用回退；普通 ``VectorStore`` 不会
    静默改变后端。写入、查询和删除都遵循同一选择结果，便于上层把
    ``last_backend`` 与 ``last_reason`` 写入 Trace。
    """

    def __init__(
        self,
        primary: VectorStore,
        fallback: VectorStore,
        *,
        primary_name: str = "qdrant_hnsw",
        fallback_name: str = "local_json",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.primary_name = primary_name
        self.fallback_name = fallback_name
        self.last_backend = primary_name
        self.last_reason: str | None = None

    def _use_fallback(self, reason: str) -> VectorStore:
        self.last_backend = self.fallback_name
        self.last_reason = reason
        return self.fallback

    def _use_primary(self) -> VectorStore:
        self.last_backend = self.primary_name
        self.last_reason = None
        return self.primary

    def _selected(self) -> VectorStore:
        try:
            if self.primary.health():
                return self._use_primary()
            return self._use_fallback("primary health check returned false")
        except Exception as error:
            return self._use_fallback(f"primary health check failed: {type(error).__name__}")

    def upsert(self, points: Iterable[VectorPoint]) -> None:
        values = tuple(points)
        store = self._selected()
        try:
            store.upsert(values)
        except Exception as error:
            if store is self.fallback:
                raise
            self._use_fallback(f"primary upsert failed: {type(error).__name__}").upsert(values)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        store = self._selected()
        try:
            return store.search(vector, limit=limit, query_filter=query_filter)
        except Exception as error:
            if store is self.fallback:
                raise
            return self._use_fallback(f"primary search failed: {type(error).__name__}").search(
                vector,
                limit=limit,
                query_filter=query_filter,
            )

    def delete(self, point_ids: Iterable[str]) -> None:
        ids = tuple(point_ids)
        store = self._selected()
        try:
            store.delete(ids)
        except Exception as error:
            if store is self.fallback:
                raise
            self._use_fallback(f"primary delete failed: {type(error).__name__}").delete(ids)

    def health(self) -> bool:
        store = self._selected()
        try:
            return store.health()
        except Exception:
            if store is self.fallback:
                return False
            return self._use_fallback("primary health check failed during health").health()


class LocalJsonVectorStore:
    """使用 JSON 文件保存向量并执行精确 cosine 搜索的回退后端。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._points: dict[str, VectorPoint] = {}
        self._load()

    def upsert(self, points: Iterable[VectorPoint]) -> None:
        for point in points:
            self._validate_point(point)
            self._points[point.point_id] = point
        self._persist()

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 5,
        query_filter: Mapping[str, object] | None = None,
    ) -> tuple[VectorSearchHit, ...]:
        if limit <= 0:
            raise ValueError("limit 必须是正整数")
        query = tuple(float(value) for value in vector)
        if not query:
            raise ValueError("查询向量不能为空")
        scored: list[tuple[float, VectorPoint]] = []
        for point in self._points.values():
            if _matches_filter(point.payload, query_filter):
                scored.append((_cosine(query, point.vector), point))
        scored.sort(key=lambda item: (-item[0], item[1].point_id))
        return tuple(
            VectorSearchHit(point.point_id, score, point.payload)
            for score, point in scored[:limit]
        )

    def delete(self, point_ids: Iterable[str]) -> None:
        changed = False
        for point_id in point_ids:
            if point_id in self._points:
                del self._points[point_id]
                changed = True
        if changed:
            self._persist()

    def health(self) -> bool:
        return True

    def _load(self) -> None:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        for item in document.get("points", ()):
            point = VectorPoint(
                point_id=str(item["point_id"]),
                vector=tuple(float(value) for value in item["vector"]),
                payload=item.get("payload", {}),
            )
            self._validate_point(point)
            self._points[point.point_id] = point

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": 1,
            "points": [
                {
                    "point_id": point.point_id,
                    "vector": list(point.vector),
                    "payload": dict(point.payload),
                }
                for point in self._points.values()
            ],
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(document, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
            Path(temporary_name).replace(self.path)
        finally:
            temporary_path = Path(temporary_name)
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _validate_point(point: VectorPoint) -> None:
        if not point.point_id.strip() or not point.vector:
            raise ValueError("向量点必须包含非空 ID 和向量")
        if any(not math.isfinite(value) for value in point.vector):
            raise ValueError("向量必须包含有限数")


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second):
        raise ValueError("查询向量与索引向量的维度不匹配")
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    return sum(left * right for left, right in zip(first, second, strict=True)) / (
        first_norm * second_norm
    )


def _matches_filter(
    payload: Mapping[str, object], query_filter: Mapping[str, object] | None
) -> bool:
    if not query_filter:
        return True
    for key, expected in query_filter.items():
        if payload.get(key) != expected:
            return False
    return True
