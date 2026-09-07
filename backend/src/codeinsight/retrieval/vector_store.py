"""向量后端的最小契约和显式选择的本地 exact 实现。"""

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


class LocalJsonVectorStore:
    """使用 JSON 文件保存向量并执行精确 cosine 搜索的独立后端。"""

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
