"""索引身份与 staging/publish/rollback 生命周期。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class IndexManifest:
    """描述一份可被检索后端安全复用的索引。"""

    index_version: str
    repo_id: str
    backend: str
    collection: str
    embedding_model: str
    dimension: int
    distance: str
    chunk_version: str
    source_hash: str
    updated_at: str

    @classmethod
    def create(
        cls,
        *,
        index_version: str,
        repo_id: str,
        backend: str,
        collection: str,
        embedding_model: str,
        dimension: int,
        distance: str,
        chunk_version: str,
        source_hash: str,
    ) -> IndexManifest:
        return cls(
            index_version=index_version,
            repo_id=repo_id,
            backend=backend,
            collection=collection,
            embedding_model=embedding_model,
            dimension=dimension,
            distance=distance,
            chunk_version=chunk_version,
            source_hash=source_hash,
            updated_at=datetime.now(UTC).isoformat(),
        )

    def __post_init__(self) -> None:
        for name in (
            "index_version",
            "repo_id",
            "backend",
            "collection",
            "embedding_model",
            "distance",
            "chunk_version",
            "source_hash",
            "updated_at",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} 不能为空")
        if self.dimension <= 0:
            raise ValueError("dimension 必须是正整数")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> IndexManifest:
        return cls(**payload)


class IndexPublisher:
    """在索引根目录外用原子 JSON 指针发布和回滚 manifest。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.staging = self.root / "staging"
        self.manifests = self.root / "manifests"
        self.pointer_path = self.root / "active.json"

    def stage(self, manifest: IndexManifest) -> Path:
        self.staging.mkdir(parents=True, exist_ok=True)
        path = self.staging / f"{manifest.index_version}.json"
        _atomic_write(path, manifest.to_dict())
        return path

    def validate_staged(self, index_version: str) -> IndexManifest:
        path = self.staging / f"{index_version}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"staging manifest 无法读取：{index_version}") from error
        return IndexManifest.from_dict(payload)

    def publish(self, index_version: str) -> IndexManifest:
        manifest = self.validate_staged(index_version)
        self.manifests.mkdir(parents=True, exist_ok=True)
        target = self.manifests / f"{index_version}.json"
        _atomic_write(target, manifest.to_dict())
        current = self._pointer()
        _atomic_write(
            self.pointer_path,
            {"active": index_version, "previous": current.get("active")},
        )
        return manifest

    def active(self) -> IndexManifest | None:
        version = self._pointer().get("active")
        if not isinstance(version, str):
            return None
        return IndexManifest.from_dict(
            json.loads((self.manifests / f"{version}.json").read_text(encoding="utf-8"))
        )

    def rollback(self) -> IndexManifest:
        pointer = self._pointer()
        previous = pointer.get("previous")
        if not isinstance(previous, str):
            raise ValueError("没有可回滚的上一份 active index")
        current = pointer.get("active")
        _atomic_write(self.pointer_path, {"active": previous, "previous": current})
        active = self.active()
        if active is None:
            raise ValueError("回滚后的 active manifest 不存在")
        return active

    def _pointer(self) -> dict[str, object]:
        try:
            payload = json.loads(self.pointer_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f"{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        Path(temporary_name).replace(path)
    finally:
        temporary_path = Path(temporary_name)
        if temporary_path.exists():
            temporary_path.unlink()
