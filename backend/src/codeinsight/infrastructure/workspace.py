"""隔离工作区的创建、检查点和恢复。

Step 6 的硬边界是：源仓库只读，所有写入都发生在受管控的临时目录。
复制时不跟随符号链接，也不复制密钥文件，避免把源仓库外的内容带入
Sandbox。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from codeinsight.domain.change import WorkspaceCheckpoint, WorkspaceRun
from codeinsight.ingestion.path_policy import (
    VCS_DIRECTORIES,
    is_sensitive,
    is_unsafe_directory,
)


@dataclass(frozen=True)
class ManagedWorkspace:
    run: WorkspaceRun
    source_fingerprint: str


class WorkspaceManager:
    """管理每个 Run 独立的临时 workspace。"""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir or (Path(tempfile.gettempdir()) / "codeinsight-workspaces"))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._workspaces: dict[str, ManagedWorkspace] = {}
        self._checkpoints: dict[str, Path] = {}
        self._state_dir = self.base_dir / ".state"
        self._state_dir.mkdir(exist_ok=True)
        self._load_state()

    def create(self, source_repo: str | Path, run_id: str) -> ManagedWorkspace:
        existing = self._workspaces.get(run_id)
        if existing is not None:
            if str(Path(source_repo).resolve()) != existing.run.source_repo_path:
                raise ValueError("同一 Run 不能绑定不同的 source_repo")
            return existing
        source = Path(source_repo).resolve()
        if not source.is_dir():
            raise ValueError("source_repo 必须是存在的目录")
        workspace_id = f"ws-{uuid4().hex[:16]}"
        destination = (self.base_dir / workspace_id).resolve()
        if destination.is_relative_to(source):
            raise ValueError("隔离工作区必须位于源仓库之外")
        destination.mkdir(parents=True)
        self._copy_tree(source, destination)
        managed = ManagedWorkspace(
            WorkspaceRun(
                workspace_id=workspace_id,
                run_id=run_id,
                source_repo_path=str(source),
                workspace_path=str(destination),
            ),
            fingerprint_tree(source),
        )
        self._workspaces[run_id] = managed
        self._persist(managed)
        return managed

    def get(self, run_id: str) -> ManagedWorkspace | None:
        return self._workspaces.get(run_id)

    def create_checkpoint(self, run_id: str) -> WorkspaceCheckpoint:
        managed = self._require(run_id)
        checkpoint_id = f"cp-{uuid4().hex[:16]}"
        destination = (self.base_dir / checkpoint_id).resolve()
        destination.mkdir(parents=True)
        self._copy_tree(Path(managed.run.workspace_path), destination)
        checkpoint = WorkspaceCheckpoint(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            workspace_path=managed.run.workspace_path,
            tree_fingerprint=fingerprint_tree(destination),
            created_at_epoch_ms=_now_ms(),
        )
        self._checkpoints[checkpoint_id] = destination
        updated = WorkspaceRun(
            workspace_id=managed.run.workspace_id,
            run_id=managed.run.run_id,
            source_repo_path=managed.run.source_repo_path,
            workspace_path=managed.run.workspace_path,
            checkpoints=managed.run.checkpoints + (checkpoint,),
        )
        self._workspaces[run_id] = ManagedWorkspace(updated, managed.source_fingerprint)
        self._persist(self._workspaces[run_id])
        return checkpoint

    def rollback(self, run_id: str, checkpoint_id: str) -> None:
        managed = self._require(run_id)
        checkpoint = self._checkpoints.get(checkpoint_id)
        if checkpoint is None:
            raise FileNotFoundError("checkpoint 不存在")
        if not self._inside(checkpoint, self.base_dir):
            raise PermissionError("checkpoint 不在受管控目录内")
        if not any(item.checkpoint_id == checkpoint_id for item in managed.run.checkpoints):
            raise PermissionError("checkpoint 不属于当前 Run")
        workspace = Path(managed.run.workspace_path)
        self._clear_directory(workspace)
        self._copy_tree(checkpoint, workspace)
        if fingerprint_tree(workspace) != fingerprint_tree(checkpoint):
            raise RuntimeError("checkpoint 回滚后指纹不一致")

    def _require(self, run_id: str) -> ManagedWorkspace:
        managed = self.get(run_id)
        if managed is None:
            raise FileNotFoundError("隔离 workspace 不存在")
        return managed

    def _persist(self, managed: ManagedWorkspace) -> None:
        payload = {
            "workspace_id": managed.run.workspace_id,
            "run_id": managed.run.run_id,
            "source_repo_path": managed.run.source_repo_path,
            "workspace_path": managed.run.workspace_path,
            "source_fingerprint": managed.source_fingerprint,
            "checkpoints": [
                {
                    "checkpoint_id": item.checkpoint_id,
                    "run_id": item.run_id,
                    "workspace_path": item.workspace_path,
                    "tree_fingerprint": item.tree_fingerprint,
                    "created_at_epoch_ms": item.created_at_epoch_ms,
                    "storage_path": str(self._checkpoints[item.checkpoint_id]),
                }
                for item in managed.run.checkpoints
            ],
        }
        target = self._state_path(managed.run.run_id)
        # Keep the temporary name independent of the long target hash:
        # pytest can place this path near Windows' MAX_PATH boundary when the
        # repository path is nested.
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        for _ in range(32):
            temporary = target.with_name(f".{uuid4().hex[:16]}.tmp")
            try:
                with temporary.open("x", encoding="utf-8") as stream:
                    stream.write(serialized)
            except FileExistsError:
                continue
            try:
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            return
        raise FileExistsError("无法为 workspace 状态分配唯一临时文件")

    def _load_state(self) -> None:
        for state_file in sorted(self._state_dir.glob("*.json")):
            try:
                payload = json.loads(state_file.read_text(encoding="utf-8"))
                workspace = Path(payload["workspace_path"]).resolve()
                if not self._inside(workspace, self.base_dir) or not workspace.is_dir():
                    continue
                checkpoints: list[WorkspaceCheckpoint] = []
                for raw in payload.get("checkpoints", []):
                    storage = Path(raw["storage_path"]).resolve()
                    if not self._inside(storage, self.base_dir) or not storage.is_dir():
                        continue
                    checkpoint = WorkspaceCheckpoint(
                        raw["checkpoint_id"],
                        raw["run_id"],
                        raw["workspace_path"],
                        raw["tree_fingerprint"],
                        int(raw["created_at_epoch_ms"]),
                    )
                    checkpoints.append(checkpoint)
                    self._checkpoints[checkpoint.checkpoint_id] = storage
                run = WorkspaceRun(
                    payload["workspace_id"],
                    payload["run_id"],
                    payload["source_repo_path"],
                    str(workspace),
                    tuple(checkpoints),
                )
                self._workspaces[run.run_id] = ManagedWorkspace(run, payload["source_fingerprint"])
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

    def _state_path(self, run_id: str) -> Path:
        name = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        return self._state_dir / f"{name}.json"

    @staticmethod
    def _inside(path: Path, parent: Path) -> bool:
        try:
            path.resolve().relative_to(parent.resolve())
        except ValueError:
            return False
        return True

    @classmethod
    def _copy_tree(cls, source: Path, destination: Path) -> None:
        for current, directories, files in os.walk(source, topdown=True, followlinks=False):
            current_path = Path(current)
            directories[:] = [
                name for name in directories
                if name.lower() not in VCS_DIRECTORIES
                and not is_unsafe_directory(Path(current) / name)
            ]
            relative = current_path.relative_to(source)
            target_dir = destination / relative
            target_dir.mkdir(parents=True, exist_ok=True)
            for name in files:
                origin = current_path / name
                if origin.is_symlink():
                    raise PermissionError("workspace 复制拒绝符号链接文件")
                if is_sensitive(origin):
                    continue
                target = target_dir / name
                shutil.copy2(origin, target)

    @staticmethod
    def _clear_directory(directory: Path) -> None:
        for item in directory.iterdir():
            if getattr(item, "is_junction", lambda: False)():
                item.rmdir()
            elif item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()


def fingerprint_tree(root: str | Path) -> str:
    path = Path(root).resolve()
    digest = hashlib.sha256()
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        directories[:] = sorted(
            name for name in directories
            if name.lower() not in VCS_DIRECTORIES
            and not is_unsafe_directory(Path(current) / name)
        )
        current_path = Path(current)
        for name in sorted(files):
            candidate = current_path / name
            if candidate.is_symlink() or is_sensitive(candidate):
                continue
            relative = candidate.relative_to(path).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(candidate.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_artifact(root: str | Path) -> str:
    """检测验证期间的产物改动，包含新增敏感文件；忽略检查生成的缓存。"""
    path = Path(root).resolve()
    digest = hashlib.sha256()
    for current, directories, files in os.walk(path, topdown=True, followlinks=False):
        directories[:] = sorted(
            name for name in directories if name not in {"__pycache__", ".pytest_cache"}
        )
        for name in directories:
            if is_unsafe_directory(Path(current) / name):
                raise PermissionError("验证产物包含符号链接或 junction")
        for name in sorted(files):
            candidate = Path(current) / name
            if candidate.is_symlink():
                raise PermissionError("验证产物包含符号链接")
            digest.update(candidate.relative_to(path).as_posix().encode())
            digest.update(b"\0")
            digest.update(candidate.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
