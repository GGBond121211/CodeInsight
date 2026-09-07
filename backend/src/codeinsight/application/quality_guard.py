"""Patch 和 validation profile 的确定性安全检查。"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from codeinsight.domain.change import PatchArtifact
from codeinsight.domain.trace import (
    VERDICT_APPLIED,
    VERDICT_NOT_APPLIED,
    VERDICT_PARTIAL,
    ReconcileResult,
)
from codeinsight.ingestion.path_policy import safe_path, safe_relative_path

MAX_PATCH_BYTES = 512_000
MAX_TOUCHED_FILES = 10


@dataclass(frozen=True)
class ValidationProfile:
    name: str
    commands: tuple[tuple[str, ...], ...]
    timeout_seconds: float = 120.0

    @property
    def display_commands(self) -> tuple[str, ...]:
        return tuple(" ".join(command) for command in self.commands)


@dataclass(frozen=True)
class FileChange:
    """一个文件的目标状态；``None`` 表示删除，缺失文件加文本表示创建。"""

    path: str
    new_content: str | None


VALIDATION_PROFILES: dict[str, ValidationProfile] = {
    "python_compile": ValidationProfile(
        "python_compile", (("python", "-m", "compileall", "-q", "."),)
    ),
    "sample_repo_import_check": ValidationProfile(
        "sample_repo_import_check", (("python", "-m", "compileall", "-q", "."),)
    ),
    "pytest": ValidationProfile("pytest", (("python", "-m", "pytest", "-q"),)),
    "requests_mutation_pytest": ValidationProfile(
        "requests_mutation_pytest",
        (("python", "-m", "pytest", "-q", "codeinsight_mutation_test.py"),),
    ),
}


def get_validation_profile(name: str) -> ValidationProfile:
    try:
        return VALIDATION_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"未登记的 validation profile：{name}") from error


def build_patch(
    repository_root: str | Path,
    *,
    run_id: str,
    patch_id: str,
    relative_path: str,
    new_content: str,
) -> tuple[PatchArtifact, str, str]:
    artifact, changes, bases = build_change_set(
        repository_root,
        run_id=run_id,
        patch_id=patch_id,
        changes={relative_path: new_content},
    )
    relative = changes[0].path
    artifact = PatchArtifact(
        patch_id=artifact.patch_id,
        goal_id=artifact.goal_id,
        diff_text=artifact.diff_text,
        diff_hash=artifact.diff_hash,
        base_fingerprint=bases[relative],
        touched_files=artifact.touched_files,
    )
    old_content = safe_path(Path(repository_root).resolve(), relative).read_text(encoding="utf-8")
    return artifact, old_content, bases[relative]


def build_change_set(
    repository_root: str | Path,
    *,
    run_id: str,
    patch_id: str,
    changes: Mapping[str, str | None],
) -> tuple[PatchArtifact, tuple[FileChange, ...], dict[str, str]]:
    """构造多文件统一 diff，同时冻结每个文件的独立基线。"""
    if not changes:
        raise ValueError("补丁必须至少包含一个文件")
    if len(changes) > MAX_TOUCHED_FILES:
        raise ValueError("补丁涉及文件数超过上限")
    root = Path(repository_root).resolve()
    normalized: list[FileChange] = []
    bases: dict[str, str] = {}
    diffs: list[str] = []
    import difflib

    for raw_path, new_content in sorted(changes.items()):
        relative = safe_relative_path(raw_path)
        target = safe_path(root, relative)
        exists = target.is_file()
        if target.exists() and not exists:
            raise ValueError("补丁目标必须是普通文件")
        if new_content is None and not exists:
            raise FileNotFoundError("不能删除不存在的文件")
        if new_content is not None and len(new_content.encode("utf-8")) > MAX_PATCH_BYTES:
            raise ValueError("补丁内容超过大小上限")
        old_content = target.read_text(encoding="utf-8") if exists else None
        if old_content == new_content:
            raise ValueError("补丁没有实际变化")
        bases[relative] = fingerprint_text(old_content) if old_content is not None else "missing"
        normalized.append(FileChange(relative, new_content))
        diffs.append(
            "".join(
                difflib.unified_diff(
                    (old_content or "").splitlines(keepends=True),
                    (new_content or "").splitlines(keepends=True),
                    fromfile=f"a/{relative}" if exists else "/dev/null",
                    tofile=f"b/{relative}" if new_content is not None else "/dev/null",
                )
            )
        )
    diff = "".join(diffs)
    if not diff.strip() or len(diff.encode("utf-8")) > MAX_PATCH_BYTES:
        raise ValueError("diff 为空或超过大小上限")
    base_fingerprint = _fingerprint_map(bases)
    artifact = PatchArtifact(
        patch_id=patch_id,
        goal_id=run_id,
        diff_text=diff,
        diff_hash=hashlib.sha256(diff.encode("utf-8")).hexdigest(),
        base_fingerprint=base_fingerprint,
        touched_files=tuple(item.path for item in normalized),
    )
    return artifact, tuple(normalized), bases


def validate_patch_base(repository_root: str | Path, artifact: PatchArtifact) -> None:
    if len(artifact.touched_files) > MAX_TOUCHED_FILES:
        raise ValueError("补丁涉及文件数超过上限")
    root = Path(repository_root).resolve()
    for relative in artifact.touched_files:
        safe_path(root, relative)
    current = safe_path(root, artifact.touched_files[0]).read_text(encoding="utf-8")
    if fingerprint_text(current) != artifact.base_fingerprint:
        raise RuntimeError("源文件指纹已变化，补丁基线失效")


def validate_change_set_base(
    repository_root: str | Path, artifact: PatchArtifact, base_fingerprints: Mapping[str, str]
) -> None:
    if tuple(sorted(artifact.touched_files)) != tuple(sorted(base_fingerprints)):
        raise ValueError("补丁文件与基线集合不一致")
    root = Path(repository_root).resolve()
    observed: dict[str, str] = {}
    for relative in artifact.touched_files:
        target = safe_path(root, relative)
        observed[relative] = (
            fingerprint_text(target.read_text(encoding="utf-8")) if target.is_file() else "missing"
        )
    if (
        observed != dict(base_fingerprints)
        or _fingerprint_map(observed) != artifact.base_fingerprint
    ):
        raise RuntimeError("源文件指纹已变化，补丁基线失效")


def apply_file_content(
    workspace_root: str | Path, artifact: PatchArtifact, new_content: str
) -> None:
    root = Path(workspace_root).resolve()
    validate_patch_base(root, artifact)
    target = safe_path(root, artifact.touched_files[0])
    temporary = target.with_name(f".{target.name}.codeinsight.tmp")
    temporary.write_text(new_content, encoding="utf-8", newline="")
    os.replace(temporary, target)


def apply_change_set(
    workspace_root: str | Path,
    artifact: PatchArtifact,
    changes: tuple[FileChange, ...],
    base_fingerprints: Mapping[str, str],
) -> None:
    """先验证全部基线并准备临时文件，再提交整组变更。异常由上层 checkpoint 回滚。"""
    root = Path(workspace_root).resolve()
    validate_change_set_base(root, artifact, base_fingerprints)
    staged: dict[str, Path] = {}
    try:
        for change in changes:
            if change.new_content is None:
                continue
            target = safe_path(root, change.path)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{artifact.patch_id}.tmp")
            temporary.write_text(change.new_content, encoding="utf-8", newline="")
            staged[change.path] = temporary
        for change in changes:
            target = safe_path(root, change.path)
            if change.new_content is None:
                target.unlink()
            else:
                os.replace(staged[change.path], target)
    finally:
        for temporary in staged.values():
            if temporary.exists():
                temporary.unlink()


def reconcile_file(
    workspace_root: str | Path,
    artifact: PatchArtifact,
    new_content: str,
    *,
    run_id: str,
) -> ReconcileResult:
    target = safe_path(Path(workspace_root).resolve(), artifact.touched_files[0])
    observed = (
        fingerprint_text(target.read_text(encoding="utf-8")) if target.exists() else "missing"
    )
    expected = fingerprint_text(new_content)
    if observed == expected:
        verdict = VERDICT_APPLIED
        evidence = "目标文件指纹与批准的补丁内容一致"
    elif observed == artifact.base_fingerprint:
        verdict = VERDICT_NOT_APPLIED
        evidence = "目标文件仍保持补丁前基线指纹"
    else:
        verdict = VERDICT_PARTIAL
        evidence = "目标文件既不匹配补丁前也不匹配补丁后指纹"
    return ReconcileResult(run_id, "apply_patch", verdict, observed, evidence)


def reconcile_change_set(
    workspace_root: str | Path,
    artifact: PatchArtifact,
    changes: tuple[FileChange, ...],
    base_fingerprints: Mapping[str, str],
    *,
    run_id: str,
) -> ReconcileResult:
    root = Path(workspace_root).resolve()
    observed: dict[str, str] = {}
    expected: dict[str, str] = {}
    for change in changes:
        target = safe_path(root, change.path)
        observed[change.path] = (
            fingerprint_text(target.read_text(encoding="utf-8")) if target.is_file() else "missing"
        )
        expected[change.path] = (
            fingerprint_text(change.new_content) if change.new_content is not None else "missing"
        )
    if observed == expected:
        verdict, evidence = VERDICT_APPLIED, "全部目标文件与批准后的指纹一致"
    elif observed == dict(base_fingerprints):
        verdict, evidence = VERDICT_NOT_APPLIED, "全部目标文件仍保持补丁前基线"
    else:
        verdict, evidence = VERDICT_PARTIAL, "多文件补丁出现部分应用或未知状态"
    return ReconcileResult(run_id, "apply_patch", verdict, _fingerprint_map(observed), evidence)


def _fingerprint_map(values: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for path, fingerprint in sorted(values.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(fingerprint.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
