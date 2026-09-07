"""读取、工具和隔离工作区共用的路径边界。"""

from __future__ import annotations

from pathlib import Path

VCS_DIRECTORIES = frozenset({".git", ".hg", ".svn"})
SENSITIVE_BASENAMES = frozenset(
    {".env", ".env.local", ".env.production", "id_rsa", "id_dsa"}
)
SENSITIVE_SUFFIXES = (".pem", ".key", ".p12", ".pfx")


def is_unsafe_directory(path: Path) -> bool:
    """在目录遍历前排除符号链接和 Windows junction。"""
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in SENSITIVE_BASENAMES
        or name.startswith(".env.")
        or name.endswith(SENSITIVE_SUFFIXES)
    )


def safe_relative_path(value: str) -> str:
    if ":" in value:
        raise PermissionError("不允许驱动器相对路径或 Windows 数据流路径")
    candidate = Path(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PermissionError("路径必须位于仓库根目录内")
    if not candidate.parts or candidate == Path("."):
        raise ValueError("路径不能为空")
    if any(part.lower() in VCS_DIRECTORIES for part in candidate.parts):
        raise PermissionError("版本控制目录不属于允许范围")
    if is_sensitive(candidate):
        raise PermissionError("敏感文件不允许读写")
    return candidate.as_posix()


def safe_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    target = root / safe_relative_path(relative)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise PermissionError("路径越界") from error
    # 检查每一级父目录，包含 is_symlink() 无法识别的 junction。
    current = target
    while current != root:
        if is_unsafe_directory(current):
            raise PermissionError("不允许通过符号链接或 junction 访问资源")
        current = current.parent
    return target
