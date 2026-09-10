"""用于读取仓库的小型文件分类辅助函数。"""

from pathlib import Path

IGNORED_DIRECTORIES: frozenset[str] = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".codeinsight",
        "dist",
        "build",
    }
)

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".pyi", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
)

REASON_UNSUPPORTED_EXTENSION = "unsupported_extension"


def is_ignored_directory(name: str) -> bool:
    """判断生成目录或依赖目录是否应该跳过。"""
    return name in IGNORED_DIRECTORIES


def skip_reason(relative_path: str) -> str | None:
    """当文件类型不在读取范围内时返回原因。"""
    if Path(relative_path).suffix.casefold() not in TEXT_EXTENSIONS:
        return REASON_UNSUPPORTED_EXTENSION
    return None
