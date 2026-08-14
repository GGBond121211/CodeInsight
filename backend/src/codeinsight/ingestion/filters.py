"""Small file classification helpers for repository reading."""

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
        "dist",
        "build",
    }
)

TEXT_EXTENSIONS: frozenset[str] = frozenset(
    {".py", ".pyi", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"}
)

REASON_UNSUPPORTED_EXTENSION = "unsupported_extension"


def is_ignored_directory(name: str) -> bool:
    """Return whether a generated or dependency directory should be skipped."""
    return name in IGNORED_DIRECTORIES


def skip_reason(relative_path: str) -> str | None:
    """Return a reason when the file type is outside the reading scope."""
    if Path(relative_path).suffix.casefold() not in TEXT_EXTENSIONS:
        return REASON_UNSUPPORTED_EXTENSION
    return None
