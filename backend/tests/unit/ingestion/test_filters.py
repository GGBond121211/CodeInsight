"""仓库文件分类测试。"""

from codeinsight.ingestion.filters import (
    REASON_UNSUPPORTED_EXTENSION,
    is_ignored_directory,
    skip_reason,
)


def test_generated_and_dependency_directories_are_ignored() -> None:
    for name in [
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
    ]:
        assert is_ignored_directory(name)
    assert not is_ignored_directory("src")


def test_supported_text_extensions_are_read() -> None:
    for suffix in [".py", ".pyi", ".md", ".txt", ".toml", ".yaml", ".yml", ".json"]:
        assert skip_reason(f"pkg/file{suffix}") is None


def test_unsupported_extensions_have_a_reason() -> None:
    for path in ["notes.rst", "data.csv", "Makefile", ".gitignore"]:
        assert skip_reason(path) == REASON_UNSUPPORTED_EXTENSION
