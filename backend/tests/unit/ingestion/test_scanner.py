"""仓库文本读取测试。"""

from pathlib import Path

import pytest

from codeinsight.domain.errors import RepositoryScanError
from codeinsight.domain.source import ScanResult, SourceFile
from codeinsight.ingestion.scanner import REASON_READ_ERROR, scan_repository


def _write(repo: Path, relative: str, text: str) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_regular_text_files_are_read_in_sorted_order(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "b.txt", "b")
    _write(repo, "a.py", "print(1)")
    _write(repo, "sub/c.md", "# c")

    result = scan_repository(repo)

    assert isinstance(result, ScanResult)
    assert isinstance(result.files[0], SourceFile)
    relative_paths = []
    file_texts = []
    for item in result.files:
        relative_paths.append(item.relative_path)
        file_texts.append(item.text)
    assert relative_paths == ["a.py", "b.txt", "sub/c.md"]
    assert file_texts == ["print(1)", "b", "# c"]
    assert result.skipped == ()


def test_generated_and_dependency_directories_are_pruned(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    for relative in [
        ".git/config",
        ".venv/lib/site.py",
        "node_modules/pkg/index.js",
        "src/__pycache__/mod.pyc",
        "dist/app.js",
        "build/x.o",
    ]:
        _write(repo, relative, "ignored")
    _write(repo, "keep.py", "kept")

    result = scan_repository(repo)

    relative_paths = []
    for item in result.files:
        relative_paths.append(item.relative_path)
    assert relative_paths == ["keep.py"]
    assert result.skipped == ()


def test_unsupported_files_are_skipped(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "notes.rst", "not indexed")
    _write(repo, "main.py", "indexed")

    result = scan_repository(repo)

    file_paths = []
    for item in result.files:
        file_paths.append(item.relative_path)
    skipped_files = []
    for item in result.skipped:
        skipped_files.append((item.relative_path, item.reason))
    assert file_paths == ["main.py"]
    assert skipped_files == [
        ("notes.rst", "unsupported_extension")
    ]


def test_unreadable_text_isolated_as_a_file_skip(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _write(repo, "broken.txt", "ok")
    (repo / "broken.txt").write_bytes(b"\xff\xfe")
    _write(repo, "ok.txt", "fine")

    result = scan_repository(repo)

    file_paths = []
    for item in result.files:
        file_paths.append(item.relative_path)
    skipped_files = []
    for item in result.skipped:
        skipped_files.append((item.relative_path, item.reason))
    assert file_paths == ["ok.txt"]
    assert skipped_files == [
        ("broken.txt", REASON_READ_ERROR)
    ]


def test_root_errors_are_reported(tmp_path: Path) -> None:
    with pytest.raises(RepositoryScanError):
        scan_repository(tmp_path / "missing")
    plain_file = tmp_path / "plain.txt"
    plain_file.write_text("x", encoding="utf-8")
    with pytest.raises(RepositoryScanError):
        scan_repository(plain_file)
