"""Integration tests for scanning a repository and chunking the result."""

from pathlib import Path

from codeinsight.ingestion.chunker import chunk_scan_result
from codeinsight.ingestion.scanner import scan_repository


def _logical_lines(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized:
        return []
    lines = normalized.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def test_chunk_scan_result_round_trips_scanned_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "node_modules").mkdir()
    (repo / "src" / "app.py").write_bytes(
        b"import os\r\n\r\nROOT = os.getcwd()\r\n\r\nprint(ROOT)\r\n"
    )
    (repo / "README.md").write_text("Title\n\nBody line.\n", encoding="utf-8")
    (repo / "node_modules" / "dep.py").write_text("hidden = True\n", encoding="utf-8")

    scan_result = scan_repository(repo)
    file_paths = []
    for file in scan_result.files:
        file_paths.append(file.relative_path)
    assert file_paths == ["README.md", "src/app.py"]

    skipped_node_module = False
    for skipped in scan_result.skipped:
        if skipped.relative_path == "node_modules/dep.py":
            skipped_node_module = True
            break
    assert not skipped_node_module

    chunks = chunk_scan_result(scan_result, max_lines=2)
    chunk_paths = []
    for chunk in chunks:
        chunk_paths.append(chunk.relative_path)
    assert chunk_paths == [
        "README.md",
        "README.md",
        "src/app.py",
        "src/app.py",
        "src/app.py",
    ]

    expected_lines = {}
    for file in scan_result.files:
        expected_lines[file.relative_path] = _logical_lines(file.text)
    for chunk in chunks:
        assert "\r" not in chunk.text
        assert chunk.start_line >= 1
        assert chunk.end_line >= chunk.start_line
        lines = expected_lines[chunk.relative_path]
        assert chunk.end_line <= len(lines)
        assert chunk.text == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])

    for path, lines in expected_lines.items():
        file_chunks = []
        for chunk in chunks:
            if chunk.relative_path == path:
                file_chunks.append(chunk)
        assert file_chunks[0].start_line == 1
        assert file_chunks[-1].end_line == len(lines)
        for previous, current in zip(file_chunks, file_chunks[1:]):
            assert previous.end_line + 1 == current.start_line
