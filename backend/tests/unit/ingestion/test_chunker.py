"""Unit tests for fixed-size source file chunking."""

import pytest

from codeinsight.domain.source import ScanResult, SourceChunk, SourceFile
from codeinsight.ingestion.chunker import chunk_scan_result, chunk_source_file


def test_short_file_is_a_single_chunk() -> None:
    source = SourceFile("app.py", "hello")
    assert chunk_source_file(source) == (SourceChunk("app.py", 1, 1, "hello"),)


def test_trailing_newline_does_not_create_an_extra_line() -> None:
    with_newline = SourceFile("a.py", "one\ntwo\n")
    without_newline = SourceFile("a.py", "one\ntwo")
    expected = (SourceChunk("a.py", 1, 2, "one\ntwo"),)
    assert chunk_source_file(with_newline) == expected
    assert chunk_source_file(without_newline) == expected


def test_single_newline_is_one_empty_line() -> None:
    assert chunk_source_file(SourceFile("a.py", "\n")) == (SourceChunk("a.py", 1, 1, ""),)


def test_max_lines_one_creates_one_chunk_per_line() -> None:
    source = SourceFile("a.py", "a\nb\nc\n")
    expected = (
        SourceChunk("a.py", 1, 1, "a"),
        SourceChunk("a.py", 2, 2, "b"),
        SourceChunk("a.py", 3, 3, "c"),
    )
    assert chunk_source_file(source, max_lines=1) == expected


def test_multiple_chunks_with_partial_last_chunk() -> None:
    source = SourceFile("a.py", "\n".join(f"line{i}" for i in range(1, 11)) + "\n")
    chunks = chunk_source_file(source, max_lines=4)
    assert chunks == (
        SourceChunk("a.py", 1, 4, "line1\nline2\nline3\nline4"),
        SourceChunk("a.py", 5, 8, "line5\nline6\nline7\nline8"),
        SourceChunk("a.py", 9, 10, "line9\nline10"),
    )


def test_internal_empty_lines_are_preserved_and_counted() -> None:
    source = SourceFile("a.py", "a\n\nb\n\nc\n")
    chunks = chunk_source_file(source, max_lines=2)
    assert chunks == (
        SourceChunk("a.py", 1, 2, "a\n"),
        SourceChunk("a.py", 3, 4, "b\n"),
        SourceChunk("a.py", 5, 5, "c"),
    )


def test_crlf_and_cr_are_normalized_to_lf() -> None:
    source = SourceFile("a.py", "one\r\ntwo\rthree\n")
    assert chunk_source_file(source) == (SourceChunk("a.py", 1, 3, "one\ntwo\nthree"),)


def test_empty_text_returns_no_chunks() -> None:
    assert chunk_source_file(SourceFile("a.py", "")) == ()


@pytest.mark.parametrize("max_lines", [0, -1])
def test_non_positive_max_lines_raises(max_lines: int) -> None:
    with pytest.raises(ValueError):
        chunk_source_file(SourceFile("a.py", "x"), max_lines=max_lines)
    with pytest.raises(ValueError):
        chunk_scan_result(
            ScanResult(files=(SourceFile("a.py", "x"),), skipped=()),
            max_lines=max_lines,
        )


def test_chunk_scan_result_validates_max_lines_without_files() -> None:
    with pytest.raises(ValueError):
        chunk_scan_result(ScanResult(files=(), skipped=()), max_lines=0)


def test_chunk_scan_result_preserves_file_order() -> None:
    files = (
        SourceFile("z.py", "z1\nz2\nz3\n"),
        SourceFile("a.py", "a1\na2\n"),
        SourceFile("m.py", "m1\nm2\nm3\nm4\n"),
    )
    scan_result = ScanResult(files=files, skipped=())
    chunks = chunk_scan_result(scan_result, max_lines=2)
    assert [(c.relative_path, c.start_line, c.end_line) for c in chunks] == [
        ("z.py", 1, 2),
        ("z.py", 3, 3),
        ("a.py", 1, 2),
        ("m.py", 1, 2),
        ("m.py", 3, 4),
    ]
