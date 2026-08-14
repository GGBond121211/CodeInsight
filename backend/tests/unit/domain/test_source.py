"""不可变扫描结果值对象测试。"""

from dataclasses import FrozenInstanceError

import pytest

from codeinsight.domain.source import ScanResult, SkippedFile, SourceFile


def test_source_file_fields_and_value_equality() -> None:
    first = SourceFile(relative_path="pkg/mod.py", text="print(1)")
    second = SourceFile(relative_path="pkg/mod.py", text="print(1)")
    assert first.relative_path == "pkg/mod.py"
    assert first.text == "print(1)"
    assert first == second


def test_source_file_is_immutable() -> None:
    item = SourceFile(relative_path="a.py", text="")
    with pytest.raises(FrozenInstanceError):
        item.relative_path = "b.py"


def test_skipped_file_is_immutable() -> None:
    item = SkippedFile(relative_path="a.py", reason="read_error")
    with pytest.raises(FrozenInstanceError):
        item.reason = "read_error"


def test_scan_result_holds_tuples() -> None:
    source = SourceFile(relative_path="a.py", text="x")
    skipped = SkippedFile(relative_path="b.txt", reason="unsupported_extension")
    result = ScanResult(files=(source,), skipped=(skipped,))
    assert result.files == (source,)
    assert result.skipped == (skipped,)
