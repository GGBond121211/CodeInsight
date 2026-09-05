"""结构优先切块测试。"""

import pytest

from codeinsight.domain.source import SourceFile
from codeinsight.ingestion.structural_chunker import chunk_structural_source_file


def test_top_level_functions_and_classes_keep_structure_ranges() -> None:
    source = SourceFile(
        "app.py",
        "\n".join(
            [
                "import os",
                "",
                "CONST = 1",
                "",
                "def first():",
                "    return 1",
                "",
                "class Service:",
                "    def run(self):",
                "        return CONST",
            ]
        ),
    )
    chunks = chunk_structural_source_file(source)
    assert [(item.start_line, item.end_line, item.symbol_path) for item in chunks] == [
        (1, 4, "<module>"),
        (5, 6, "first"),
        (7, 7, "<module>"),
        (8, 10, "Service"),
    ]


def test_long_structure_falls_back_to_fixed_ranges_without_losing_lines() -> None:
    lines = ["def large():"]
    for index in range(1, 8):
        lines.append(f"    value_{index} = {index}")
    chunks = chunk_structural_source_file(
        SourceFile("large.py", "\n".join(lines)),
        fallback_max_lines=3,
        max_structure_lines=4,
    )
    assert [(item.start_line, item.end_line) for item in chunks] == [
        (1, 3),
        (4, 6),
        (7, 8),
    ]
    assert all(item.symbol_path == "large" for item in chunks)


@pytest.mark.parametrize(
    ("path", "text"),
    [("notes.txt", "one\ntwo"), ("broken.py", "def broken(:\n")],
)
def test_non_python_and_syntax_error_fall_back_to_fixed_chunks(path: str, text: str) -> None:
    chunks = chunk_structural_source_file(SourceFile(path, text), fallback_max_lines=1)
    expected_ranges = [(1, 1), (2, 2)] if path == "notes.txt" else [(1, 1)]
    assert [(item.start_line, item.end_line) for item in chunks] == expected_ranges


def test_invalid_structure_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_structural_source_file(SourceFile("a.py", "x = 1"), max_structure_lines=0)
