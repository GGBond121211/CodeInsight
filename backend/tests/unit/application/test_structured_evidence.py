"""Q-012 结构化结果区的契约测试。

这一层的价值全在「可核对」：每一行都必须来自真实工具返回，路径必须在
仓库内，缺行号就说缺行号。任何一条不成立，表格就退化成另一种散文。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeinsight.agent.tool_loop import ToolResult
from codeinsight.application.structured_evidence import (
    DEFINITION,
    EVIDENCE,
    MAP_FILE,
    REFERENCE,
    StructuredEvidenceRow,
    build_structured_evidence,
    is_safe_relative_path,
)


@dataclass(frozen=True)
class FakeEvidenceRecord:
    evidence_id: str
    source_tool: str
    query_or_symbol: str
    path: str
    start_line: int
    end_line: int


def _location_result(tool_name: str, locations: list[dict[str, object]]) -> ToolResult:
    return ToolResult.success(
        "call-1",
        tool_name,
        {"status": "AVAILABLE", "locations": locations, "untrusted": True},
    )


def test_definitions_and_references_become_rows():
    rows, truncated = build_structured_evidence(
        tool_results=[
            _location_result(
                "lsp_definition",
                [{"path": "src/app.py", "start_line": 12, "end_line": 18, "symbol_id": "app.main"}],
            ),
            _location_result(
                "scip_references",
                [{"path": "tests/test_app.py", "start_line": 31, "symbol_id": "app.main"}],
            ),
        ],
        evidence=[],
    )

    assert [row.kind for row in rows] == [DEFINITION, REFERENCE]
    assert rows[0].symbol == "app.main"
    assert (rows[0].start_line, rows[0].end_line) == (12, 18)
    # 没有 end_line 时不补默认值，行区间就等于起点那一行。
    assert (rows[1].start_line, rows[1].end_line) == (31, 31)
    assert truncated is False


def test_paths_outside_the_repository_are_dropped():
    rows, _ = build_structured_evidence(
        tool_results=[
            _location_result(
                "lsp_definition",
                [
                    {"path": "/etc/passwd", "start_line": 1},
                    {"path": "../../secrets.env", "start_line": 1},
                    {"path": "C:/Windows/system32/drivers/etc/hosts", "start_line": 1},
                    {"path": "src/ok.py", "start_line": 3},
                ],
            )
        ],
        evidence=[],
    )

    assert [row.path for row in rows] == ["src/ok.py"]
    assert is_safe_relative_path("src/ok.py") is True
    assert is_safe_relative_path("/abs.py") is False
    assert is_safe_relative_path("a/../../b.py") is False


def test_repository_map_contributes_only_real_file_rows():
    result = ToolResult.success(
        "call-2",
        "get_repository_map",
        {
            "items": [
                {"path": "src/app.py", "kind": "file", "module": "src.app", "line_hint": 1},
                {"path": "src/app.py", "kind": "symbol", "symbol": "main", "line_hint": None},
                {"path": "src/util.py", "kind": "file", "module": "src.util"},
            ]
        },
    )

    rows, _ = build_structured_evidence(tool_results=[result], evidence=[])

    # 符号条目没有可信行号，宁可不要那一行，也不写一个假的行号。
    assert [row.kind for row in rows] == [MAP_FILE, MAP_FILE]
    assert [row.path for row in rows] == ["src/app.py", "src/util.py"]
    assert rows[0].start_line is None


def test_evidence_records_carry_their_ledger_id():
    rows, _ = build_structured_evidence(
        tool_results=[],
        evidence=[
            FakeEvidenceRecord(
                "E1", "search_repository", "checkout 校验", "src/checkout.py", 5, 20
            ),
            FakeEvidenceRecord("E2", "read_file", "src/app.py", "src/app.py", 1, 9),
        ],
    )

    assert [row.kind for row in rows] == [EVIDENCE, EVIDENCE]
    assert [row.evidence_id for row in rows] == ["E1", "E2"]
    assert rows[0].symbol == "checkout 校验"


def test_failed_tool_results_never_become_rows():
    failed = ToolResult.failure(
        "call-3",
        "lsp_definition",
        "NOT_CONFIGURED",
        "未配置语言服务器",
        data={"locations": [{"path": "src/app.py", "start_line": 1}]},
    )

    rows, _ = build_structured_evidence(tool_results=[failed], evidence=[])

    assert rows == ()


def test_rows_are_deduplicated_and_truncation_is_reported():
    duplicate = {"path": "src/app.py", "start_line": 4, "end_line": 8}
    rows, truncated = build_structured_evidence(
        tool_results=[_location_result("lsp_definition", [duplicate, dict(duplicate)])],
        evidence=[],
        max_rows=1,
    )
    assert len(rows) == 1
    assert truncated is False

    rows, truncated = build_structured_evidence(
        tool_results=[
            _location_result(
                "lsp_definition",
                [
                    {"path": "src/a.py", "start_line": 1},
                    {"path": "src/b.py", "start_line": 2},
                ],
            )
        ],
        evidence=[],
        max_rows=1,
    )
    assert len(rows) == 1
    assert truncated is True


def test_unknown_kind_and_empty_path_are_rejected():
    with pytest.raises(ValueError):
        StructuredEvidenceRow(kind="guess", source_tool="lsp_definition", path="src/a.py")
    with pytest.raises(ValueError):
        StructuredEvidenceRow(kind=DEFINITION, source_tool="lsp_definition", path="  ")
    with pytest.raises(ValueError):
        StructuredEvidenceRow(
            kind=DEFINITION,
            source_tool="lsp_definition",
            path="src/a.py",
            start_line=0,
        )
