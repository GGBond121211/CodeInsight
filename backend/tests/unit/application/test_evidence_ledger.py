"""Evidence Ledger 的契约测试（Q-008 / Q-009）。"""

from codeinsight.agent.tool_loop import ToolCall, ToolResult
from codeinsight.application.evidence_ledger import EvidenceLedger


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=f"call-{name}", name=name, arguments=dict(arguments))


def _search_result(*items: dict) -> ToolResult:
    return ToolResult.success(
        "call-search_repository", "search_repository", {"results": list(items)}
    )


def _hit(path: str, start: int, end: int, text: str, rank: int = 1) -> dict:
    return {
        "relative_path": path,
        "start_line": start,
        "end_line": end,
        "text": text,
        "rank": rank,
    }


def test_search_results_become_numbered_evidence() -> None:
    ledger = EvidenceLedger()
    report = ledger.ingest(
        call=_call("search_repository", question="checkout"),
        result=_search_result(
            _hit("src/a.py", 1, 5, "def a(): pass"),
            _hit("src/b.py", 7, 9, "def b(): pass", rank=2),
        ),
    )
    assert [record.evidence_id for record in report.added] == ["E1", "E2"]
    assert [record.path for record in ledger.records] == ["src/a.py", "src/b.py"]
    assert report.added[0].query_or_symbol == "checkout"
    assert report.added[0].untrusted is True


def test_same_location_is_deduplicated_across_tools_and_rounds() -> None:
    ledger = EvidenceLedger()
    hit = _hit("src/a.py", 1, 5, "def a(): pass")
    first = ledger.ingest(call=_call("search_repository"), result=_search_result(hit))
    second = ledger.ingest(
        call=_call("get_evidence_context"),
        result=ToolResult.success(
            "call-get_evidence_context", "get_evidence_context", {"evidence": [hit]}
        ),
    )
    assert first.new_count == 1
    assert second.new_count == 0
    assert second.duplicates == ("E1",)
    assert len(ledger) == 1
    # 多个来源命中同一块时保留来源标签，但不重复占用额度。
    assert ledger.by_id("E1").routes == ("get_evidence_context", "search_repository")


def test_navigation_tools_never_become_evidence() -> None:
    ledger = EvidenceLedger()
    report = ledger.ingest(
        call=_call("get_repository_map"),
        result=ToolResult.success(
            "call-get_repository_map", "get_repository_map", {"files": [{"path": "src/a.py"}]}
        ),
    )
    assert report.new_count == 0
    assert len(ledger) == 0


def test_failed_tool_result_is_not_evidence() -> None:
    ledger = EvidenceLedger()
    report = ledger.ingest(
        call=_call("search_repository"),
        result=ToolResult.failure("call-search_repository", "search_repository", "NOT_FOUND", "空"),
    )
    assert report.new_count == 0
    assert len(ledger) == 0


def test_out_of_repository_paths_are_rejected_with_a_reason() -> None:
    ledger = EvidenceLedger()
    report = ledger.ingest(
        call=_call("search_repository"),
        result=_search_result(
            _hit("../../etc/passwd", 1, 2, "root:x:0:0"),
            _hit("/abs/path.py", 1, 2, "x = 1"),
        ),
    )
    assert report.new_count == 0
    assert report.skipped_reasons == ("path 越界或格式非法", "path 越界或格式非法")


def test_invalid_line_range_is_rejected() -> None:
    ledger = EvidenceLedger()
    report = ledger.ingest(
        call=_call("read_file"),
        result=ToolResult.success(
            "call-read_file",
            "read_file",
            {"path": "src/a.py", "start_line": 9, "end_line": 3, "text": "x"},
        ),
    )
    assert report.new_count == 0
    assert report.skipped_reasons == ("行号非法",)


def test_max_evidence_caps_the_ledger() -> None:
    ledger = EvidenceLedger(max_evidence=2)
    report = ledger.ingest(
        call=_call("search_repository"),
        result=_search_result(
            _hit("src/a.py", 1, 2, "a"),
            _hit("src/b.py", 1, 2, "b"),
            _hit("src/c.py", 1, 2, "c"),
        ),
    )
    assert report.new_count == 2
    assert report.truncated is True
    assert len(ledger) == 2


def test_unknown_ids_are_reported_for_model_self_reported_citations() -> None:
    ledger = EvidenceLedger()
    ledger.ingest(
        call=_call("read_file"),
        result=ToolResult.success(
            "call-read_file",
            "read_file",
            {"path": "src/a.py", "start_line": 1, "end_line": 2, "text": "x"},
        ),
    )
    assert ledger.unknown_ids(("E1", "E9", "E9")) == ("E9",)


def test_excerpt_is_bounded() -> None:
    ledger = EvidenceLedger(excerpt_chars=10)
    report = ledger.ingest(
        call=_call("read_file"),
        result=ToolResult.success(
            "call-read_file",
            "read_file",
            {"path": "src/a.py", "start_line": 1, "end_line": 1, "text": "x" * 500},
        ),
    )
    assert len(report.added[0].excerpt) == 10
