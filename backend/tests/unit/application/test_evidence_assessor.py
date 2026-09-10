"""Evidence 评估器的确定性契约测试（Q-009）。"""

from codeinsight.agent.tool_loop import ToolCall, ToolResult
from codeinsight.application import evidence_assessor as assessor_module
from codeinsight.application.evidence_assessor import (
    CONTRADICTORY,
    CONTRADICTORY_EVIDENCE,
    INSUFFICIENT,
    INVALID,
    LOW_COVERAGE,
    MISSING_REQUIRED_EVIDENCE,
    NO_RESULTS,
    SUFFICIENT,
    EvidenceAssessor,
    extract_hard_anchors,
)
from codeinsight.application.evidence_ledger import EvidenceLedger


def _ledger_with(items: list[tuple[str, int, int, str]]) -> EvidenceLedger:
    ledger = EvidenceLedger()
    ledger.ingest(
        call=ToolCall(id="call", name="search_repository", arguments={}),
        result=ToolResult.success(
            "call",
            "search_repository",
            {
                "results": [
                    {
                        "relative_path": path,
                        "start_line": start,
                        "end_line": end,
                        "text": text,
                        "rank": index,
                    }
                    for index, (path, start, end, text) in enumerate(items, start=1)
                ]
            },
        ),
    )
    return ledger


def test_no_results_is_insufficient_with_no_results_reason() -> None:
    assessment = EvidenceAssessor().assess(question="购物车怎么算总价", ledger=EvidenceLedger())
    assert assessment.status == INSUFFICIENT
    assert assessment.reasons == (NO_RESULTS,)
    assert "导航线索" in assessment.guidance


def test_navigation_only_run_is_flagged() -> None:
    assessment = EvidenceAssessor().assess(
        question="购物车怎么算总价",
        ledger=EvidenceLedger(),
        navigation_seen=True,
    )
    assert assessment.navigation_only is True


def test_sufficient_needs_both_count_and_anchor_coverage() -> None:
    ledger = _ledger_with(
        [
            ("src/cart.py", 1, 40, "def total(items): return sum(price(item) for item in items)"),
            ("src/price.py", 1, 30, "def price(item): return item.price"),
        ]
    )
    assessment = EvidenceAssessor().assess(question="购物车怎么算总价", ledger=ledger)
    assert assessment.status == SUFFICIENT
    assert assessment.evidence_count == 2
    assert assessment.distinct_paths == 2


def test_low_coverage_when_below_min_evidence() -> None:
    ledger = _ledger_with([("src/cart.py", 1, 40, "def total(items): pass")])
    assessment = EvidenceAssessor().assess(question="购物车怎么算总价", ledger=ledger)
    assert assessment.status == INSUFFICIENT
    assert LOW_COVERAGE in assessment.reasons


def test_missing_hard_anchor_blocks_sufficiency() -> None:
    ledger = _ledger_with(
        [
            ("src/cart.py", 1, 40, "def total(items): pass"),
            ("src/price.py", 1, 30, "def price(item): pass"),
        ]
    )
    assessment = EvidenceAssessor().assess(
        question="localHelper 是怎么实现的", ledger=ledger
    )
    assert assessment.status == INSUFFICIENT
    assert assessment.reasons == (MISSING_REQUIRED_EVIDENCE,)
    assert assessment.missing_anchors == ("localHelper",)


def test_covered_hard_anchor_does_not_block() -> None:
    ledger = _ledger_with(
        [
            ("src/use.ts", 1, 40, "export const value = localHelper();"),
            ("src/use.ts", 41, 60, "function localHelper(): number { return 42; }"),
        ]
    )
    assessment = EvidenceAssessor().assess(question="localHelper 做什么", ledger=ledger)
    assert assessment.status == SUFFICIENT
    assert assessment.covered_anchors == ("localHelper",)


def test_same_location_with_two_fingerprints_is_contradictory() -> None:
    ledger = EvidenceLedger()
    for index, text in enumerate(("def total(): return 1", "def total(): return 2"), start=1):
        ledger.ingest(
            call=ToolCall(
                id=f"call-{index}", name="read_file", arguments={"path": "src/cart.py"}
            ),
            result=ToolResult.success(
                f"call-{index}",
                "read_file",
                {
                    "path": "src/cart.py",
                    "start_line": 10,
                    "end_line": 20,
                    "text": text,
                },
            ),
        )
    assessment = EvidenceAssessor().assess(question="购物车怎么算总价", ledger=ledger)
    assert assessment.status == CONTRADICTORY
    assert assessment.reasons == (CONTRADICTORY_EVIDENCE,)


def test_invalid_signal_wins_over_everything_else() -> None:
    ledger = _ledger_with(
        [
            ("src/cart.py", 1, 40, "def total(items): pass"),
            ("src/price.py", 1, 30, "def price(item): pass"),
        ]
    )
    assessment = EvidenceAssessor().assess(
        question="购物车怎么算总价",
        ledger=ledger,
        invalid_signals=("search_repository:path 越界或格式非法",),
    )
    assert assessment.status == INVALID
    assert assessment.reasons == (assessor_module.INVALID_OR_STALE_LOCATION,)


def test_hard_anchor_extraction_ignores_plain_english_words() -> None:
    assert extract_hard_anchors("how does the function work") == ()
    assert extract_hard_anchors("explain chunk_max_lines") == ("chunk_max_lines",)
    assert extract_hard_anchors("看一下 localHelper") == ("localHelper",)
    assert extract_hard_anchors("读 src/shop/inventory.py") == ("src/shop/inventory.py",)
    assert extract_hard_anchors("same word twice: chunk_max_lines and chunk_max_lines") == (
        "chunk_max_lines",
    )
