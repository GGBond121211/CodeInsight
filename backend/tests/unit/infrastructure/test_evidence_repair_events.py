"""Evidence 评估与 Repair 的公开事件契约（Q-009 Task 5）。"""

from codeinsight.agent.code_understanding_tool_loop import CodeUnderstandingToolLoop
from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse, ToolResult
from codeinsight.infrastructure.tool_registry import build_default_registry


class _Host:
    def __init__(self, results: list[dict]) -> None:
        self._results = results
        self.tools = build_default_registry().list_tools()

    def list_tools(self):
        return self.tools

    def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name == "search_repository":
            return ToolResult.success(
                call.id, call.name, {"results": self._results, "untrusted": True}
            )
        return ToolResult.success(call.id, call.name, {"ok": True})


class _Model:
    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self._responses = list(responses)

    def complete_with_tools(self, messages, tools):
        if self._responses:
            return self._responses.pop(0)
        return ToolModelResponse("{}", (), "fake")


def _search(call_id: str) -> ToolModelResponse:
    return ToolModelResponse(
        None,
        (ToolCall(call_id, "search_repository", {"question": "购物车怎么计算总价"}),),
        "fake",
    )


def test_assessment_event_records_only_status_and_counts() -> None:
    events: list[tuple[str, dict]] = []
    model = _Model(
        [
            _search("c1"),
            ToolModelResponse(
                '{"outcome":"insufficient_evidence","answer":"不足","citations":[]}',
                (),
                "fake",
            ),
        ]
    )
    CodeUnderstandingToolLoop(
        model,
        _Host([]),
        emit=lambda event_type, payload: events.append((event_type, dict(payload))),
    ).run("购物车怎么计算总价")

    assessed = [payload for kind, payload in events if kind == "evidence_assessed"]
    assert assessed
    assert assessed[0]["status"] == "insufficient"
    assert assessed[0]["reasons"] == "NO_RESULTS"
    assert assessed[0]["evidence_count"] == "0"
    # 只允许出现状态、计数和动作字段，不携带仓库正文或推理。
    for payload in assessed:
        assert set(payload) <= {
            "step",
            "retrieval_round",
            "status",
            "reasons",
            "evidence_count",
            "new_evidence_count",
        }


def test_repair_round_has_start_and_finish_events() -> None:
    events: list[tuple[str, dict]] = []
    model = _Model(
        [
            _search("c1"),
            _search("c2"),
            ToolModelResponse(
                '{"outcome":"insufficient_evidence","answer":"不足","citations":[]}',
                (),
                "fake",
            ),
        ]
    )
    CodeUnderstandingToolLoop(
        model,
        _Host([]),
        emit=lambda event_type, payload: events.append((event_type, dict(payload))),
    ).run("购物车怎么计算总价")

    kinds = [kind for kind, _ in events]
    assert "evidence_repair_started" in kinds
    assert "evidence_repair_finished" in kinds
    assert kinds.index("evidence_repair_started") < kinds.index("evidence_repair_finished")
    started = next(payload for kind, payload in events if kind == "evidence_repair_started")
    assert started["round"] == "1"
    assert started["repair_action"] == "A_query_broaden"
    finished = next(payload for kind, payload in events if kind == "evidence_repair_finished")
    assert finished["new_evidence_count"] == "0"


def test_no_repair_events_when_evidence_is_sufficient_immediately() -> None:
    events: list[tuple[str, dict]] = []
    hits = [
        {
            "relative_path": "src/cart.py",
            "start_line": 1,
            "end_line": 40,
            "text": "def total(items): pass",
            "rank": 1,
        },
        {
            "relative_path": "src/price.py",
            "start_line": 1,
            "end_line": 30,
            "text": "def price(item): pass",
            "rank": 2,
        },
    ]
    model = _Model(
        [
            _search("c1"),
            ToolModelResponse(
                '{"outcome":"answered","answer":"ok","citations":["E1","E2"]}',
                (),
                "fake",
            ),
        ]
    )
    CodeUnderstandingToolLoop(
        model,
        _Host(hits),
        emit=lambda event_type, payload: events.append((event_type, dict(payload))),
    ).run("购物车怎么计算总价")
    kinds = [kind for kind, _ in events]
    assert "evidence_assessed" in kinds
    assert "evidence_repair_started" not in kinds
