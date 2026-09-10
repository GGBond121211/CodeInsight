"""代码理解 MCP Tool Loop 的只读边界、证据映射与 Repair 接入测试（Q-008 / Q-009）。"""

import time
from threading import Event

from codeinsight.agent.code_understanding_tool_loop import (
    ABORTED_STATUS,
    ANSWERED_STATUS,
    INSUFFICIENT_STATUS,
    PARTIAL_ANSWERED_STATUS,
    STUCK_STATUS,
    TIMEOUT_STATUS,
    CodeUnderstandingConfig,
    CodeUnderstandingToolLoop,
    filter_readonly_tools,
)
from codeinsight.agent.tool_loop import (
    ToolCall,
    ToolLoopConfig,
    ToolModelResponse,
    ToolResult,
)
from codeinsight.application.evidence_repair import RepairBudget
from codeinsight.infrastructure.tool_registry import build_default_registry

HIT_TOTAL = {
    "relative_path": "src/cart.py",
    "start_line": 1,
    "end_line": 40,
    "text": "def total(items): return sum(price(item) for item in items)",
    "rank": 1,
}
HIT_PRICE = {
    "relative_path": "src/price.py",
    "start_line": 1,
    "end_line": 30,
    "text": "def price(item): return item.price",
    "rank": 2,
}


class FakeHost:
    def __init__(
        self, results: dict[str, ToolResult] | None = None, *, delay_seconds: float = 0.0
    ) -> None:
        self.tools = build_default_registry().list_tools()
        self.results = results or {}
        self.delay_seconds = delay_seconds
        self.calls: list[ToolCall] = []
        self.listed = 0

    def list_tools(self):
        self.listed += 1
        return self.tools

    def call_tool(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        found = self.results.get(call.name)
        if found is not None:
            return ToolResult(
                call.id, call.name, found.ok, dict(found.data), found.error_code,
                found.error_message, found.state_fingerprint,
            )
        return ToolResult.success(call.id, call.name, {"ok": True})


class ScriptedModel:
    """按脚本逐轮返回响应；脚本用尽后返回空答案。"""

    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self._responses = list(responses)
        self.seen_messages: list[tuple] = []
        self.seen_tools: list[tuple] = []

    def complete_with_tools(self, messages, tools):
        self.seen_messages.append(tuple(messages))
        self.seen_tools.append(tuple(tools))
        if self._responses:
            return self._responses.pop(0)
        return ToolModelResponse("{}", (), "fake")


def _search_call(question: str = "购物车怎么计算总价", call_id: str = "c1") -> ToolModelResponse:
    return ToolModelResponse(
        None,
        (ToolCall(call_id, "search_repository", {"question": question}),),
        "fake",
        1,
        1,
    )


def _final(payload: str) -> ToolModelResponse:
    return ToolModelResponse(payload, (), "fake", 1, 1)


def _search_host(*hits: dict) -> FakeHost:
    return FakeHost(
        {
            "search_repository": ToolResult.success(
                "c1", "search_repository", {"results": list(hits), "untrusted": True}
            )
        }
    )


def test_readonly_view_hides_every_write_tool() -> None:
    names = {item["name"] for item in filter_readonly_tools(build_default_registry().list_tools())}
    assert names == {
        "get_repository_map",
        "search_repository",
        "read_file",
        "get_evidence_context",
        "lsp_definition",
        "scip_references",
    }
    for forbidden in ("generate_patch", "apply_patch_isolated", "rollback_workspace"):
        assert forbidden not in names


def test_write_tool_call_is_rejected_even_if_the_model_asks_for_it() -> None:
    host = FakeHost()
    model = ScriptedModel(
        [
            ToolModelResponse(
                None,
                (ToolCall("c1", "generate_patch", {"path": "src/a.py"}),),
                "fake",
            ),
            _final('{"outcome":"insufficient_evidence","answer":"无法","citations":[]}'),
        ]
    )
    loop = CodeUnderstandingToolLoop(model, host)
    result = loop.run("购物车怎么计算总价")
    # 写工具必须在 MCP 视图层就被拒绝，不落到真实 Client。
    assert all(call.name != "generate_patch" for call in host.calls)
    assert result.evidence == ()


def test_happy_path_maps_ledger_ids_to_citations() -> None:
    host = _search_host(HIT_TOTAL, HIT_PRICE)
    model = ScriptedModel(
        [
            _search_call(),
            _final(
                '{"outcome":"answered","answer":"总价由 total 汇总",'
                '"citations":["E1","E2"]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.status == ANSWERED_STATUS
    assert [item.evidence_id for item in result.citations] == ["E1", "E2"]
    assert [item.relative_path for item in result.citations] == ["src/cart.py", "src/price.py"]
    assert result.repair_rounds == 0
    assert result.assessment.sufficient is True


def test_evidence_ids_are_backfilled_to_the_model() -> None:
    host = _search_host(HIT_TOTAL, HIT_PRICE)
    model = ScriptedModel(
        [
            _search_call(),
            _final(
                '{"outcome":"answered","answer":"x","citations":["E1","E2"]}'
            ),
        ]
    )
    CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    second_turn = model.seen_messages[1]
    injected = [item for item in second_turn if item.get("role") == "user"]
    assert any("evidence-index" in str(item.get("content")) for item in injected)
    assert any("E1 = src/cart.py:1-40" in str(item.get("content")) for item in injected)


def test_model_cannot_cite_an_id_it_invented() -> None:
    host = _search_host(HIT_TOTAL, HIT_PRICE)
    model = ScriptedModel(
        [
            _search_call(),
            _final('{"outcome":"answered","answer":"x","citations":["E9"]}'),
            _final(
                '{"outcome":"answered","answer":"x","citations":["E1","E2"]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.status == ANSWERED_STATUS
    assert [item.evidence_id for item in result.citations] == ["E1", "E2"]


def test_invented_id_that_is_never_fixed_fails_closed() -> None:
    host = _search_host(HIT_TOTAL, HIT_PRICE)
    model = ScriptedModel(
        [
            _search_call(),
            _final('{"outcome":"answered","answer":"x","citations":["E9"]}'),
            _final('{"outcome":"answered","answer":"x","citations":["E8"]}'),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.status == STUCK_STATUS
    assert result.unknown_evidence_ids == ("E8",)
    assert result.citations == ()


def test_answered_claim_is_downgraded_when_evidence_is_thin() -> None:
    host = _search_host(HIT_TOTAL)
    model = ScriptedModel(
        [
            _search_call(),
            _final('{"outcome":"answered","answer":"x","citations":["E1"]}'),
            _final('{"outcome":"answered","answer":"仍然坚持","citations":["E1"]}'),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.status == PARTIAL_ANSWERED_STATUS
    assert result.answer == "仍然坚持"
    assert result.assessment.sufficient is False


def test_insufficient_outcome_is_a_legitimate_final_state() -> None:
    host = _search_host(HIT_TOTAL)
    model = ScriptedModel(
        [
            _search_call(),
            _final(
                '{"outcome":"insufficient_evidence","answer":"只有一处证据",'
                '"citations":[]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.status == INSUFFICIENT_STATUS
    assert result.citations == ()


def test_navigation_tools_alone_never_produce_evidence() -> None:
    host = FakeHost(
        {
            "get_repository_map": ToolResult.success(
                "c1", "get_repository_map", {"files": [{"path": "src/cart.py"}]}
            )
        }
    )
    model = ScriptedModel(
        [
            ToolModelResponse(
                None, (ToolCall("c1", "get_repository_map", {}),), "fake"
            ),
            _final(
                '{"outcome":"insufficient_evidence","answer":"只有导航线索",'
                '"citations":[]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.evidence == ()
    assert result.navigation_seen is True
    assert result.status == INSUFFICIENT_STATUS


def test_repair_round_is_injected_once_and_recorded() -> None:
    host = FakeHost(
        {
            "search_repository": ToolResult.success(
                "c1", "search_repository", {"results": [], "untrusted": True}
            )
        }
    )
    model = ScriptedModel(
        [
            _search_call(),
            _final(
                '{"outcome":"insufficient_evidence","answer":"没找到",'
                '"citations":[]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(model, host).run("购物车怎么计算总价")
    assert result.repair_rounds == 1
    assert result.repair_actions == ("A_query_broaden",)
    assert result.status == INSUFFICIENT_STATUS


def test_repair_budget_zero_disables_the_loop() -> None:
    host = FakeHost(
        {
            "search_repository": ToolResult.success(
                "c1", "search_repository", {"results": [], "untrusted": True}
            )
        }
    )
    model = ScriptedModel(
        [
            _search_call(),
            _final(
                '{"outcome":"insufficient_evidence","answer":"没找到",'
                '"citations":[]}'
            ),
        ]
    )
    result = CodeUnderstandingToolLoop(
        model,
        host,
        config=CodeUnderstandingConfig(repair=RepairBudget(max_repair_rounds=0)),
    ).run("购物车怎么计算总价")
    assert result.repair_rounds == 0


def test_deadline_maps_to_timeout() -> None:
    # deadline 用真实耗时触发，不用亚毫秒值：Windows 的 time.monotonic()
    # 分辨率约 15.6ms，微秒级 deadline 会让这个断言变成随机结果。
    host = FakeHost(
        {
            "search_repository": ToolResult.success(
                "c1", "search_repository", {"results": [HIT_TOTAL, HIT_PRICE], "untrusted": True}
            )
        },
        delay_seconds=0.12,
    )
    model = ScriptedModel([_search_call(call_id="c1"), _search_call(call_id="c2")])
    result = CodeUnderstandingToolLoop(
        model,
        host,
        config=CodeUnderstandingConfig(
            loop=ToolLoopConfig(max_steps=6, max_tool_calls=32, deadline_seconds=0.05)
        ),
    ).run("购物车怎么计算总价")
    assert result.status == TIMEOUT_STATUS
    assert "deadline" in result.termination_reason


def test_max_steps_terminates_the_run() -> None:
    host = _search_host(HIT_TOTAL, HIT_PRICE)
    model = ScriptedModel(
        [
            _search_call(call_id="c1"),
            _search_call(call_id="c2"),
            _search_call(call_id="c3"),
            _search_call(call_id="c4"),
        ]
    )
    result = CodeUnderstandingToolLoop(
        model,
        host,
        config=CodeUnderstandingConfig(
            loop=ToolLoopConfig(max_steps=2, max_tool_calls=64, deadline_seconds=60.0)
        ),
    ).run("购物车怎么计算总价")
    assert result.status in {STUCK_STATUS, TIMEOUT_STATUS}
    assert result.steps <= 2


def test_cancel_returns_aborted() -> None:
    cancelled = Event()
    cancelled.set()
    host = _search_host(HIT_TOTAL)
    model = ScriptedModel([_search_call()])
    result = CodeUnderstandingToolLoop(model, host).run(
        "购物车怎么计算总价", cancel_event=cancelled
    )
    assert result.status == ABORTED_STATUS
    assert result.evidence == ()
