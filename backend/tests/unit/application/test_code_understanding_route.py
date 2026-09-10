"""Q-008 路由收口：explain 的唯一入口，以及旧 LangGraph 路线的封闭守卫。

2026-09-10 起 explain 只走只读 Tool Loop。旧的 ``run_citation_agent`` 与
``agent/workflow.py`` 保留在仓库中作为历史实现，但不接受任何运行入口调用。
本文件既验证新路径的映射，也验证三条入口都没有把旧路接回来。
"""

from pathlib import Path
from types import SimpleNamespace

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse, ToolResult
from codeinsight.application.code_understanding_route import (
    run_code_understanding_answer,
    to_auto_answer,
)
from codeinsight.domain.answer import ANSWERED, INSUFFICIENT_EVIDENCE
from codeinsight.infrastructure.tool_registry import build_default_registry

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "codeinsight"

HITS = [
    {
        "relative_path": "src/cart.py",
        "start_line": 1,
        "end_line": 40,
        "text": "def total(items): return sum(price(item) for item in items)",
        "rank": 1,
    },
    {
        "relative_path": "src/price.py",
        "start_line": 1,
        "end_line": 30,
        "text": "def price(item): return item.price",
        "rank": 2,
    },
]


class _FakeClient:
    """最小 MCP Client；只实现只读代码理解需要的两个方法。"""

    def __init__(self, root: str, results: list[dict]) -> None:
        self.root = root
        self._results = results

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def list_tools(self):
        return build_default_registry().list_tools()

    def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name == "search_repository":
            return ToolResult.success(
                call.id, call.name, {"results": self._results, "untrusted": True}
            )
        return ToolResult.success(call.id, call.name, {"ok": True})


class _ScriptedModel:
    def __init__(self, responses: list[ToolModelResponse]) -> None:
        self._responses = list(responses)
        self.model = "fake-model"

    def complete_with_tools(self, messages, tools):
        if self._responses:
            return self._responses.pop(0)
        return ToolModelResponse("{}", (), "fake-model")


def _search(call_id: str = "c1") -> ToolModelResponse:
    return ToolModelResponse(
        None,
        (ToolCall(call_id, "search_repository", {"question": "购物车怎么计算总价"}),),
        "fake-model",
    )


def _final(payload: str) -> ToolModelResponse:
    return ToolModelResponse(payload, (), "fake-model")


def _router_result() -> SimpleNamespace:
    plan = SimpleNamespace(
        subquestions=(
            SimpleNamespace(
                question="购物车怎么计算总价",
                intent="explain",
                retrieval_mode="hybrid",
            ),
        )
    )
    return SimpleNamespace(
        plan=plan,
        model="router-model",
        input_tokens=11,
        output_tokens=2,
        elapsed_milliseconds=3.0,
        fallback_reason=None,
    )


# --- 新路径映射 -------------------------------------------------------------


def test_answered_run_maps_to_answered_with_citations() -> None:
    model = _ScriptedModel(
        [
            _search(),
            _final(
                '{"outcome":"answered","answer":"由 total 汇总",'
                '"citations":["E1","E2"]}'
            ),
        ]
    )
    loop_result = run_code_understanding_answer(
        "C:/fixture",
        "购物车怎么计算总价",
        model=model,
        mcp_client_factory=lambda root: _FakeClient(root, HITS),
    )
    answer = to_auto_answer(loop_result, _router_result(), model_name=model.model)
    assert answer.outcome == ANSWERED
    assert answer.answer == "由 total 汇总"
    assert [item.relative_path for item in answer.citations] == [
        "src/cart.py",
        "src/price.py",
    ]
    assert answer.model == "fake-model"
    assert answer.router_input_tokens == 11
    assert answer.subquestions[0].question == "购物车怎么计算总价"


def test_insufficient_run_maps_to_insufficient_and_never_claims_answered() -> None:
    model = _ScriptedModel(
        [
            _search(),
            _final(
                '{"outcome":"insufficient_evidence","answer":"证据不足",'
                '"citations":[]}'
            ),
        ]
    )
    loop_result = run_code_understanding_answer(
        "C:/fixture",
        "购物车怎么计算总价",
        model=model,
        mcp_client_factory=lambda root: _FakeClient(root, []),
    )
    answer = to_auto_answer(loop_result, _router_result(), model_name=model.model)
    assert answer.outcome == INSUFFICIENT_EVIDENCE
    assert answer.citations == ()
    assert loop_result.repair_rounds == 1


def test_non_answered_status_never_surfaces_as_answered() -> None:
    """部分回答、超时、卡住都不得以 answered 的身份交付。"""
    from codeinsight.agent.code_understanding_tool_loop import (
        ABORTED_STATUS,
        FAILED_STATUS,
        PARTIAL_ANSWERED_STATUS,
        STUCK_STATUS,
        TIMEOUT_STATUS,
    )

    for status in (
        PARTIAL_ANSWERED_STATUS,
        STUCK_STATUS,
        TIMEOUT_STATUS,
        FAILED_STATUS,
        ABORTED_STATUS,
    ):
        stub = SimpleNamespace(
            status=status,
            answer="文字保留",
            citations=(),
            prompt_version="p",
            input_tokens=0,
            output_tokens=0,
            evidence=(),
            repair_rounds=0,
            repair_actions=(),
            assessment=SimpleNamespace(status="insufficient"),
            tool_calls=0,
            steps=0,
            termination_reason="stub",
        )
        answer = to_auto_answer(stub, _router_result(), model_name="m")
        assert answer.outcome == INSUFFICIENT_EVIDENCE, status


def test_public_events_carry_counts_not_reasoning() -> None:
    model = _ScriptedModel(
        [
            _search(),
            _final(
                '{"outcome":"answered","answer":"ok","citations":["E1","E2"]}'
            ),
        ]
    )
    loop_result = run_code_understanding_answer(
        "C:/fixture",
        "购物车怎么计算总价",
        model=model,
        mcp_client_factory=lambda root: _FakeClient(root, HITS),
    )
    answer = to_auto_answer(loop_result, _router_result(), model_name="m")
    assert answer.events
    for event in answer.events:
        assert event.summary
        # 事件只描述计数与状态，不携带仓库正文。
        assert "def total" not in event.summary


# --- 封闭守卫：旧 LangGraph 路线不得被接回 ------------------------------------


def test_entry_points_do_not_reference_the_legacy_agent() -> None:
    """三条入口都不得导入或调用 ``run_citation_agent``。

    这条断言是这次封闭的守卫：如果以后有人把旧分支接回来，测试会立刻失败。
    只检查「实际导入」和「实际调用」，不检查注释里提到这个名字——
    停用说明本来就需要写出被停用的函数名。
    """
    for relative in (
        "application/conversation_service.py",
        "api/routes.py",
        "cli/main.py",
    ):
        text = (SOURCE_ROOT / relative).read_text(encoding="utf-8")
        assert "run_citation_agent(" not in text, relative
        assert "from codeinsight.agent.workflow import" not in text, relative


def test_legacy_implementation_still_exists_for_history() -> None:
    """封闭不等于删除：旧实现按用户要求保留在仓库里。"""
    workflow = SOURCE_ROOT / "agent" / "workflow.py"
    assert workflow.is_file()
    assert "def run_citation_agent" in workflow.read_text(encoding="utf-8")


def test_legacy_agent_is_not_imported_by_the_new_path() -> None:
    from codeinsight.application import code_understanding_route

    assert not hasattr(code_understanding_route, "run_citation_agent")
