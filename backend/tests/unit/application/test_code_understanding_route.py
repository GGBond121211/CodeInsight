"""Q-008 路由接入：迁移开关与结果映射的契约测试。"""

from types import SimpleNamespace

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse, ToolResult
from codeinsight.application.conversation_service import (
    CODE_UNDERSTANDING_ENV_FLAG,
    ConversationService,
    code_understanding_tool_loop_enabled,
)
from codeinsight.domain.answer import ANSWERED, INSUFFICIENT_EVIDENCE
from codeinsight.infrastructure.tool_registry import build_default_registry


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


def _service(client_factory) -> ConversationService:
    return ConversationService(
        lambda: object(),
        lambda: object(),
        reranker_factory=lambda: object(),
        change_service=SimpleNamespace(),
        mcp_client_factory=client_factory,
    )


def _router_result() -> SimpleNamespace:
    plan = SimpleNamespace(
        subquestions=(SimpleNamespace(question="购物车怎么计算总价", intent="explain"),)
    )
    return SimpleNamespace(
        plan=plan,
        model="router-model",
        input_tokens=11,
        output_tokens=2,
        elapsed_milliseconds=3.0,
        fallback_reason=None,
    )


def test_switch_is_off_by_default() -> None:
    assert code_understanding_tool_loop_enabled({}) is False
    assert code_understanding_tool_loop_enabled({CODE_UNDERSTANDING_ENV_FLAG: "0"}) is False
    assert code_understanding_tool_loop_enabled({CODE_UNDERSTANDING_ENV_FLAG: "1"}) is True
    assert code_understanding_tool_loop_enabled({CODE_UNDERSTANDING_ENV_FLAG: " 1 "}) is True


def test_service_defaults_to_the_legacy_path(monkeypatch) -> None:
    monkeypatch.delenv(CODE_UNDERSTANDING_ENV_FLAG, raising=False)
    service = _service(lambda root: _FakeClient(root, []))
    assert service.code_understanding_enabled is False


def test_switch_on_runs_the_readonly_tool_loop(monkeypatch) -> None:
    monkeypatch.setenv(CODE_UNDERSTANDING_ENV_FLAG, "1")
    hits = [
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
    model = _ScriptedModel(
        [
            ToolModelResponse(
                None,
                (ToolCall("c1", "search_repository", {"question": "购物车怎么计算总价"}),),
                "fake-model",
            ),
            ToolModelResponse(
                '{"outcome":"answered","answer":"由 total 汇总",'
                '"citations":["E1","E2"]}',
                (),
                "fake-model",
            ),
        ]
    )
    service = _service(lambda root: _FakeClient(root, hits))
    answer, payload = service._run_code_understanding(  # noqa: SLF001 - 直接测映射
        SimpleNamespace(run_id="run-1", user_message="购物车怎么计算总价"),
        SimpleNamespace(repository_root="C:/fixture"),
        model,
        _router_result(),
    )
    assert answer.outcome == ANSWERED
    assert answer.answer == "由 total 汇总"
    assert [item.relative_path for item in answer.citations] == ["src/cart.py", "src/price.py"]
    assert answer.model == "fake-model"
    assert answer.router_input_tokens == 11
    # 工具循环的确定性判定随公开 payload 一起暴露，便于回溯。
    assert payload["status"] == "ANSWERED"
    assert payload["assessment"]["status"] == "sufficient"
    assert payload["assessment"]["evidence_count"] == 2
    assert payload["repair"]["rounds"] == 0
    assert payload["navigation_seen"] is False
    assert payload["prompt_version"].startswith("code-understanding-tool-loop")


def test_tool_loop_insufficient_maps_to_insufficient_outcome(monkeypatch) -> None:
    monkeypatch.setenv(CODE_UNDERSTANDING_ENV_FLAG, "1")
    model = _ScriptedModel(
        [
            ToolModelResponse(
                None,
                (ToolCall("c1", "search_repository", {"question": "购物车怎么计算总价"}),),
                "fake-model",
            ),
            ToolModelResponse(
                '{"outcome":"insufficient_evidence","answer":"证据不足",'
                '"citations":[]}',
                (),
                "fake-model",
            ),
        ]
    )
    service = _service(lambda root: _FakeClient(root, []))
    answer, payload = service._run_code_understanding(  # noqa: SLF001
        SimpleNamespace(run_id="run-2", user_message="购物车怎么计算总价"),
        SimpleNamespace(repository_root="C:/fixture"),
        model,
        _router_result(),
    )
    assert answer.outcome == INSUFFICIENT_EVIDENCE
    assert answer.citations == ()
    assert payload["status"] == "INSUFFICIENT_EVIDENCE"
    assert payload["repair"]["rounds"] == 1
