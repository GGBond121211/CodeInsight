import time

from codeinsight.agent.tool_loop import (
    ToolCall,
    ToolLoop,
    ToolLoopConfig,
    ToolModelResponse,
    ToolResult,
)
from codeinsight.infrastructure.tool_registry import build_default_registry


class FakeHost:
    def __init__(self, results):
        self.tools = build_default_registry().list_tools()
        self.results = results
        self.calls = []

    def list_tools(self):
        return self.tools

    def call_tool(self, call):
        self.calls.append(call)
        return self.results.get(call.name, ToolResult.success(call.id, call.name, {"ok": True}))


class AdaptiveModel:
    """第二步的选择依赖第一步 ToolResult，而不是 request_type if 分支。"""

    def __init__(self):
        self.messages = []

    def complete_with_tools(self, messages, tools):
        self.messages.append(messages)
        if not any(item.get("role") == "tool" for item in messages):
            return ToolModelResponse(
                None, (ToolCall("c1", "search_repository", {"question": "checkout"}),), "fake", 1, 1
            )
        return (
            ToolModelResponse(
                "已经读取搜索结果。",
                (ToolCall("c2", "read_file", {"path": "src/service.py"}),),
                "fake",
                1,
                1,
            )
            if sum(item.get("role") == "tool" for item in messages) == 1
            else ToolModelResponse("完成", (), "fake", 1, 1)
        )


def test_tool_result_changes_next_native_tool_call():
    host = FakeHost(
        {
            "search_repository": ToolResult.success(
                "c1", "search_repository", {"results": [{"relative_path": "src/service.py"}]}
            ),
            "read_file": ToolResult.success(
                "c2", "read_file", {"path": "src/service.py", "text": "def checkout(): pass"}
            ),
        }
    )
    result = ToolLoop(AdaptiveModel(), host).run("system", "find checkout")
    assert result.status == "COMPLETED"
    assert [call.name for call in result.tool_calls] == ["search_repository", "read_file"]
    assert result.messages[-1]["content"] == "完成"


def test_readonly_calls_parallel_but_side_effect_calls_are_serial():
    class RecordingHost(FakeHost):
        def __init__(self):
            super().__init__({})
            self.active = 0
            self.max_active = 0

        def call_tool(self, call):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            time.sleep(0.03)
            self.active -= 1
            return ToolResult.success(call.id, call.name, {"done": call.name})

    host = RecordingHost()

    class TwoReadonly:
        def __init__(self):
            self.round = 0

        def complete_with_tools(self, messages, tools):
            self.round += 1
            if self.round == 1:
                return ToolModelResponse(
                    None,
                    (ToolCall("a", "get_repository_map", {}), ToolCall("b", "get_diff", {})),
                    "fake",
                )
            return ToolModelResponse("done", (), "fake")

    result = ToolLoop(TwoReadonly(), host).run("system", "inspect")
    assert result.status == "COMPLETED"
    assert host.max_active == 2


def test_repeated_tool_calls_become_stuck_without_infinite_loop():
    host = FakeHost(
        {"get_diff": ToolResult.success("x", "get_diff", {"diff": ""}, state_fingerprint="same")}
    )

    class Repeater:
        def complete_with_tools(self, messages, tools):
            return ToolModelResponse(None, (ToolCall("x", "get_diff", {}),), "fake")

    result = ToolLoop(Repeater(), host, config=ToolLoopConfig(max_steps=8)).run("system", "inspect")
    assert result.status == "STUCK"
    assert "重复工具调用" in (result.reason or "")


def test_plain_json_text_is_not_converted_to_tool_call():
    host = FakeHost({})

    class TextOnly:
        def complete_with_tools(self, messages, tools):
            return ToolModelResponse('{"name":"read_file","arguments":{"path":"x"}}', (), "fake")

    result = ToolLoop(TextOnly(), host).run("system", "answer")
    assert result.status == "COMPLETED"
    assert host.calls == []


def test_model_provider_failure_is_a_safe_failed_result():
    host = FakeHost({})

    class BrokenModel:
        def complete_with_tools(self, messages, tools):
            raise RuntimeError("provider secret details")

    result = ToolLoop(BrokenModel(), host).run("system", "answer")
    assert result.status == "FAILED"
    assert result.reason == "模型 Provider 调用失败"
    assert "secret" not in (result.reason or "")
