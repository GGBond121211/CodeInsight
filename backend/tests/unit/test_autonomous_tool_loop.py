import time

from codeinsight.agent.tool_loop import (
    ToolCall,
    ToolLoop,
    ToolLoopConfig,
    ToolModelResponse,
    ToolResult,
)
from codeinsight.application.context_budget import estimate_messages_tokens
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


def test_default_budget_allows_extended_readonly_exploration():
    host = FakeHost({})

    class Explorer:
        def __init__(self):
            self.round = 0

        def complete_with_tools(self, messages, tools):
            self.round += 1
            if self.round <= 4:
                return ToolModelResponse(
                    None,
                    tuple(
                        ToolCall(
                            f"search-{self.round}-{index}",
                            "search_repository",
                            {"question": f"evidence-{self.round}-{index}"},
                        )
                        for index in range(5)
                    ),
                    "fake",
                )
            return ToolModelResponse("patch plan ready", (), "fake")

    result = ToolLoop(Explorer(), host).run("system", "inspect")

    assert result.status == "COMPLETED"
    assert len(result.tool_calls) == 20


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


class RecordingModel:
    """记录每次收到什么，用来验证裁剪只作用于发给模型的视图。"""

    def __init__(self, steps):
        self._steps = steps
        self.seen = []

    def complete_with_tools(self, messages, tools):
        self.seen.append([dict(message) for message in messages])
        if len(self.seen) > self._steps:
            return ToolModelResponse("完成", (), "fake", 1, 1)
        return ToolModelResponse(
            None,
            (
                ToolCall(
                    f"c{len(self.seen)}",
                    "read_file",
                    {"path": f"src/app{len(self.seen)}.py"},
                ),
            ),
            "fake",
            1,
            1,
        )


class BigResultHost:
    def __init__(self):
        self.tools = build_default_registry().list_tools()

    def list_tools(self):
        return self.tools

    def call_tool(self, call):
        return ToolResult.success(
            call.id,
            "read_file",
            {
                "path": "src/app.py",
                "start_line": 1,
                "end_line": 200,
                "text": "def handler(): pass\n" * 300,
                "untrusted": True,
            },
            state_fingerprint="fp-app",
        )


def test_tool_loop_prunes_old_results_only_in_the_model_view() -> None:
    model = RecordingModel(steps=5)
    loop = ToolLoop(model, BigResultHost(), config=ToolLoopConfig(max_steps=8))

    result = loop.run("system", "解释入口")

    assert len(result.prune_events) >= 1
    pruned = result.prune_events[-1]
    assert pruned.tokens_saved > 0
    assert {item.tool_name for item in pruned.pruned} == {"read_file"}
    assert len(pruned.pruned) >= 1

    # 模型看到的视图确实变小了。
    last_seen = model.seen[-1]
    seen_tools = [item for item in last_seen if item.get("role") == "tool"]
    assert "text_excerpt" in str(seen_tools[0].get("content"))

    # 但完整记录仍留在结果里，审计和回放不受影响。
    full_tools = [item for item in result.messages if item.get("role") == "tool"]
    assert "text_excerpt" not in str(full_tools[0].get("content"))


def test_pruning_can_be_switched_off() -> None:
    model = RecordingModel(steps=4)
    loop = ToolLoop(
        model,
        BigResultHost(),
        config=ToolLoopConfig(max_steps=6, prune_tool_results=False),
    )

    result = loop.run("system", "解释入口")

    assert result.prune_events == ()
    for seen in model.seen:
        for message in seen:
            if message.get("role") == "tool":
                assert "text_excerpt" not in str(message.get("content"))


def test_keep_recent_tool_results_controls_the_window() -> None:
    model = RecordingModel(steps=3)
    loop = ToolLoop(
        model,
        BigResultHost(),
        config=ToolLoopConfig(max_steps=6, keep_recent_tool_results=3),
    )

    result = loop.run("system", "解释入口")

    assert result.prune_events == ()


def test_context_checks_record_every_call() -> None:
    model = RecordingModel(steps=2)
    loop = ToolLoop(
        model,
        BigResultHost(),
        config=ToolLoopConfig(
            max_steps=4,
            context_window_tokens=128_000,
            reserved_output_tokens=40_960,
        ),
    )

    result = loop.run("system", "解释入口")

    assert len(result.context_checks) == len(model.seen)
    for check in result.context_checks:
        assert check.context_window_tokens == 128_000
        assert check.reserved_output_tokens == 40_960
        assert check.estimated_input_tokens > 0
        assert check.guard_result == "fits"


def test_no_call_is_sent_without_a_window_is_declared() -> None:
    model = RecordingModel(steps=1)
    loop = ToolLoop(model, BigResultHost(), config=ToolLoopConfig(max_steps=2))

    result = loop.run("system", "解释入口")

    assert [check.guard_result for check in result.context_checks] == [
        "not_checked",
        "not_checked",
    ]


def test_loop_stops_locally_when_even_full_pruning_cannot_fit() -> None:
    window = 600
    reserved = 100
    model = RecordingModel(steps=6)
    loop = ToolLoop(
        model,
        BigResultHost(),
        config=ToolLoopConfig(
            max_steps=8,
            context_window_tokens=window,
            reserved_output_tokens=reserved,
        ),
    )

    result = loop.run("system", "解释入口")

    assert result.status == "STUCK"
    assert result.reason == "上下文预算已用尽：压缩后仍超出工作窗口"
    # 关键断言：没有任何一次调用是超窗口发出去的。
    assert len(model.seen) >= 1
    for seen in model.seen:
        assert estimate_messages_tokens(seen) + reserved <= window
    assert result.context_checks[-1].guard_result == "overflow"
