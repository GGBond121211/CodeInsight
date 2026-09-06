from threading import Event

from codeinsight.agent.tool_loop import ToolCall, ToolLoop, ToolModelResponse, ToolResult
from codeinsight.domain.trace import (
    TOOL_ABORTED,
    TOOL_CALL_REQUESTED,
    TOOL_CALL_VALIDATED,
    TOOL_DISPATCHED,
    TOOL_RESULT_COMMITTED,
)
from codeinsight.infrastructure.event_log import InMemoryEventLog

TOOLS = (
    {"name": "read_a", "readOnly": True, "inputSchema": {"type": "object"}},
    {"name": "read_b", "readOnly": True, "inputSchema": {"type": "object"}},
)


class Host:
    def __init__(self):
        self.calls: list[str] = []

    def list_tools(self):
        return TOOLS

    def call_tool(self, call):
        self.calls.append(call.name)
        return ToolResult.success(call.id, call.name, {"ok": True})


def test_parallel_submission_and_model_commit_order_are_both_recorded() -> None:
    host = Host()

    class Model:
        count = 0

        def complete_with_tools(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return ToolModelResponse(
                    None,
                    (ToolCall("a", "read_a", {}), ToolCall("b", "read_b", {})),
                    "fake",
                )
            return ToolModelResponse("done", (), "fake")

    log = InMemoryEventLog()
    result = ToolLoop(Model(), host, run_id="run-tools", event_log=log).run("system", "task")
    events = log.read_events("run-tools")

    assert result.status == "COMPLETED"
    assert [event.event_type for event in events[:4]] == [
        TOOL_CALL_REQUESTED,
        TOOL_CALL_VALIDATED,
        TOOL_CALL_REQUESTED,
        TOOL_CALL_VALIDATED,
    ]
    dispatched = [event for event in events if event.event_type == TOOL_DISPATCHED]
    committed = [event for event in events if event.event_type == TOOL_RESULT_COMMITTED]
    assert [event.payload["submission_order"] for event in dispatched] == ["0", "1"]
    assert [event.payload["call_id"] for event in committed] == ["a", "b"]
    assert all(event.payload["execution_mode"] == "parallel" for event in dispatched)


def test_cancelled_tool_gets_an_explicit_aborted_result() -> None:
    cancel = Event()

    class Model:
        def complete_with_tools(self, messages, tools):
            cancel.set()
            return ToolModelResponse(None, (ToolCall("a", "read_a", {}),), "fake")

    host = Host()
    result = ToolLoop(Model(), host, run_id="run-cancel").run(
        "system", "task", cancel_event=cancel
    )

    assert result.status == "ABORTED"
    assert result.tool_results[0].error_code == "ABORTED"
    assert host.calls == []
    assert any(event.event_type == TOOL_ABORTED for event in result.lifecycle_events)


def test_unregistered_tool_is_rejected_before_dispatch() -> None:
    class Model:
        count = 0

        def complete_with_tools(self, messages, tools):
            self.count += 1
            if self.count == 1:
                return ToolModelResponse(
                    None,
                    (ToolCall("unknown", "not_registered", {}),),
                    "fake",
                )
            return ToolModelResponse("done", (), "fake")

    host = Host()
    result = ToolLoop(Model(), host, run_id="run-reject").run("system", "task")

    assert result.status == "COMPLETED"
    assert result.tool_results[0].error_code == "VALIDATION"
    assert host.calls == []
    assert not any(event.event_type == TOOL_DISPATCHED for event in result.lifecycle_events)
    assert any(
        event.event_type == TOOL_CALL_VALIDATED
        and event.payload["outcome"] == "rejected"
        for event in result.lifecycle_events
    )
