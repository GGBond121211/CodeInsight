"""模型原生 Tool Calling 的有界执行循环。

这个模块只负责三件事：把供应商响应归一化、把工具结果重新交给模型、
以及在循环外加上预算和防失控闸门。工具本身必须通过 MCP Client 调用，
不能在这里偷偷 import 某个具体工具函数。
"""

from __future__ import annotations

import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event
from typing import Protocol

from codeinsight.agent.tool_lifecycle import ToolLifecycleRecorder
from codeinsight.domain.trace import RunEvent

TOOL_ERROR_CODES = frozenset(
    {
        "VALIDATION",
        "PERMISSION",
        "NOT_FOUND",
        "TIMEOUT",
        "UPSTREAM_5XX",
        "UNKNOWN",
        "DB_CONFLICT",
        "ABORTED",
        "NOT_CONFIGURED",
        "QDRANT_NOT_CONFIGURED",
        "QDRANT_UNAVAILABLE",
        "INDEX_NOT_FOUND",
        "INDEX_STALE",
        "UNSUPPORTED_LANGUAGE",
        "INVALID_RESPONSE",
    }
)


@dataclass(frozen=True)
class ToolCall:
    """供应商原生 tool_call 的稳定内部表示。"""

    id: str
    name: str
    arguments: dict[str, object]

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.name.strip():
            raise ValueError("ToolCall 的 id 和 name 不能为空")

    @property
    def signature(self) -> str:
        return f"{self.name}:{json.dumps(self.arguments, sort_keys=True, ensure_ascii=False)}"


@dataclass(frozen=True)
class ToolResult:
    """工具成功或失败的统一返回值。

    ``data`` 会进入下一轮模型上下文，因而始终被当作不可信工具输出；
    ``state_fingerprint`` 只用于循环的状态变化检测，不承载秘密或完整文件内容。
    """

    call_id: str
    tool_name: str
    ok: bool
    data: Mapping[str, object] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    state_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.call_id.strip() or not self.tool_name.strip():
            raise ValueError("ToolResult 的 call_id 和 tool_name 不能为空")
        if self.ok and (self.error_code or self.error_message):
            raise ValueError("成功的 ToolResult 不能同时携带错误")
        if not self.ok:
            if self.error_code not in TOOL_ERROR_CODES:
                raise ValueError(f"不支持的工具错误类型：{self.error_code}")
            if not self.error_message:
                raise ValueError("失败的 ToolResult 必须说明 error_message")

    @classmethod
    def success(
        cls,
        call_id: str,
        tool_name: str,
        data: Mapping[str, object],
        *,
        state_fingerprint: str | None = None,
    ) -> ToolResult:
        return cls(call_id, tool_name, True, dict(data), state_fingerprint=state_fingerprint)

    @classmethod
    def failure(
        cls,
        call_id: str,
        tool_name: str,
        error_code: str,
        error_message: str,
        *,
        data: Mapping[str, object] | None = None,
        state_fingerprint: str | None = None,
    ) -> ToolResult:
        return cls(
            call_id,
            tool_name,
            False,
            dict(data or {}),
            error_code,
            error_message,
            state_fingerprint,
        )

    def as_model_content(self) -> str:
        payload: dict[str, object] = {
            "ok": self.ok,
            "tool": self.tool_name,
            "data": dict(self.data),
        }
        if self.state_fingerprint is not None:
            payload["state_fingerprint"] = self.state_fingerprint
        if not self.ok:
            payload["error"] = {"code": self.error_code, "message": self.error_message}
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class ToolModelResponse:
    """模型一轮响应；tool_calls 来自协议字段，不从普通文本猜测。"""

    content: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    # 不进入 ToolLoop 消息或事件日志，仅由调试实时通道临时消费。
    reasoning_content: str | None = None


class ToolModel(Protocol):
    def complete_with_tools(
        self, messages: Sequence[Mapping[str, object]], tools: Sequence[Mapping[str, object]]
    ) -> ToolModelResponse: ...


class MCPToolClient(Protocol):
    def list_tools(self) -> tuple[Mapping[str, object], ...]: ...

    def call_tool(self, call: ToolCall) -> ToolResult: ...


@dataclass(frozen=True)
class ToolLoopConfig:
    max_steps: int = 8
    deadline_seconds: float = 60.0
    # Read-only repository exploration can legitimately require more than one
    # batch before the model has enough evidence to emit generate_patch. Keep
    # this finite so a malformed tool plan still fails closed.
    max_tool_calls: int = 64
    token_budget: int | None = None
    repeated_call_limit: int = 2
    repeated_error_limit: int = 2

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("Tool Loop 的步数和工具调用预算必须为正")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds 必须为正")


@dataclass(frozen=True)
class ToolLoopResult:
    status: str
    final_content: str | None
    messages: tuple[Mapping[str, object], ...]
    tool_calls: tuple[ToolCall, ...]
    tool_results: tuple[ToolResult, ...]
    steps: int
    input_tokens: int
    output_tokens: int
    reason: str | None = None
    lifecycle_events: tuple[RunEvent, ...] = ()


class ToolLoop:
    """执行一个有界、可审计但不保存隐藏推理的工具循环。"""

    def __init__(
        self,
        model: ToolModel,
        mcp_client: MCPToolClient,
        *,
        config: ToolLoopConfig | None = None,
        run_id: str = "local-run",
        event_log=None,
    ) -> None:
        self._model = model
        self._mcp = mcp_client
        self._config = config or ToolLoopConfig()
        self._lifecycle = ToolLifecycleRecorder(run_id, event_log)

    def run(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        cancel_event: Event | None = None,
    ) -> ToolLoopResult:
        messages: list[Mapping[str, object]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        discovered = self._mcp.list_tools()
        tools = tuple(_as_openai_tool(item) for item in discovered)
        calls: list[ToolCall] = []
        results: list[ToolResult] = []
        call_counts: Counter[str] = Counter()
        error_counts: Counter[tuple[str, str, str]] = Counter()
        previous_state: str | None = None
        unchanged_write_streak = 0
        input_tokens = 0
        output_tokens = 0
        start = time.monotonic()

        for step in range(1, self._config.max_steps + 1):
            if cancel_event is not None and cancel_event.is_set():
                return self._result(
                    "ABORTED",
                    None,
                    messages,
                    calls,
                    results,
                    step - 1,
                    input_tokens,
                    output_tokens,
                    "Run 已取消，未开始下一次模型调用",
                )
            if time.monotonic() - start >= self._config.deadline_seconds:
                return self._stuck(
                    messages,
                    calls,
                    results,
                    step - 1,
                    input_tokens,
                    output_tokens,
                    "总 deadline 已到",
                )
            try:
                response = self._model.complete_with_tools(tuple(messages), tools)
            except Exception:
                return self._result(
                    "FAILED",
                    None,
                    messages,
                    calls,
                    results,
                    step,
                    input_tokens,
                    output_tokens,
                    "模型 Provider 调用失败",
                )
            input_tokens += response.input_tokens or 0
            output_tokens += response.output_tokens or 0
            if (
                self._config.token_budget is not None
                and input_tokens + output_tokens > self._config.token_budget
            ):
                return self._stuck(
                    messages, calls, results, step, input_tokens, output_tokens, "Token 预算已用尽"
                )
            assistant_message: dict[str, object] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant_message["tool_calls"] = tuple(
                    _as_native_call(item) for item in response.tool_calls
                )
            messages.append(assistant_message)
            if not response.tool_calls:
                return self._result(
                    "COMPLETED",
                    response.content,
                    messages,
                    calls,
                    results,
                    step,
                    input_tokens,
                    output_tokens,
                    )

            submission_orders = {
                index: len(calls) + index for index in range(len(response.tool_calls))
            }
            for index, call in enumerate(response.tool_calls):
                self._lifecycle.requested(
                    call_id=call.id,
                    tool_name=call.name,
                    submission_order=submission_orders[index],
                )
                self._lifecycle.validated(
                    call_id=call.id,
                    tool_name=call.name,
                    submission_order=submission_orders[index],
                    allowed=_find_tool(discovered, call.name) is not None,
                )

            if len(calls) + len(response.tool_calls) > self._config.max_tool_calls:
                self._abort_unexecuted_calls(
                    response.tool_calls,
                    submission_orders,
                    reason="tool_call_budget",
                )
                return self._stuck(
                    messages,
                    calls,
                    results,
                    step,
                    input_tokens,
                    output_tokens,
                    "工具调用预算已用尽",
                )
            for call in response.tool_calls:
                call_counts[call.signature] += 1
                if call_counts[call.signature] > self._config.repeated_call_limit:
                    self._abort_unexecuted_calls(
                        response.tool_calls,
                        submission_orders,
                        reason="repeated_call_limit",
                    )
                    return self._stuck(
                        messages,
                        calls,
                        results,
                        step,
                        input_tokens,
                        output_tokens,
                        f"重复工具调用达到上限：{call.name}",
                    )

            batch, execution_orders = self._execute_batch(
                response.tool_calls,
                discovered,
                submission_orders=submission_orders,
                cancel_event=cancel_event,
            )
            for index, (call, result) in enumerate(zip(response.tool_calls, batch, strict=True)):
                calls.append(call)
                results.append(result)
                self._lifecycle.committed(
                    call_id=call.id,
                    tool_name=call.name,
                    submission_order=submission_orders[index],
                    execution_order=execution_orders[index],
                    ok=result.ok,
                    error_code=result.error_code,
                    state_fingerprint=result.state_fingerprint,
                )
                if not result.ok:
                    key = (call.name, result.error_code or "UNKNOWN", result.error_message or "")
                    error_counts[key] += 1
                    if error_counts[key] >= self._config.repeated_error_limit:
                        return self._stuck(
                            messages,
                            calls,
                            results,
                            step,
                            input_tokens,
                            output_tokens,
                            f"重复工具错误达到上限：{call.name}/{result.error_code}",
                        )
                spec = _find_tool(discovered, call.name)
                if spec and not bool(spec.get("readOnly", True)):
                    if result.state_fingerprint and result.state_fingerprint == previous_state:
                        unchanged_write_streak += 1
                    else:
                        unchanged_write_streak = 0
                    previous_state = result.state_fingerprint
            for result in batch:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": result.call_id,
                        "content": result.as_model_content(),
                    }
                )
            if unchanged_write_streak >= 2:
                return self._stuck(
                    messages,
                    calls,
                    results,
                    step,
                    input_tokens,
                    output_tokens,
                    "副作用工具连续执行但状态没有变化",
                )
        return self._stuck(
            messages,
            calls,
            results,
            self._config.max_steps,
            input_tokens,
            output_tokens,
            "达到最大步骤数",
        )

    def _execute_batch(
        self,
        calls: Sequence[ToolCall],
        discovered: Sequence[Mapping[str, object]],
        *,
        submission_orders: Mapping[int, int],
        cancel_event: Event | None,
    ) -> tuple[tuple[ToolResult, ...], tuple[int | None, ...]]:
        readonly: list[tuple[int, ToolCall]] = []
        results: list[ToolResult | None] = [None] * len(calls)
        execution_orders: list[int | None] = [None] * len(calls)
        execution_order = 0
        for index, call in enumerate(calls):
            spec = _find_tool(discovered, call.name)
            if spec is None:
                results[index] = ToolResult.failure(
                    call.id,
                    call.name,
                    "VALIDATION",
                    "工具未在 MCP 目录登记；执行层按 fail-closed 拒绝",
                )
                continue
            if cancel_event is not None and cancel_event.is_set():
                results[index] = ToolResult.failure(
                    call.id,
                    call.name,
                    "ABORTED",
                    "Run 已取消，工具尚未开始执行",
                )
                self._lifecycle.aborted(
                    call_id=call.id,
                    tool_name=call.name,
                    submission_order=submission_orders[index],
                    reason="cancelled_before_dispatch",
                )
                continue
            execution_orders[index] = execution_order
            execution_order += 1
            parallel = bool(spec.get("readOnly", False))
            self._lifecycle.dispatched(
                call_id=call.id,
                tool_name=call.name,
                submission_order=submission_orders[index],
                execution_order=execution_orders[index],
                parallel=parallel,
            )
            if parallel:
                readonly.append((index, call))
            else:
                # 有副作用工具必须串行；工具 Client 仍是唯一执行入口。
                results[index] = self._safe_call_tool(call)
        if readonly:
            with ThreadPoolExecutor(max_workers=len(readonly)) as pool:
                futures = [
                    (index, pool.submit(self._safe_call_tool, call)) for index, call in readonly
                ]
                for index, future in futures:
                    results[index] = future.result()
        return tuple(item for item in results if item is not None), tuple(execution_orders)

    def _safe_call_tool(self, call: ToolCall) -> ToolResult:
        try:
            return self._mcp.call_tool(call)
        except Exception:
            return ToolResult.failure(call.id, call.name, "UNKNOWN", "工具执行失败")

    def _abort_unexecuted_calls(
        self,
        calls: Sequence[ToolCall],
        submission_orders: Mapping[int, int],
        *,
        reason: str,
    ) -> None:
        for index, call in enumerate(calls):
            self._lifecycle.aborted(
                call_id=call.id,
                tool_name=call.name,
                submission_order=submission_orders[index],
                reason=reason,
            )

    def _result(
        self,
        status: str,
        final_content: str | None,
        messages: list[Mapping[str, object]],
        calls: list[ToolCall],
        results: list[ToolResult],
        steps: int,
        input_tokens: int,
        output_tokens: int,
        reason: str | None = None,
    ) -> ToolLoopResult:
        return ToolLoopResult(
            status,
            final_content,
            tuple(messages),
            tuple(calls),
            tuple(results),
            steps,
            input_tokens,
            output_tokens,
            reason,
            self._lifecycle.events,
        )

    def _stuck(
        self,
        messages: list[Mapping[str, object]],
        calls: list[ToolCall],
        results: list[ToolResult],
        steps: int,
        input_tokens: int,
        output_tokens: int,
        reason: str,
    ) -> ToolLoopResult:
        return self._result(
            "STUCK",
            None,
            messages,
            calls,
            results,
            steps,
            input_tokens,
            output_tokens,
            reason,
        )


def _find_tool(tools: Sequence[Mapping[str, object]], name: str) -> Mapping[str, object] | None:
    for item in tools:
        if item.get("name") == name:
            return item
    return None


def _as_openai_tool(tool: Mapping[str, object]) -> Mapping[str, object]:
    """MCP tools/list 的 inputSchema 转为 Chat Completions tools 形状。"""
    return {
        "type": "function",
        "function": {
            "name": str(tool["name"]),
            "description": str(tool.get("description", "")),
            "parameters": dict(tool.get("inputSchema", {"type": "object"})),
        },
    }


def _as_native_call(call: ToolCall) -> Mapping[str, object]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": json.dumps(call.arguments, ensure_ascii=False),
        },
    }
