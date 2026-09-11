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
from codeinsight.application.tool_result_pruner import (
    PruneOutcome,
    ToolResultPruner,
)
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


# 控制层允许使用的终止状态；COMPLETED 只能由模型正常收尾产生。
LOOP_TERMINAL_STATUSES: frozenset[str] = frozenset({"STUCK", "FAILED", "ABORTED"})


@dataclass(frozen=True)
class LoopIntervention:
    """控制层在循环中途注入的一次干预。

    ``message`` 会作为一条 user 消息追加到对话里，让模型带着新信息继续；
    ``terminate_reason`` 非空则立即结束循环。两者至少要有一个。
    """

    message: str | None = None
    terminate_reason: str | None = None
    status: str = "STUCK"

    def __post_init__(self) -> None:
        if self.message is None and self.terminate_reason is None:
            raise ValueError("干预必须给出 message 或 terminate_reason")
        if self.terminate_reason is not None and self.status not in LOOP_TERMINAL_STATUSES:
            raise ValueError(f"不支持的循环终止状态：{self.status}")


class ToolOutcomeController(Protocol):
    """夹在 ToolResult 和下一次模型调用之间的控制层。

    它只负责「还能不能继续、要不要补充检索」，不执行工具、不修改权限。
    """

    def after_tools(
        self,
        *,
        step: int,
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> LoopIntervention | None: ...

    def after_final_answer(
        self, *, step: int, content: str | None
    ) -> LoopIntervention | None: ...


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
    # 每轮模型调用前按工具类型裁剪旧工具结果。默认开启：不裁剪的话消息会
    # 一直单调增长，直到某次调用直接撞上上游上下文上限。
    prune_tool_results: bool = True
    keep_recent_tool_results: int = 2

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("Tool Loop 的步数和工具调用预算必须为正")
        if self.deadline_seconds <= 0:
            raise ValueError("deadline_seconds 必须为正")
        if self.keep_recent_tool_results < 0:
            raise ValueError("keep_recent_tool_results 不能为负")


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
    # 每轮实际发生的裁剪；用于回答「模型这一轮到底看到了什么」。
    prune_events: tuple[PruneOutcome, ...] = ()


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
        outcome_controller: ToolOutcomeController | None = None,
        result_pruner: ToolResultPruner | None = None,
    ) -> None:
        self._model = model
        self._mcp = mcp_client
        self._config = config or ToolLoopConfig()
        self._lifecycle = ToolLifecycleRecorder(run_id, event_log)
        # 不传控制层时行为与之前完全一致：循环只在模型的 tool_calls 上推进。
        self._controller = outcome_controller
        if result_pruner is not None:
            self._pruner: ToolResultPruner | None = result_pruner
        elif self._config.prune_tool_results:
            self._pruner = ToolResultPruner(
                keep_recent_results=self._config.keep_recent_tool_results
            )
        else:
            self._pruner = None
        self._prune_events: list[PruneOutcome] = []

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
        self._prune_events = []

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
                response = self._model.complete_with_tools(
                    self._visible_messages(messages), tools
                )
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
                intervention = self._intervene_final(step, response.content)
                if intervention is not None:
                    if intervention.terminate_reason is not None:
                        return self._result(
                            intervention.status,
                            None,
                            messages,
                            calls,
                            results,
                            step,
                            input_tokens,
                            output_tokens,
                            intervention.terminate_reason,
                        )
                    if intervention.message:
                        messages.append({"role": "user", "content": intervention.message})
                        continue
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
            intervention = self._intervene_tools(step, response.tool_calls, batch)
            if intervention is not None:
                if intervention.terminate_reason is not None:
                    return self._result(
                        intervention.status,
                        None,
                        messages,
                        calls,
                        results,
                        step,
                        input_tokens,
                        output_tokens,
                        intervention.terminate_reason,
                    )
                if intervention.message:
                    messages.append({"role": "user", "content": intervention.message})
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

    def _visible_messages(
        self, messages: list[Mapping[str, object]]
    ) -> tuple[Mapping[str, object], ...]:
        """本轮发给模型的消息视图。

        裁剪只作用于这一次调用，``messages`` 本身仍是完整记录：审计和回放要
        看的是模型真实收到过什么，而不是被裁剪后的副本。
        """
        if self._pruner is None:
            return tuple(messages)
        outcome = self._pruner.prune(messages)
        if outcome.pruned:
            self._prune_events.append(outcome)
        return outcome.messages

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

    def _intervene_tools(
        self,
        step: int,
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> LoopIntervention | None:
        """把一批 ToolResult 交给控制层；控制层异常不得变成静默放行。"""
        if self._controller is None:
            return None
        try:
            return self._controller.after_tools(
                step=step, calls=tuple(calls), results=tuple(results)
            )
        except Exception:
            return LoopIntervention(
                terminate_reason="控制层回调失败，按 fail-closed 终止",
                status="FAILED",
            )

    def _intervene_final(self, step: int, content: str | None) -> LoopIntervention | None:
        """模型想收尾时先问控制层；控制层异常同样 fail-closed。"""
        if self._controller is None:
            return None
        try:
            return self._controller.after_final_answer(step=step, content=content)
        except Exception:
            return LoopIntervention(
                terminate_reason="控制层回调失败，按 fail-closed 终止",
                status="FAILED",
            )

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
            tuple(self._prune_events),
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
