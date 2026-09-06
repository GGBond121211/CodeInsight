"""Tool Loop 的可回放生命周期记录器。

生命周期事件只保存调用标识、策略结果、顺序和安全指纹，不保存原始参数、
工具输出或模型隐藏推理。它同时支持本地事件日志和无持久化的单元测试，
因此并行工具的提交顺序也能被验证。
"""

from __future__ import annotations

import threading
import time
from collections.abc import Mapping
from typing import Protocol

from codeinsight.domain.trace import (
    TOOL_ABORTED,
    TOOL_CALL_REQUESTED,
    TOOL_CALL_VALIDATED,
    TOOL_DISPATCHED,
    TOOL_RESULT_COMMITTED,
    RunEvent,
)


class EventLogWriter(Protocol):
    def next_sequence(self, run_id: str) -> int: ...

    def append(self, event: RunEvent) -> None: ...


_LIFECYCLE_EVENT_TYPES = frozenset(
    {
        TOOL_CALL_REQUESTED,
        TOOL_CALL_VALIDATED,
        TOOL_DISPATCHED,
        TOOL_RESULT_COMMITTED,
        TOOL_ABORTED,
    }
)


class ToolLifecycleRecorder:
    """把 Tool Loop 生命周期写入一个 Run 事件流，并保证终态幂等。"""

    def __init__(self, run_id: str, event_log: EventLogWriter | None = None) -> None:
        if not run_id.strip():
            raise ValueError("Tool 生命周期必须绑定 run_id")
        self.run_id = run_id
        self._event_log = event_log
        self._events: list[RunEvent] = []
        self._committed_call_ids: set[str] = set()
        self._sequence = 0
        self._lock = threading.RLock()

    @property
    def events(self) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def record(
        self,
        event_type: str,
        *,
        call_id: str,
        tool_name: str,
        submission_order: int,
        execution_order: int | None = None,
        outcome: str | None = None,
        error_code: str | None = None,
        state_fingerprint: str | None = None,
        terminal: bool = False,
        extra: Mapping[str, str] | None = None,
    ) -> RunEvent | None:
        """记录一条生命周期事件。

        ``terminal=True`` 只允许同一个 ``call_id`` 写入一次。这样即使取消、
        超时和 worker 回调同时到达，也不会产生两个互相矛盾的最终结果。
        """
        if event_type not in _LIFECYCLE_EVENT_TYPES:
            raise ValueError(f"不是 Tool 生命周期事件：{event_type}")
        if not call_id.strip() or not tool_name.strip():
            raise ValueError("Tool 生命周期必须包含 call_id 和 tool_name")
        if submission_order < 0:
            raise ValueError("submission_order 不能为负")
        if execution_order is not None and execution_order < 0:
            raise ValueError("execution_order 不能为负")

        payload: dict[str, str] = {
            "call_id": call_id,
            "tool_name": tool_name,
            "submission_order": str(submission_order),
        }
        if execution_order is not None:
            payload["execution_order"] = str(execution_order)
        if outcome is not None:
            payload["outcome"] = outcome
        if error_code is not None:
            payload["error_code"] = error_code
        if state_fingerprint is not None:
            payload["state_fingerprint"] = state_fingerprint
        if extra:
            payload.update({str(key): str(value) for key, value in extra.items()})

        with self._lock:
            if terminal and call_id in self._committed_call_ids:
                return None
            if self._event_log is not None:
                sequence = self._event_log.next_sequence(self.run_id)
            else:
                sequence = self._sequence + 1
            event = RunEvent(
                event_id=f"{self.run_id}:{sequence}",
                run_id=self.run_id,
                sequence=sequence,
                event_type=event_type,
                occurred_at_epoch_ms=int(time.time() * 1000),
                payload=payload,
            )
            if self._event_log is not None:
                self._event_log.append(event)
            self._events.append(event)
            self._sequence = sequence
            if terminal:
                self._committed_call_ids.add(call_id)
            return event

    def requested(
        self, *, call_id: str, tool_name: str, submission_order: int
    ) -> RunEvent | None:
        return self.record(
            TOOL_CALL_REQUESTED,
            call_id=call_id,
            tool_name=tool_name,
            submission_order=submission_order,
            outcome="requested",
        )

    def validated(
        self,
        *,
        call_id: str,
        tool_name: str,
        submission_order: int,
        allowed: bool,
    ) -> RunEvent | None:
        return self.record(
            TOOL_CALL_VALIDATED,
            call_id=call_id,
            tool_name=tool_name,
            submission_order=submission_order,
            outcome="allowed" if allowed else "rejected",
            extra={"policy": "fail_closed", "validation_scope": "registry"},
        )

    def dispatched(
        self,
        *,
        call_id: str,
        tool_name: str,
        submission_order: int,
        execution_order: int,
        parallel: bool,
    ) -> RunEvent | None:
        return self.record(
            TOOL_DISPATCHED,
            call_id=call_id,
            tool_name=tool_name,
            submission_order=submission_order,
            execution_order=execution_order,
            outcome="dispatched",
            extra={"execution_mode": "parallel" if parallel else "serial"},
        )

    def committed(
        self,
        *,
        call_id: str,
        tool_name: str,
        submission_order: int,
        execution_order: int | None,
        ok: bool,
        error_code: str | None,
        state_fingerprint: str | None,
    ) -> RunEvent | None:
        return self.record(
            TOOL_RESULT_COMMITTED,
            call_id=call_id,
            tool_name=tool_name,
            submission_order=submission_order,
            execution_order=execution_order,
            outcome="success" if ok else "error",
            error_code=error_code,
            state_fingerprint=state_fingerprint,
            terminal=True,
        )

    def aborted(
        self,
        *,
        call_id: str,
        tool_name: str,
        submission_order: int,
        reason: str = "cancelled",
    ) -> RunEvent | None:
        return self.record(
            TOOL_ABORTED,
            call_id=call_id,
            tool_name=tool_name,
            submission_order=submission_order,
            outcome="aborted",
            error_code="ABORTED",
            terminal=True,
            extra={"reason": reason},
        )
