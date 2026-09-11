"""统一对话的进程内异步运行时与实时事件总线。

本地调试阶段使用线程池让 HTTP 接口快速返回 accepted，浏览器通过 SSE 读取
阶段事件。事件日志仍然只保存公开状态；供应商显式返回的 reasoning 只留在
进程内的临时队列中，断线后可丢失，也不会写入磁盘或数据库。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any
from uuid import uuid4

from codeinsight.domain.chat import (
    CHAT_CANCELLED,
    CHAT_FAILED,
    CHAT_QUEUED,
    CHAT_RUNNING,
    CHAT_WAITING_APPROVAL,
    TERMINAL_CHAT_STATUSES,
    ChatTurn,
)
from codeinsight.domain.trace import (
    RUN_FINISHED,
    RUN_STARTED,
    STATE_TRANSITIONED,
    TURN_ACCEPTED,
    RunEvent,
)
from codeinsight.infrastructure.event_log import InMemoryEventLog


class ChatRuntimeError(ValueError):
    """聊天 Run 不满足当前运行时状态。"""


@dataclass(frozen=True)
class ChatExecution:
    status: str
    assistant_message: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None


class LiveEventLog:
    """在 InMemoryEventLog 之上增加 SSE 订阅，不改变事件日志契约。"""

    def __init__(self) -> None:
        self._events = InMemoryEventLog()
        self._lock = threading.RLock()
        self._subscribers: dict[str, set[queue.Queue[tuple[str, Any]]]] = {}
        self._debug_backlog: dict[str, list[dict[str, object]]] = {}

    def append(self, event: RunEvent) -> None:
        with self._lock:
            self._events.append(event)
            for subscriber in tuple(self._subscribers.get(event.run_id, ())):
                subscriber.put(("event", event))

    def next_sequence(self, run_id: str) -> int:
        with self._lock:
            return self._events.next_sequence(run_id)

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return self._events.read_events(run_id, after_sequence=after_sequence)

    def subscribe(
        self, run_id: str, *, after_sequence: int = 0
    ) -> tuple[tuple[RunEvent, ...], tuple[dict[str, object], ...], queue.Queue]:
        """原子地取得 backlog 并建立订阅，避免 SSE 建连竞态丢事件。"""
        if after_sequence < 0:
            raise ValueError("after_sequence 不能为负")
        with self._lock:
            backlog = self._events.read_events(run_id, after_sequence=after_sequence)
            # reasoning 没有公开 RunEvent sequence：只在首次订阅时尽力回放，
            # 续订只接收新到达的临时内容，避免审批恢复时重复展示上一阶段 reasoning。
            debug = tuple(self._debug_backlog.get(run_id, ())) if after_sequence == 0 else ()
            subscriber: queue.Queue[tuple[str, Any]] = queue.Queue()
            self._subscribers.setdefault(run_id, set()).add(subscriber)
            return backlog, debug, subscriber

    def unsubscribe(self, run_id: str, subscriber: queue.Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def publish_debug_reasoning(self, run_id: str, content: str, *, model: str) -> None:
        """只把供应商显式 reasoning 放进临时内存 SSE 队列。"""
        if not content.strip():
            return
        payload = {
            "run_id": run_id,
            "model": model,
            "content": content,
            "occurred_at_epoch_ms": int(time.time() * 1000),
        }
        with self._lock:
            self._debug_backlog.setdefault(run_id, []).append(payload)
            for subscriber in tuple(self._subscribers.get(run_id, ())):
                subscriber.put(("debug_reasoning", payload))


class ChatRuntime:
    """一次进程内服务实例共享的后台聊天运行时。"""

    def __init__(self, *, max_workers: int = 4) -> None:
        if max_workers < 1:
            raise ValueError("max_workers 必须为正")
        self.event_log = LiveEventLog()
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="codeinsight-chat"
        )
        self._lock = threading.RLock()
        self._turns: dict[str, ChatTurn] = {}
        self._active_by_session: dict[str, str] = {}
        self._running: set[str] = set()

    def submit(
        self,
        *,
        session_id: str,
        task_type: str,
        user_message: str,
        show_debug_reasoning: bool,
        worker: Callable[[ChatTurn, bool], ChatExecution],
        turn_id: str | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
    ) -> ChatTurn:
        now = int(time.time() * 1000)
        turn = ChatTurn(
            turn_id=turn_id or f"turn-{uuid4().hex[:16]}",
            session_id=session_id,
            run_id=run_id or f"chat-run-{uuid4().hex[:16]}",
            task_type=task_type,
            status=CHAT_QUEUED,
            user_message=user_message,
            created_at_epoch_ms=now,
            updated_at_epoch_ms=now,
            task_id=task_id,
        )
        with self._lock:
            if session_id in self._active_by_session:
                raise ChatRuntimeError("同一会话上一轮仍在执行或等待审批")
            self._turns[turn.turn_id] = turn
            self._active_by_session[session_id] = turn.turn_id
        self.emit(
            turn.run_id,
            RUN_STARTED,
            {"turn_id": turn.turn_id, "task_type": task_type, "status": CHAT_QUEUED},
        )
        self.emit(
            turn.run_id,
            TURN_ACCEPTED,
            {"turn_id": turn.turn_id, "task_type": task_type, "status": CHAT_QUEUED},
        )
        self._executor.submit(self._execute, turn.turn_id, worker, show_debug_reasoning)
        return turn

    def resume(
        self,
        turn_id: str,
        *,
        show_debug_reasoning: bool,
        worker: Callable[[ChatTurn, bool], ChatExecution],
    ) -> ChatTurn:
        with self._lock:
            turn = self._require(turn_id)
            if turn.status != CHAT_WAITING_APPROVAL:
                raise ChatRuntimeError("当前聊天 Run 不在等待审批状态")
            if turn_id in self._running:
                raise ChatRuntimeError("当前聊天 Run 已经恢复执行")
            turn = replace(
                turn,
                status=CHAT_QUEUED,
                updated_at_epoch_ms=int(time.time() * 1000),
            )
            self._turns[turn_id] = turn
            self._running.add(turn_id)
        self.emit(
            turn.run_id,
            STATE_TRANSITIONED,
            {"from_status": CHAT_WAITING_APPROVAL, "to_status": CHAT_QUEUED},
        )
        self._executor.submit(self._execute_existing, turn_id, worker, show_debug_reasoning)
        return turn

    def get_turn(self, turn_id: str) -> ChatTurn:
        with self._lock:
            return self._require(turn_id)

    def get_turn_or_none(self, turn_id: str) -> ChatTurn | None:
        """本进程见过这一轮就返回它，没见过就返回 None。

        受理与执行分到不同进程之后，「本地没有」不再等于「不存在」。
        调用方需要能区分这两种情况，再决定要不要去事实 Store 里读。
        """

        with self._lock:
            return self._turns.get(turn_id)

    def get_turn_by_run_id(self, run_id: str) -> ChatTurn:
        with self._lock:
            for turn in self._turns.values():
                if turn.run_id == run_id:
                    return turn
        raise ChatRuntimeError("聊天 Run 不存在")

    def active_turn_for_session(self, session_id: str) -> ChatTurn | None:
        with self._lock:
            turn_id = self._active_by_session.get(session_id)
            return self._turns.get(turn_id) if turn_id else None

    def cancel_waiting(self, turn_id: str) -> ChatTurn:
        with self._lock:
            turn = self._require(turn_id)
            if turn.status in TERMINAL_CHAT_STATUSES:
                return turn
            if turn.status != "WAITING_APPROVAL":
                raise ChatRuntimeError("运行中的聊天任务暂不支持强制终止")
            updated = replace(
                turn,
                status=CHAT_CANCELLED,
                updated_at_epoch_ms=int(time.time() * 1000),
            )
            self._turns[turn_id] = updated
            self._active_by_session.pop(turn.session_id, None)
        self.emit(updated.run_id, "cancel_requested", {"reason": "user_request"})
        self.emit(
            updated.run_id,
            STATE_TRANSITIONED,
            {"from_status": turn.status, "to_status": CHAT_CANCELLED},
        )
        self.emit(updated.run_id, RUN_FINISHED, {"status": CHAT_CANCELLED})
        return updated

    def mark_reasoning_available(self, turn_id: str) -> None:
        with self._lock:
            turn = self._require(turn_id)
            self._turns[turn_id] = replace(
                turn, reasoning_available=True, updated_at_epoch_ms=int(time.time() * 1000)
            )

    def publish_reasoning(self, turn_id: str, content: str, *, model: str) -> None:
        turn = self.get_turn(turn_id)
        self.mark_reasoning_available(turn_id)
        self.event_log.publish_debug_reasoning(turn.run_id, content, model=model)

    def emit(self, run_id: str, event_type: str, payload: dict[str, str]) -> RunEvent:
        sequence = self.event_log.next_sequence(run_id)
        event = RunEvent(
            event_id=f"{run_id}:{sequence}",
            run_id=run_id,
            sequence=sequence,
            event_type=event_type,
            occurred_at_epoch_ms=int(time.time() * 1000),
            payload=payload,
        )
        self.event_log.append(event)
        return event

    def subscribe(self, run_id: str, *, after_sequence: int = 0):
        return self.event_log.subscribe(run_id, after_sequence=after_sequence)

    def unsubscribe(self, run_id: str, subscriber) -> None:
        self.event_log.unsubscribe(run_id, subscriber)

    def is_terminal(self, run_id: str) -> bool:
        with self._lock:
            for turn in self._turns.values():
                if turn.run_id == run_id:
                    return turn.status in TERMINAL_CHAT_STATUSES
        return False

    def _execute(
        self,
        turn_id: str,
        worker: Callable[[ChatTurn, bool], ChatExecution],
        show_debug_reasoning: bool,
    ) -> None:
        with self._lock:
            self._running.add(turn_id)
        self._execute_existing(turn_id, worker, show_debug_reasoning)

    def _execute_existing(
        self,
        turn_id: str,
        worker: Callable[[ChatTurn, bool], ChatExecution],
        show_debug_reasoning: bool,
    ) -> None:
        try:
            with self._lock:
                turn = self._require(turn_id)
                previous = turn.status
                running = replace(
                    turn,
                    status=CHAT_RUNNING,
                    updated_at_epoch_ms=int(time.time() * 1000),
                )
                self._turns[turn_id] = running
            self.emit(
                running.run_id,
                STATE_TRANSITIONED,
                {"from_status": previous, "to_status": CHAT_RUNNING},
            )
            execution = worker(running, show_debug_reasoning)
        except Exception as error:  # pragma: no cover - defensive worker boundary
            execution = ChatExecution(
                status=CHAT_FAILED,
                assistant_message="本轮执行失败，请查看事件详情。",
                error=f"执行失败：{type(error).__name__}",
            )
        with self._lock:
            current = self._require(turn_id)
            updated = replace(
                current,
                status=execution.status,
                assistant_message=execution.assistant_message,
                result=execution.result,
                error=execution.error,
                updated_at_epoch_ms=int(time.time() * 1000),
            )
            self._turns[turn_id] = updated
            self._running.discard(turn_id)
            if execution.status in TERMINAL_CHAT_STATUSES:
                self._active_by_session.pop(updated.session_id, None)
        self.emit(
            updated.run_id,
            STATE_TRANSITIONED,
            {"from_status": CHAT_RUNNING, "to_status": execution.status},
        )
        if execution.status != CHAT_WAITING_APPROVAL:
            self.emit(updated.run_id, RUN_FINISHED, {"status": execution.status})

    def _require(self, turn_id: str) -> ChatTurn:
        try:
            return self._turns[turn_id]
        except KeyError as error:
            raise ChatRuntimeError("聊天 Run 不存在") from error
