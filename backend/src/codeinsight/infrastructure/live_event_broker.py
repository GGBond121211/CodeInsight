"""进程内实时事件广播：把刚落下的事实推给正在看的连接。

它不是事实来源，也不做持久化：序号连续、能不能重放，都由事件 Store 那一层回答。
这里只回答「现在有哪些连接在听这个 run」。

为什么要单独一层：断线重连能补齐的前提是「先落 Store，再推连接」。广播先于落库，
客户端就会先看到一条查不到的事件；只落库不广播，实时性就没了。两件事分开写，
顺序才有地方说清楚。

订阅的原子性也在这里：取回已落事实的事件与登记新订阅必须在同一把锁里完成，
否则「读历史」和「订阅未来」之间到达的事件会被静默丢掉。
"""

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from codeinsight.domain.trace import RunEvent

# 一条连接最多积压多少条事件。慢客户端不该把 Worker 的内存吃光：满了就丢最旧的，
# 客户端重连时从 Store 补齐——丢失是可恢复的，内存涨死不是。
DEFAULT_QUEUE_SIZE = 256

ReplayFn = Callable[[], tuple[RunEvent, ...]]


@dataclass(frozen=True)
class EventSubscription:
    """一条订阅：backlog 是订阅之前已落事实的事件，subscriber 之后的新事件。"""

    run_id: str
    backlog: tuple[RunEvent, ...]
    debug_backlog: tuple[dict[str, object], ...]
    subscriber: queue.Queue[tuple[str, Any]] = field(repr=False)


class LiveEventBroker:
    """run_id → 订阅者集合的广播表。"""

    def __init__(self, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        if queue_size < 1:
            raise ValueError("queue_size 必须为正")
        self._queue_size = queue_size
        self._lock = threading.RLock()
        self._subscribers: dict[str, set[queue.Queue[tuple[str, Any]]]] = {}
        self._debug: dict[str, list[dict[str, object]]] = {}

    def subscribe(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        replay: ReplayFn | None = None,
    ) -> EventSubscription:
        """登记订阅并取回 backlog。

        ``replay`` 由事件日志提供：它在自己的锁里调用 Store，所以「读到哪里」与
        「从哪一条开始收广播」之间不会漏事件。
        """

        if after_sequence < 0:
            raise ValueError("after_sequence 不能为负")
        with self._lock:
            backlog = replay() if replay is not None else ()
            # reasoning 没有公开事件序号：只在首次订阅时回放，续订只接新的，
            # 避免审批恢复时把上一阶段的思考重复展示一遍。
            debug = tuple(self._debug.get(run_id, ())) if after_sequence == 0 else ()
            subscriber: queue.Queue[tuple[str, Any]] = queue.Queue(self._queue_size)
            self._subscribers.setdefault(run_id, set()).add(subscriber)
            return EventSubscription(
                run_id=run_id,
                backlog=tuple(backlog),
                debug_backlog=debug,
                subscriber=subscriber,
            )

    def unsubscribe(self, run_id: str, subscriber: queue.Queue[tuple[str, Any]]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if subscribers is None:
                return
            subscribers.discard(subscriber)
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def subscriber_count(self, run_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(run_id, ()))

    def publish(self, event: RunEvent) -> None:
        """推一条已落事实的事件。推不进去的连接会被丢下最旧的一条，而不是无限积压。"""

        self._push(event.run_id, "event", event)

    def publish_debug_reasoning(self, run_id: str, content: str, *, model: str) -> None:
        """只把供应商显式 reasoning 放进临时队列：它不落盘，断线后可丢。"""

        if not content.strip():
            return
        payload: dict[str, object] = {
            "run_id": run_id,
            "model": model,
            "content": content,
            "occurred_at_epoch_ms": int(time.time() * 1000),
        }
        with self._lock:
            self._debug.setdefault(run_id, []).append(payload)
        self._push(run_id, "debug_reasoning", payload)

    def _push(self, run_id: str, kind: str, payload: object) -> None:
        with self._lock:
            subscribers = tuple(self._subscribers.get(run_id, ()))
        for subscriber in subscribers:
            try:
                subscriber.put_nowait((kind, payload))
            except queue.Full:
                # 慢连接：丢掉最旧的一条，把位置让给新的。断线重连时客户端会
                # 从 Store 补齐，所以这里丢的不是事实，只是一次推送。
                try:
                    subscriber.get_nowait()
                except queue.Empty:
                    pass
                try:
                    subscriber.put_nowait((kind, payload))
                except queue.Full:
                    pass
