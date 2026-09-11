"""事件日志与审计流的内存实现。

两者刻意用两个类、两块存储，而不是一张带 ``is_audit`` 列的表——
理由见 ``domain/trace.py`` 的模块 docstring：保留期与采样策略不同，
混在一起迟早会因为「清理旧事件」把审计记录一起删掉。
"""

from __future__ import annotations

import threading

from codeinsight.domain.ports import EventLog
from codeinsight.domain.trace import AuditRecord, RunEvent
from codeinsight.infrastructure.live_event_broker import EventSubscription, LiveEventBroker


class EventSequenceError(Exception):
    """事件序号不连续或重复。

    序号有洞时回放结果不可信，SSE 的 ``Last-Event-ID`` 续传也会错位。
    这类错误必须显式失败，不能容忍。
    """


class InMemoryEventLog:
    """按 run 分组的追加型事件日志。

    保证同一个 run 内 ``sequence`` 从 1 开始、连续、唯一。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, list[RunEvent]] = {}

    def append(self, event: RunEvent) -> None:
        with self._lock:
            existing = self._events.get(event.run_id)
            if existing is None:
                existing = []
                self._events[event.run_id] = existing
            expected = len(existing) + 1
            if event.sequence != expected:
                raise EventSequenceError(
                    f"run {event.run_id} 的下一个事件序号应为 {expected}，"
                    f"实际收到 {event.sequence}。"
                    "序号必须连续——有洞会让回放和断线续传都不可信。"
                )
            existing.append(event)

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        """读取事件。``after_sequence`` 对应 SSE 重连时客户端回传的位置。"""
        if after_sequence < 0:
            raise ValueError("after_sequence 不能为负")
        with self._lock:
            existing = self._events.get(run_id)
            if existing is None:
                return ()
            found: list[RunEvent] = []
            for event in existing:
                if event.sequence > after_sequence:
                    found.append(event)
            return tuple(found)

    def next_sequence(self, run_id: str) -> int:
        with self._lock:
            existing = self._events.get(run_id)
            if existing is None:
                return 1
            return len(existing) + 1

    def count(self, run_id: str) -> int:
        with self._lock:
            existing = self._events.get(run_id)
            if existing is None:
                return 0
            return len(existing)


class InMemoryAuditLog:
    """独立审计流的内存实现。

    刻意**不提供删除或清理方法**。审计记录的价值在于它一直在；
    提供了清理接口，早晚会有一段「顺手清理过期数据」的代码调用它。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, list[AuditRecord]] = {}

    def record(self, entry: AuditRecord) -> None:
        with self._lock:
            existing = self._records.get(entry.run_id)
            if existing is None:
                existing = []
                self._records[entry.run_id] = existing
            for present in existing:
                if present.audit_id == entry.audit_id:
                    raise ValueError(f"审计记录 {entry.audit_id} 已存在，不能重复写入")
            existing.append(entry)

    def read_records(self, run_id: str) -> tuple[AuditRecord, ...]:
        with self._lock:
            existing = self._records.get(run_id)
            if existing is None:
                return ()
            found: list[AuditRecord] = []
            for entry in existing:
                found.append(entry)
            return tuple(found)

    def total_records(self) -> int:
        with self._lock:
            total = 0
            for entries in self._records.values():
                total += len(entries)
            return total
class LiveEventLog:
    """在事件 Store 之上增加实时订阅。

    写入顺序固定为「先落 Store，再广播」：连接的 backlog 来自 Store，实时推送来自
    广播。两步在同一把锁里串起来，所以事件不会出现「推得出去但查不到」，也不会在
    「读完历史」和「订阅未来」之间被丢掉。

    默认的 Store 是进程内存：单进程本地开发够用，但重启即失忆。要跨进程或跨重启
    回放，就在装配时换成真表（见 ``default_event_log``）。
    """

    def __init__(
        self,
        store: EventLog | None = None,
        *,
        broker: LiveEventBroker | None = None,
    ) -> None:
        self._events = store if store is not None else InMemoryEventLog()
        self._broker = broker or LiveEventBroker()
        self._lock = threading.RLock()

    @property
    def store(self) -> EventLog:
        return self._events

    def append(self, event: RunEvent) -> None:
        with self._lock:
            self._events.append(event)
            self._broker.publish(event)

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        with self._lock:
            return self._events.read_events(run_id, after_sequence=after_sequence)

    def next_sequence(self, run_id: str) -> int:
        with self._lock:
            return self._events.next_sequence(run_id)

    def count(self, run_id: str) -> int:
        with self._lock:
            attributes = getattr(self._events, "count", None)
            if not callable(attributes):
                return len(self._events.read_events(run_id))
            return attributes(run_id)

    def subscribe(self, run_id: str, *, after_sequence: int = 0) -> EventSubscription:
        """取回 backlog 并登记订阅。两步在事件日志的锁里完成，中间不会漏事件。"""

        with self._lock:
            return self._broker.subscribe(
                run_id,
                after_sequence=after_sequence,
                replay=lambda: self._events.read_events(
                    run_id, after_sequence=after_sequence
                ),
            )

    def unsubscribe(self, run_id: str, subscriber) -> None:
        self._broker.unsubscribe(run_id, subscriber)

    def publish_debug_reasoning(self, run_id: str, content: str, *, model: str) -> None:
        """供应商显式 reasoning 只进临时队列：不落盘，断线后可以丢。"""

        self._broker.publish_debug_reasoning(run_id, content, model=model)


def default_event_log() -> LiveEventLog:
    """按环境装配事件日志。

    配了 MySQL 就用可回放的真表：事件是断线续传与重启回放的唯一依据，只存在进程
    内存里时，API 一重启，「那一轮发生过什么」就没人答得上来。
    """

    from codeinsight.infrastructure.db.engine import (
        MySqlConfig,
        create_all_tables,
        create_db_engine,
        create_session_factory,
    )
    from codeinsight.infrastructure.db.stores import MySqlEventLog

    mysql = MySqlConfig.from_env()
    if mysql is None:
        return LiveEventLog()
    engine = create_db_engine(mysql)
    create_all_tables(engine)
    return LiveEventLog(MySqlEventLog(create_session_factory(engine)))

