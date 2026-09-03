"""事件日志与审计流的内存实现。

两者刻意用两个类、两块存储，而不是一张带 ``is_audit`` 列的表——
理由见 ``domain/trace.py`` 的模块 docstring：保留期与采样策略不同，
混在一起迟早会因为「清理旧事件」把审计记录一起删掉。
"""

from __future__ import annotations

import threading

from codeinsight.domain.trace import AuditRecord, RunEvent


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
