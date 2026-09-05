"""Memory 记录的进程内实现。

它是三层 Memory 的轻量 Store，负责按 layer/owner 隔离记录；MySQL 实现位于
``infrastructure/db/stores.py``，两者遵守同一个 ``MemoryStore`` 契约。
"""

from __future__ import annotations

import threading

from codeinsight.domain.memory import MemoryRecord


class InMemoryMemoryStore:
    """用于本地开发、测试和 MySQL 不可用时的 Memory Store。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, MemoryRecord] = {}

    def save(self, record: MemoryRecord) -> None:
        with self._lock:
            self._records[record.record_id] = record

    def get(
        self,
        layer: str,
        owner_id: str,
        record_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> MemoryRecord | None:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return None
            if (
                record.layer != layer
                or record.owner_id != owner_id
                or record.tenant_id != tenant_id
                or record.user_id != user_id
                or record.repo_id != repo_id
            ):
                return None
            return record

    def list(
        self,
        layer: str,
        owner_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> tuple[MemoryRecord, ...]:
        with self._lock:
            found: list[MemoryRecord] = []
            for record in self._records.values():
                if (
                    record.layer == layer
                    and record.owner_id == owner_id
                    and record.tenant_id == tenant_id
                    and record.user_id == user_id
                    and record.repo_id == repo_id
                ):
                    found.append(record)
            found.sort(key=lambda item: item.record_id)
            return tuple(found)

    def delete(
        self,
        layer: str,
        owner_id: str,
        record_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> None:
        with self._lock:
            record = self._records.get(record_id)
            if record is None:
                return
            if (
                record.layer == layer
                and record.owner_id == owner_id
                and record.tenant_id == tenant_id
                and record.user_id == user_id
                and record.repo_id == repo_id
            ):
                del self._records[record_id]
