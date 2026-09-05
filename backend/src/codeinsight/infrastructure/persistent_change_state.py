"""Step 8 代码变更状态的本地持久实现。

文件只放在受管控 workspace 根目录中；写入使用临时文件 + ``os.replace``。
审批令牌只以 SHA-256 文件名落盘，原始 token 不持久化。
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from codeinsight.domain.change import ChangeApproval
from codeinsight.domain.trace import AuditRecord, IdempotencyKey, RunEvent
from codeinsight.infrastructure.event_log import EventSequenceError
from codeinsight.infrastructure.run_store import (
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)


class JsonObjectStore:
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def put(self, namespace: str, key: str, payload: dict[str, Any]) -> None:
        with self._lock:
            target = self._path(namespace, key)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
            os.replace(temporary, target)

    def get(self, namespace: str, key: str) -> dict[str, Any] | None:
        target = self._path(namespace, key)
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def list(self, namespace: str) -> tuple[dict[str, Any], ...]:
        directory = self.base_dir / namespace
        if not directory.is_dir():
            return ()
        return tuple(
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        )

    def _path(self, namespace: str, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self.base_dir / namespace / f"{digest}.json"


class PersistentApprovalStore:
    def __init__(self, base_dir: str | Path) -> None:
        self._objects = JsonObjectStore(Path(base_dir) / "approvals")
        self._lock = threading.RLock()

    def issue(self, approval: ChangeApproval) -> None:
        key = self._key(approval.token)
        if self._objects.get("active", key) or self._objects.get("consumed", key):
            raise ValueError("审批令牌已存在，不能重复签发")
        payload = asdict(approval)
        payload["token"] = key
        self._objects.put("active", key, payload)

    def get(self, token: str) -> ChangeApproval | None:
        key = self._key(token)
        payload = self._objects.get("active", key) or self._objects.get("consumed", key)
        if payload is None:
            return None
        payload["token"] = token
        payload["scope"] = tuple(payload["scope"])
        return ChangeApproval(**payload)

    def consume(self, token: str, *, now_epoch_ms: int) -> ChangeApproval:
        key = self._key(token)
        with self._lock:
            approval = self.get(token)
            if approval is None:
                raise ApprovalNotFoundError("审批令牌不存在")
            if approval.is_consumed:
                raise ApprovalAlreadyConsumedError("审批令牌已被消费")
            if now_epoch_ms >= approval.expires_at_epoch_ms:
                raise ApprovalExpiredError("审批令牌已过期")
            consumed = replace(approval, consumed_at_epoch_ms=now_epoch_ms)
            payload = asdict(consumed)
            payload["token"] = key
            self._objects.put("consumed", key, payload)
            active = self._objects._path("active", key)
            if active.exists():
                active.unlink()
            return consumed

    @staticmethod
    def _key(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


class PersistentIdempotencyStore:
    def __init__(self, base_dir: str | Path) -> None:
        self._objects = JsonObjectStore(Path(base_dir) / "idempotency")
        self._lock = threading.RLock()

    def register(self, key: IdempotencyKey, *, result_ref: str) -> bool:
        composite = f"{key.scope}\0{key.key}"
        with self._lock:
            if self._objects.get("entries", composite) is not None:
                return False
            self._objects.put("entries", composite, {"result_ref": result_ref})
            return True

    def lookup(self, key: IdempotencyKey) -> str | None:
        found = self._objects.get("entries", f"{key.scope}\0{key.key}")
        return str(found["result_ref"]) if found else None


class PersistentEventLog:
    def __init__(self, base_dir: str | Path) -> None:
        self._objects = JsonObjectStore(Path(base_dir) / "events")
        self._lock = threading.RLock()

    def append(self, event: RunEvent) -> None:
        with self._lock:
            existing = list(self.read_events(event.run_id))
            expected = len(existing) + 1
            if event.sequence != expected:
                raise EventSequenceError(f"事件序号应为 {expected}，实际为 {event.sequence}")
            existing.append(event)
            self._objects.put("runs", event.run_id, {"events": [asdict(x) for x in existing]})

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        found = self._objects.get("runs", run_id)
        if found is None:
            return ()
        return tuple(RunEvent(**raw) for raw in found["events"] if raw["sequence"] > after_sequence)

    def next_sequence(self, run_id: str) -> int:
        return len(self.read_events(run_id)) + 1

    def count(self, run_id: str) -> int:
        return len(self.read_events(run_id))


class PersistentAuditLog:
    def __init__(self, base_dir: str | Path) -> None:
        self._objects = JsonObjectStore(Path(base_dir) / "audit")
        self._lock = threading.RLock()

    def record(self, entry: AuditRecord) -> None:
        with self._lock:
            existing = list(self.read_records(entry.run_id))
            if any(item.audit_id == entry.audit_id for item in existing):
                raise ValueError("审计记录已存在")
            existing.append(entry)
            self._objects.put("runs", entry.run_id, {"records": [asdict(x) for x in existing]})

    def read_records(self, run_id: str) -> tuple[AuditRecord, ...]:
        found = self._objects.get("runs", run_id)
        if found is None:
            return ()
        return tuple(AuditRecord(**raw) for raw in found["records"])

    def total_records(self) -> int:
        return sum(len(item.get("records", [])) for item in self._objects.list("runs"))
