"""可恢复 TaskEnvelope 与本地队列契约。"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from typing import Any

from redis import Redis

from codeinsight.infrastructure.redis_cache import ENV_REDIS_URL

# Agent Run 不在这条队列上：它走 Celery（codeinsight.run_agent），状态事实在
# domain.agent_run 与 AgentRunStore 里。下面这组类型会被拒收 payload——消息里
# 一旦出现正文、源码或凭据，队列就成了第二份事实，恢复时两处必然对不上。
AGENT_RUN_TASK_TYPES: frozenset[str] = frozenset(
    {"agent_run", "resume_after_approval", "resume_after_validation"}
)

# 校验任务在共享 Redis 上的命名空间。同一个 Redis 可以同时承载 Celery broker、
# 热点缓存和任务协调状态，但三者不能共用一个前缀：清理缓存时顺手删掉任务租约，
# 恢复巡检就会看到一条其实是「正在跑」的任务。
VALIDATION_QUEUE_PREFIX = "codeinsight:tasks:validation:v1"

QUEUED = "QUEUED"
RUNNING = "RUNNING"
COMPLETED = "COMPLETED"
MANUAL_REQUIRED = "MANUAL_REQUIRED"


@dataclass(frozen=True)
class TaskEnvelope:
    task_id: str
    run_id: str
    session_id: str
    idempotency_key: str
    task_type: str
    payload: dict[str, str]
    deadline_epoch_ms: int
    max_attempts: int = 2
    attempt: int = 0
    status: str = QUEUED
    worker_id: str | None = None
    lease_until_epoch_ms: int | None = None
    error_class: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id or not self.run_id or not self.idempotency_key:
            raise ValueError("TaskEnvelope 标识不能为空")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 至少为 1")
        if self.task_type in AGENT_RUN_TASK_TYPES and self.payload:
            raise ValueError(
                f"Agent Run 任务（{self.task_type}）不得携带 payload："
                "队列消息只传标识与版本，其余从事实 Store 读取。"
            )


class InMemoryTaskQueue:
    """用 lease 表达 Worker 崩溃恢复；幂等键阻止重复投递。"""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskEnvelope] = {}
        self._by_idempotency: dict[str, str] = {}
        self._lock = threading.RLock()

    def submit(self, task: TaskEnvelope) -> TaskEnvelope:
        with self._lock:
            existing_id = self._by_idempotency.get(task.idempotency_key)
            if existing_id is not None:
                return self._tasks[existing_id]
            if task.task_id in self._tasks:
                raise ValueError("task_id 已存在")
            self._tasks[task.task_id] = task
            self._by_idempotency[task.idempotency_key] = task.task_id
            return task

    def get_task(self, task_id: str) -> TaskEnvelope | None:
        """读一条任务当前的事实，不改变它的状态。"""

        with self._lock:
            return self._tasks.get(task_id)

    def claim_task(
        self, task_id: str, *, worker_id: str, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        """按标识原子领取指定的一条任务。

        投递过来的消息点名要跑哪一条任务，Worker 不能挑「队列里的下一条」。
        重复投递的第二条消息在这里拿到 None：租约还在，同一条任务不会再跑一遍。
        """

        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        with self._lock:
            return self._claim_locked(
                task_id, worker_id=worker_id, now_epoch_ms=now_epoch_ms, lease_ms=lease_ms
            )

    def claim(
        self, worker_id: str, *, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        """拉取式入口：给「谁有空谁跑」的场景用；点名的投递走 claim_task。"""

        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        with self._lock:
            for candidate in sorted(self._tasks):
                claimed = self._claim_locked(
                    candidate,
                    worker_id=worker_id,
                    now_epoch_ms=now_epoch_ms,
                    lease_ms=lease_ms,
                )
                if claimed is not None:
                    return claimed
            return None

    def _claim_locked(
        self, task_id: str, *, worker_id: str, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None
        expired = (
            task.status == RUNNING
            and task.lease_until_epoch_ms is not None
            and task.lease_until_epoch_ms < now_epoch_ms
        )
        if task.status != QUEUED and not expired:
            return None
        if task.deadline_epoch_ms <= now_epoch_ms or task.attempt >= task.max_attempts:
            self._tasks[task_id] = replace(
                task, status=MANUAL_REQUIRED, error_class="DEADLINE_OR_RETRY_EXHAUSTED"
            )
            return None
        claimed = replace(
            task,
            status=RUNNING,
            worker_id=worker_id,
            lease_until_epoch_ms=now_epoch_ms + lease_ms,
            attempt=task.attempt + 1,
        )
        self._tasks[task_id] = claimed
        return claimed

    def complete(self, task_id: str) -> TaskEnvelope:
        with self._lock:
            task = self._require(task_id)
            completed = replace(
                task, status=COMPLETED, worker_id=None, lease_until_epoch_ms=None
            )
            self._tasks[task_id] = completed
            return completed

    def fail(self, task_id: str, error_class: str, *, retryable: bool) -> TaskEnvelope:
        with self._lock:
            task = self._require(task_id)
            retry = retryable and task.attempt < task.max_attempts
            failed = replace(
                task,
                status=QUEUED if retry else MANUAL_REQUIRED,
                worker_id=None,
                lease_until_epoch_ms=None,
                error_class=error_class,
            )
            self._tasks[task_id] = failed
            return failed

    def list_tasks(self) -> tuple[TaskEnvelope, ...]:
        with self._lock:
            return tuple(self._tasks[key] for key in sorted(self._tasks))

    def _require(self, task_id: str) -> TaskEnvelope:
        try:
            return self._tasks[task_id]
        except KeyError:
            raise KeyError(f"任务不存在：{task_id}") from None


class RedisTaskQueue:
    """跨 Worker 进程共享 lease/idempotency 状态的 Redis 实现。"""

    def __init__(self, client: Redis[Any], *, prefix: str = "codeinsight:tasks:v1") -> None:
        self.client = client
        self.prefix = prefix.rstrip(":")

    def submit(self, task: TaskEnvelope) -> TaskEnvelope:
        with self.client.lock(f"{self.prefix}:lock", timeout=5, blocking_timeout=5):
            existing = self.client.get(self._idempotency_key(task.idempotency_key))
            if existing is not None:
                return self._require(_text(existing))
            if self.client.exists(self._task_key(task.task_id)):
                raise ValueError("task_id 已存在")
            self._save(task)
            self.client.set(self._idempotency_key(task.idempotency_key), task.task_id)
            return task

    def get_task(self, task_id: str) -> TaskEnvelope | None:
        """读一条任务当前的事实，不改变它的状态。"""

        if not self.client.exists(self._task_key(task_id)):
            return None
        return self._require(task_id)

    def claim_task(
        self, task_id: str, *, worker_id: str, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        """按标识原子领取指定的一条任务，与内存实现同语义。"""

        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        with self.client.lock(f"{self.prefix}:lock", timeout=5, blocking_timeout=5):
            return self._claim_locked(
                task_id, worker_id=worker_id, now_epoch_ms=now_epoch_ms, lease_ms=lease_ms
            )

    def claim(
        self, worker_id: str, *, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        """拉取式入口：给「谁有空谁跑」的场景用；点名的投递走 claim_task。"""

        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        with self.client.lock(f"{self.prefix}:lock", timeout=5, blocking_timeout=5):
            for task in self.list_tasks():
                claimed = self._claim_locked(
                    task.task_id,
                    worker_id=worker_id,
                    now_epoch_ms=now_epoch_ms,
                    lease_ms=lease_ms,
                )
                if claimed is not None:
                    return claimed
            return None

    def _claim_locked(
        self, task_id: str, *, worker_id: str, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None:
        if not self.client.exists(self._task_key(task_id)):
            return None
        task = self._require(task_id)
        expired = (
            task.status == RUNNING
            and task.lease_until_epoch_ms is not None
            and task.lease_until_epoch_ms < now_epoch_ms
        )
        if task.status != QUEUED and not expired:
            return None
        if task.deadline_epoch_ms <= now_epoch_ms or task.attempt >= task.max_attempts:
            self._save(
                replace(
                    task,
                    status=MANUAL_REQUIRED,
                    error_class="DEADLINE_OR_RETRY_EXHAUSTED",
                )
            )
            return None
        claimed = replace(
            task,
            status=RUNNING,
            worker_id=worker_id,
            lease_until_epoch_ms=now_epoch_ms + lease_ms,
            attempt=task.attempt + 1,
        )
        self._save(claimed)
        return claimed

    def complete(self, task_id: str) -> TaskEnvelope:
        with self.client.lock(f"{self.prefix}:lock", timeout=5, blocking_timeout=5):
            task = self._require(task_id)
            completed = replace(
                task, status=COMPLETED, worker_id=None, lease_until_epoch_ms=None
            )
            self._save(completed)
            return completed

    def fail(self, task_id: str, error_class: str, *, retryable: bool) -> TaskEnvelope:
        with self.client.lock(f"{self.prefix}:lock", timeout=5, blocking_timeout=5):
            task = self._require(task_id)
            retry = retryable and task.attempt < task.max_attempts
            failed = replace(
                task,
                status=QUEUED if retry else MANUAL_REQUIRED,
                worker_id=None,
                lease_until_epoch_ms=None,
                error_class=error_class,
            )
            self._save(failed)
            return failed

    def list_tasks(self) -> tuple[TaskEnvelope, ...]:
        tasks = [
            self._load_key(key)
            for key in self.client.scan_iter(match=f"{self.prefix}:task:*")
        ]
        tasks.sort(key=lambda item: item.task_id)
        return tuple(tasks)

    def clear_test_namespace(self) -> None:
        """仅删除当前测试 prefix；生产调用方不应使用。"""
        keys = list(self.client.scan_iter(match=f"{self.prefix}:*"))
        if keys:
            self.client.delete(*keys)

    def _save(self, task: TaskEnvelope) -> None:
        self.client.set(
            self._task_key(task.task_id),
            json.dumps(asdict(task), ensure_ascii=False, sort_keys=True),
        )

    def _require(self, task_id: str) -> TaskEnvelope:
        value = self.client.get(self._task_key(task_id))
        if value is None:
            raise KeyError(f"任务不存在：{task_id}")
        payload = json.loads(_text(value))
        return TaskEnvelope(**payload)

    def _load_key(self, key: Any) -> TaskEnvelope:
        value = self.client.get(key)
        if value is None:
            raise KeyError("扫描期间任务消失")
        return TaskEnvelope(**json.loads(_text(value)))

    def _task_key(self, task_id: str) -> str:
        return f"{self.prefix}:task:{task_id}"

    def _idempotency_key(self, value: str) -> str:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
        return f"{self.prefix}:idempotency:{digest}"


def default_validation_queue() -> InMemoryTaskQueue | RedisTaskQueue:
    """按环境装配校验队列。

    配了 Redis 就用共享队列：校验 Worker 可能在另一个进程里，进程内存里的队列
    对它来说等于不存在。没配就退回内存实现——那是单进程本地开发的语义，跨进程
    投递时会找不到任务，而不是「跑过了但没人记得」。
    """

    url = os.environ.get(ENV_REDIS_URL, "").strip()
    if not url:
        return InMemoryTaskQueue()
    return RedisTaskQueue(
        Redis.from_url(url, decode_responses=True), prefix=VALIDATION_QUEUE_PREFIX
    )


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
