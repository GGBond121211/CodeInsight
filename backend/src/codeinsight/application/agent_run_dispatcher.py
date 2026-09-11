"""Agent Run 的投递契约与投递实现。

为什么要有这一层：API 进程和 Worker 进程必须对「这一轮已经在路上」给出同一个
答案。如果 API 直接调用 Celery 的 apply_async，「已入队」这句话就只存在于 broker
里；进程重启、broker 丢消息、投递超时之后，没有任何可查询的事实能证明它跑没跑。

所以投递固定成两步，顺序不能换：

    1. 先把 Run 写成 QUEUED（事实 Store，可查询、可恢复）
    2. 再投递消息（broker，只传标识与版本）

第 2 步失败时 Run 必须落到 FAILED / MANUAL_REQUIRED，不能留在 QUEUED——
「已排队但永远不会有人执行」是本项目最容易骗过自己的状态。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from codeinsight.domain.agent_run import (
    FAILED,
    MANUAL_REQUIRED,
    QUEUED,
    AgentRunRecord,
    AgentRunTask,
)
from codeinsight.domain.ports import AgentRunStore

# 投递失败的公开错误类别。区分两种失败是必要的：只读任务没投出去就是没跑，
# 重发即可以后再说；续跑任务则可能已经在别处生效，归成 MANUAL_REQUIRED
# 交人工对账，而不是自动重放一遍。
DISPATCH_FAILED = "DISPATCH_FAILED"
DISPATCH_UNKNOWN = "DISPATCH_UNKNOWN"


@runtime_checkable
class AgentRunTransport(Protocol):
    """把一条任务消息送到 Worker 那里。只传标识与版本。"""

    def publish(self, task: AgentRunTask) -> str:
        """投递并返回传输层消息 ID。投不出去时抛异常。"""
        ...


class InMemoryAgentRunTransport:
    """单进程模式与单元测试用的投递实现。"""

    def __init__(self, *, failure: Exception | None = None) -> None:
        self.failure = failure
        self._published: list[AgentRunTask] = []

    @property
    def published(self) -> tuple[AgentRunTask, ...]:
        return tuple(self._published)

    def publish(self, task: AgentRunTask) -> str:
        if self.failure is not None:
            raise self.failure
        self._published.append(task)
        return f"memory-{len(self._published)}"


class AgentRunDispatcher:
    """「写事实 + 投消息」的组合，application 层唯一的投递入口。

    幂等落在 turn_id 上，而不是内存字典里：同一条用户消息重发时
    find_run_by_turn 会命中原记录并直接返回，因此不会出现第二个 Run，
    也不会第二次调用模型。
    """

    def __init__(
        self,
        *,
        store: AgentRunStore,
        transport: AgentRunTransport,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._store = store
        self._transport = transport
        self._clock_ms = clock_ms or _now_ms

    def dispatch(self, task: AgentRunTask) -> AgentRunRecord:
        """创建（或返回已有的）Run，并把任务投给 Worker。"""

        existing = self._store.find_run_by_turn(task.turn_id)
        if existing is not None:
            return existing
        record = AgentRunRecord(
            run_id=task.run_id,
            turn_id=task.turn_id,
            session_id=task.session_id,
            task_id=task.task_id,
            task_kind=task.task_kind,
            status=QUEUED,
            policy_version=task.policy_version,
            idempotency_key=task.idempotency_key,
            deadline_epoch_ms=task.deadline_epoch_ms,
            updated_at_epoch_ms=self._clock_ms(),
            attempt=task.attempt,
            max_attempts=task.max_attempts,
        )
        self._store.save_run(record)
        return self._publish(task, record)

    def _publish(self, task: AgentRunTask, record: AgentRunRecord) -> AgentRunRecord:
        try:
            self._transport.publish(task)
        except Exception:
            ambiguous = task.may_have_side_effects
            stopped = record.advanced(
                status=MANUAL_REQUIRED if ambiguous else FAILED,
                updated_at_epoch_ms=self._clock_ms(),
                error_class=DISPATCH_UNKNOWN if ambiguous else DISPATCH_FAILED,
            )
            self._store.save_run(stopped)
            return stopped
        return record


def _now_ms() -> int:
    return int(time.time() * 1000)

