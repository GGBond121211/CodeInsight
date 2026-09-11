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
DISPATCH_REJECTED = "DISPATCH_REJECTED"
# 队列满不是「通道坏了」，而是「现在不收」。它必须自己占一个错误类别：
# 混进 DISPATCH_REJECTED 之后，运维分不清是容量问题还是这个请求本身被拒。
QUEUE_FULL = "QUEUE_FULL"


class TransportRejected(RuntimeError):
    """投递被业务规则拒绝（例如同一会话已有在跑的 Run）。

    它和「通道坏了」不是一回事：任务根本没发出去，调用方必须看到真实原因，
    而不是收到一次静默失败。Run 仍会留下一条失败事实。
    """


class QueueFullRejected(TransportRejected):
    """等待领取的 Run 已达上限，这一轮没有被受理。

    它和「通道坏了」的区别在于可预期：背压是设计内的行为，调用方应当退避重试，
    而不是当成故障来排查。
    """


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


class CallbackAgentRunTransport:
    """把投递接到一个进程内回调上：单进程模式与单元测试使用。

    跨进程执行要换成 CeleryAgentRunTransport。这里保留回调，是因为 API 进程
    仍然持有 Worker 需要的进程内上下文。
    """

    def __init__(self, callback: Callable[[AgentRunTask], str]) -> None:
        self._callback = callback

    def publish(self, task: AgentRunTask) -> str:
        return self._callback(task)


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
        queue_limit: int | None = None,
    ) -> None:
        self._store = store
        self._transport = transport
        self._clock_ms = clock_ms or _now_ms
        if queue_limit is not None and queue_limit < 1:
            raise ValueError("queue_limit 必须为正整数或 None")
        self._queue_limit = queue_limit

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
            options=task.options,
        )
        if self._queue_full():
            # 拒绝也要留事实：否则压测里「被拒了多少」只能靠日志数，
            # 而日志不是可查询的账。这里先落一条 FAILED/QUEUE_FULL 再抛。
            stopped = record.advanced(
                status=FAILED,
                updated_at_epoch_ms=self._clock_ms(),
                error_class=QUEUE_FULL,
            )
            self._store.save_run(stopped)
            raise QueueFullRejected(
                f"等待领取的 Run 已达上限 {self._queue_limit}，这一轮没有被受理。"
            )
        self._store.save_run(record)
        return self._publish(task, record)

    def _queue_full(self) -> bool:
        if self._queue_limit is None:
            return False
        return self._store.count_queued_runs() >= self._queue_limit

    def dispatch_continuation(
        self,
        task: AgentRunTask,
        *,
        expected_status: str,
        patch_id: str,
        approval_token: str | None = None,
    ) -> AgentRunRecord:
        """把一个续跑任务接到已有的 Run 上。

        与 dispatch 的区别是它不新建 Run，而是先确认这个 Run 确实停在预期状态
        （例如还在等审批），然后换 task_id、递增 attempt、记下补丁与审批令牌，
        再把消息投出去。审批恢复走这条路。

        为什么要换 task_id 而不是复用：一次 Run 可以由多个后台任务推进，
        任务标识变了，才看得出「这是同一个 Run 的第二次尝试」。
        """

        record = self._store.get_run(task.run_id)
        if record is None:
            raise TransportRejected(f"Run {task.run_id} 不在事实层，无法续跑")
        if record.status != expected_status:
            raise TransportRejected(
                f"Run {task.run_id} 当前是 {record.status}，不是 {expected_status}，不能续跑"
            )
        next_attempt = record.attempt + 1
        queued = record.advanced(
            status=QUEUED,
            updated_at_epoch_ms=self._clock_ms(),
            task_id=task.task_id,
            task_kind=task.task_kind,
            attempt=next_attempt,
            # 每次续跑都是一次新的合法尝试，额度跟着涨；否则第一次续跑就会撞上限。
            max_attempts=max(record.max_attempts, next_attempt),
            patch_id=patch_id,
            approval_token=approval_token,
            # 用户看 diff 花的时间不属于执行预算：续跑从登记这一刻重新计时，
            # 否则「想清楚再点」会把 Run 直接推到 deadline 超时。
            deadline_epoch_ms=task.deadline_epoch_ms,
        )
        self._store.save_run(queued)
        return self._publish(task, queued)

    def _publish(self, task: AgentRunTask, record: AgentRunRecord) -> AgentRunRecord:
        try:
            self._transport.publish(task)
        except TransportRejected:
            self._mark_failed(record, error_class=DISPATCH_REJECTED, needs_attention=False)
            raise
        except Exception:
            ambiguous = task.may_have_side_effects
            self._mark_failed(
                record,
                error_class=DISPATCH_UNKNOWN if ambiguous else DISPATCH_FAILED,
                needs_attention=ambiguous,
            )
            return self._store.get_run(record.run_id) or record
        return record

    def _mark_failed(
        self, record: AgentRunRecord, *, error_class: str, needs_attention: bool
    ) -> AgentRunRecord:
        stopped = record.advanced(
            status=MANUAL_REQUIRED if needs_attention else FAILED,
            updated_at_epoch_ms=self._clock_ms(),
            error_class=error_class,
        )
        self._store.save_run(stopped)
        return stopped


def _now_ms() -> int:
    return int(time.time() * 1000)
