"""把「跑固定校验」从 Agent Worker 里拆出来，并让结果成为 Run 状态机的正式输入。

为什么不能留在 Agent Worker 里：Docker 沙箱是最慢、最容易被环境卡住的一步。把它
挂在 Agent 的 attempt 上，等于让一个卡住的镜像占住 Agent 队列，而 Agent Worker 本来
该负责的是「模型与工具」那一段。

这一层只做三件事：

1. 登记：应用完补丁的那个 attempt 把「要跑的固定 profile + 要校验的隔离 workspace」
   写成一条可恢复的 validation 任务，然后立刻结束自己。
2. 执行：ValidationWorker 按标识原子领取任务，跑登记过的 profile，把 started /
   finished 与脱敏后的失败摘要写进事实。
3. 通知：结果出来以后投递一条 Agent continuation。要不要继续修、还能不能自动修，
   由 continuation 按事实层里已经登记的预算决定。

边界：

- profile 与 workspace 路径都取自登记事实，模型不能临时决定跑什么命令，Worker 也不能
  自己挑一个目录。
- 原始沙箱日志不进模型 Context：进 Context 的只有脱敏摘要、错误类别和稳定引用。
- 同一条 validation 任务只能被领取一次；重复投递的第二条消息拿不到租约。
- 这一层不阻塞 Agent Worker：登记完就返回，谁去跑沙箱是另一条消息的事。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

from codeinsight.application.agent_run_dispatcher import (
    AgentRunDispatcher,
    TransportRejected,
)
from codeinsight.application.change_service import ChangeService
from codeinsight.domain.agent_run import (
    TASK_RESUME_AFTER_VALIDATION,
    WAITING_VALIDATION,
    AgentRunRecord,
    AgentRunTask,
)
from codeinsight.domain.ports import AgentRunStore
from codeinsight.domain.trace import VALIDATION_QUEUED
from codeinsight.infrastructure.task_queue import TaskEnvelope

# 校验任务自己的预算：这是固定检查的执行预算，不是用户等待的时间。
DEFAULT_VALIDATION_DEADLINE_MS = 10 * 60 * 1000
# 领取租约：Worker 崩了以后，只有租约到期才能被下一条消息恢复——恰好恢复一次。
DEFAULT_VALIDATION_LEASE_MS = 60_000
# 同一条 validation 任务最多被领取两次：一次正常，一次崩溃恢复。
VALIDATION_MAX_ATTEMPTS = 2

VALIDATION_TASK_TYPE = "registered_validation"

# 结果状态。它们同时是校验任务和 Agent continuation 的输入。
VALIDATION_PASSED = "PASSED"
VALIDATION_REVIEW_REQUIRED = "REVIEW_REQUIRED"
VALIDATION_INCONCLUSIVE = "INCONCLUSIVE"
# 这条消息没有拿到任务：别人在跑、租约未到期，或者尝试次数已用完。
VALIDATION_NOT_CLAIMED = "NOT_CLAIMED"


@runtime_checkable
class ValidationTransport(Protocol):
    """把一条校验任务送到会跑它的地方。消息本身只携带标识与登记过的参数。"""

    def publish(self, task: TaskEnvelope) -> str: ...


@runtime_checkable
class ValidationTaskQueue(Protocol):
    """校验任务的队列能力。语义与 Agent Run 的租约完全一样，只是对象不同。"""

    def submit(self, task: TaskEnvelope) -> TaskEnvelope: ...

    def get_task(self, task_id: str) -> TaskEnvelope | None: ...

    def claim_task(
        self, task_id: str, *, worker_id: str, now_epoch_ms: int, lease_ms: int
    ) -> TaskEnvelope | None: ...

    def complete(self, task_id: str) -> TaskEnvelope: ...

    def fail(self, task_id: str, error_class: str, *, retryable: bool) -> TaskEnvelope: ...


@dataclass(frozen=True)
class ValidationOutcome:
    """一次 validation 任务的结论。

    ``claimed=False`` 表示这条消息没有拿到任务，调用方据此判断「不该继续往下推」。
    """

    task_id: str
    run_id: str
    claimed: bool
    status: str
    profile: str | None = None
    error_class: str | None = None

    @property
    def review_required(self) -> bool:
        return self.status == VALIDATION_REVIEW_REQUIRED


class ValidationWorker:
    """按标识领取并执行一条校验任务。

    它对「该跑什么」没有发言权：profile 与路径都来自登记过的 payload，执行体是
    ``ChangeService.run_registered_validation``，而它只肯在 WAITING_VALIDATION 上启动。
    """

    def __init__(
        self,
        *,
        queue: ValidationTaskQueue,
        change_service: ChangeService,
        emit: Callable[[str, str, dict[str, str]], None],
        worker_id: str | None = None,
        lease_ms: int = DEFAULT_VALIDATION_LEASE_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        self._queue = queue
        self._change_service = change_service
        self._emit = emit
        self._worker_id = worker_id or f"validation-worker-{os.getpid()}"
        self._lease_ms = lease_ms
        self._clock_ms = clock_ms or _now_ms

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def handle(self, task_id: str) -> ValidationOutcome:
        envelope = self._queue.get_task(task_id)
        if envelope is None:
            raise KeyError(f"校验任务不存在：{task_id}")
        claimed = self._queue.claim_task(
            task_id,
            worker_id=self._worker_id,
            now_epoch_ms=self._clock_ms(),
            lease_ms=self._lease_ms,
        )
        if claimed is None:
            return ValidationOutcome(
                task_id=task_id,
                run_id=envelope.run_id,
                claimed=False,
                status=VALIDATION_NOT_CLAIMED,
                profile=_payload_text(envelope, "profile"),
            )
        return self._run(claimed)

    def _run(self, claimed: TaskEnvelope) -> ValidationOutcome:
        run_id = claimed.run_id
        patch_id = _payload_text(claimed, "patch_id")
        profile = _payload_text(claimed, "profile")
        existing = self._change_service.get_result(run_id, patch_id)
        if existing is not None and existing.status != WAITING_VALIDATION:
            # 上一次 attempt 其实已经跑完校验，只是没来得及在队列上收尾。以事实
            # 为准收尾，而不是把同一套检查再跑一遍。
            self._queue.complete(claimed.task_id)
            return self._outcome_for(claimed, existing.status)
        try:
            result = self._change_service.run_registered_validation(
                run_id, patch_id, event_run_id=run_id
            )
        except Exception as error:  # noqa: BLE001 - Worker 边界：任何异常都要变成事实
            error_class = type(error).__name__
            # 环境不可用是「这一轮没有结论」，不是「代码校验失败」：任务允许被
            # 下一条消息恢复一次，而 Run 侧不会因此被说成通过。
            self._queue.fail(claimed.task_id, error_class, retryable=True)
            return ValidationOutcome(
                task_id=claimed.task_id,
                run_id=run_id,
                claimed=True,
                status=VALIDATION_INCONCLUSIVE,
                profile=profile,
                error_class=error_class,
            )
        self._queue.complete(claimed.task_id)
        return self._outcome_for(claimed, result.status)

    def _outcome_for(self, claimed: TaskEnvelope, result_status: str) -> ValidationOutcome:
        status = (
            VALIDATION_PASSED
            if result_status == "COMPLETED"
            else VALIDATION_REVIEW_REQUIRED
        )
        return ValidationOutcome(
            task_id=claimed.task_id,
            run_id=claimed.run_id,
            claimed=True,
            status=status,
            profile=_payload_text(claimed, "profile"),
        )


class CallbackValidationTransport:
    """投递到本地回调：单进程模式下 ValidationWorker 就住在同一个进程里。"""

    def __init__(self, callback: Callable[[TaskEnvelope], str]) -> None:
        self._callback = callback

    def publish(self, task: TaskEnvelope) -> str:
        return self._callback(task)


class ValidationCoordinator:
    """登记校验任务，并在结果出来后投递 Agent continuation。"""

    def __init__(
        self,
        *,
        queue: ValidationTaskQueue,
        worker: ValidationWorker,
        transport: ValidationTransport,
        dispatcher: AgentRunDispatcher,
        store: AgentRunStore,
        emit: Callable[[str, str, dict[str, str]], None],
        deadline_ms: int = DEFAULT_VALIDATION_DEADLINE_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._queue = queue
        self._worker = worker
        self._transport = transport
        self._dispatcher = dispatcher
        self._store = store
        self._emit = emit
        self._deadline_ms = deadline_ms
        self._clock_ms = clock_ms or _now_ms

    @property
    def worker(self) -> ValidationWorker:
        return self._worker

    def register(
        self, record: AgentRunRecord, *, patch_id: str, profile: str, workspace_path: str
    ) -> TaskEnvelope:
        """把「这一轮要跑的固定校验」登记成一条事实，然后投出去。

        幂等键是 run_id + patch_id + profile：同一个补丁的同一套检查只会有一条
        任务，重复登记不会让沙箱跑第二遍。
        """

        envelope = TaskEnvelope(
            task_id=f"validation-{uuid4().hex[:16]}",
            run_id=record.run_id,
            session_id=record.session_id,
            idempotency_key=f"{record.run_id}:{patch_id}:{profile}",
            task_type=VALIDATION_TASK_TYPE,
            payload={
                "profile": profile,
                "patch_id": patch_id,
                "workspace_path": workspace_path,
            },
            deadline_epoch_ms=self._clock_ms() + self._deadline_ms,
            max_attempts=VALIDATION_MAX_ATTEMPTS,
        )
        stored = self._queue.submit(envelope)
        self._emit(
            record.run_id,
            VALIDATION_QUEUED,
            {
                "profile": profile,
                "patch_id": patch_id,
                "task_id": stored.task_id,
                "attempt": str(stored.attempt),
            },
        )
        self._transport.publish(stored)
        return stored

    def after_validation(self, outcome: ValidationOutcome) -> AgentRunRecord | None:
        """把校验结论变成一次 Agent continuation。

        没有拿到任务的返回 None：那条消息没有产生任何结论，不该推动状态机。
        结论的解读（通过、要不要重修、还是交人工）留给 continuation，那里才有
        用户目标、预算和当前 diff 的完整事实。
        """

        if not outcome.claimed:
            return None
        record = self._store.get_run(outcome.run_id)
        if record is None or record.status != WAITING_VALIDATION:
            return None
        task = AgentRunTask(
            task_id=f"task-{uuid4().hex[:16]}",
            session_id=record.session_id,
            turn_id=record.turn_id,
            run_id=record.run_id,
            task_kind=TASK_RESUME_AFTER_VALIDATION,
            idempotency_key=f"{record.idempotency_key}:validation:{outcome.task_id}",
            policy_version=record.policy_version,
            deadline_epoch_ms=self._clock_ms() + self._deadline_ms,
            options=record.options,
        )
        try:
            return self._dispatcher.dispatch_continuation(
                task,
                expected_status=WAITING_VALIDATION,
                patch_id=record.patch_id or "",
            )
        except TransportRejected:
            # 续跑登记不上（Run 已经被别的东西推走了）。事实层说了算，不在这里抢。
            return None


def _payload_text(task: TaskEnvelope, key: str) -> str:
    return str(task.payload.get(key, "")).strip()


def _now_ms() -> int:
    return int(time.time() * 1000)
