"""AgentRunWorker：承载一次后台 Agent Run 的领取、执行与落状态。

它和 ValidationWorker 的分工是清楚的：这里跑的是模型与工具循环（慢、贵、有上下文），
那边跑的是固定的 Docker Sandbox 校验（快、可重放）。合成一个 Worker 会让两件事共用
一套超时与并发设置，而它们的最优取值恰好相反。

一轮的处理顺序固定：

    1. 用租约领取 Run——重复投递的第二条消息在这里被挡掉，不会跑第二遍模型
    2. 只凭事实 Store 重建上下文（AgentRunContextLoader）
    3. 执行只读路径，把结果翻译成 Run 状态
    4. 把状态写回事实层，并留下可回放的事件

失败路径由 Worker 自己关闭（run_failed / run_unknown），成功路径的 run_finished
目前由 ChatRuntime 在应用执行结果时补上。两条都会写，含义不同：一条说「Worker
放弃了这一轮，原因是 X」，另一条说「这一轮结束了」。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass

from codeinsight.application.agent_run_executor import (
    DEADLINE_EXCEEDED,
    EXECUTION_FAILED,
    AgentRunContext,
    AgentRunContextError,
    AgentRunContextLoader,
    AgentRunExecutionError,
    AgentRunExecutor,
)
from codeinsight.domain.agent_run import (
    CANCELLED,
    COMPLETED,
    FAILED,
    MANUAL_REQUIRED,
    RUNNING,
    UNKNOWN,
    WAITING_APPROVAL,
    WAITING_VALIDATION,
    AgentRunRecord,
    AgentRunTask,
)
from codeinsight.domain.chat import (
    CHAT_CANCELLED,
    CHAT_COMPLETED,
    CHAT_FAILED,
    CHAT_MANUAL_REQUIRED,
    CHAT_RUNNING,
    CHAT_UNKNOWN,
    CHAT_WAITING_APPROVAL,
    CHAT_WAITING_VALIDATION,
)
from codeinsight.domain.ports import AgentRunStore
from codeinsight.domain.trace import RUN_FAILED, RUN_UNKNOWN, WORKER_CLAIMED
from codeinsight.infrastructure.chat_runtime import ChatExecution

# 领取租约的默认长度：Worker 崩掉后，超过它才允许别人接管。太短会把还在跑的
# Run 抢走，太长会让恢复变慢。具体取值由并发实验决定（计划 Task 11）。
DEFAULT_LEASE_MS = 60_000

# 公开错误类别。只记类别，不记异常消息——消息里可能带供应商响应或路径。
RUN_RECORD_MISSING = "RUN_RECORD_MISSING"
UNEXPECTED_ERROR = "UNEXPECTED_ERROR"
# 续跑任务会写隔离工作区，属于计划 Task 6/7 的范围。宁可明确停住，也不猜着跑。
RESUME_TASK_NOT_WIRED = "RESUME_TASK_NOT_WIRED"

_RUN_STATUS_BY_CHAT_STATUS: dict[str, str] = {
    CHAT_COMPLETED: COMPLETED,
    CHAT_WAITING_APPROVAL: WAITING_APPROVAL,
    CHAT_WAITING_VALIDATION: WAITING_VALIDATION,
    CHAT_FAILED: FAILED,
    CHAT_CANCELLED: CANCELLED,
    CHAT_UNKNOWN: UNKNOWN,
    CHAT_MANUAL_REQUIRED: MANUAL_REQUIRED,
    CHAT_RUNNING: RUNNING,
}


@dataclass(frozen=True)
class AgentRunOutcome:
    """一次 handle 的结果。execution 为 None 表示这一轮没有产出回答。

    claimed 回答的是「这次有没有真的开始跑」：没领到租约、或者任务类型根本不在
    本 Worker 职责范围内时，它是 False——调用方据此判断该不该报告失败。
    """

    run_id: str
    status: str
    execution: ChatExecution | None
    error_class: str | None
    claimed: bool

    @property
    def executed(self) -> bool:
        return self.claimed and self.execution is not None

    @property
    def ok(self) -> bool:
        return self.execution is not None and self.status in {COMPLETED, WAITING_APPROVAL}


class AgentRunWorker:
    """领取并执行一次 Agent Run。"""

    def __init__(
        self,
        *,
        store: AgentRunStore,
        loader: AgentRunContextLoader,
        executor: AgentRunExecutor,
        emit: Callable[[str, str, dict[str, str]], None],
        worker_id: str | None = None,
        lease_ms: int = DEFAULT_LEASE_MS,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if lease_ms <= 0:
            raise ValueError("lease_ms 必须为正")
        self._store = store
        self._loader = loader
        self._executor = executor
        self._emit = emit
        self._worker_id = worker_id or f"agent-run-worker-{os.getpid()}"
        self._lease_ms = lease_ms
        self._clock_ms = clock_ms or _now_ms

    @property
    def worker_id(self) -> str:
        return self._worker_id

    def handle(
        self,
        task: AgentRunTask,
        *,
        runner: Callable[[AgentRunContext], ChatExecution],
    ) -> AgentRunOutcome:
        record = self._store.get_run(task.run_id)
        if record is None:
            # 任务指向一个不存在的 Run：不是「还没开始」，而是事实不一致。
            # 直接停住等人工对账，不要凭一条孤立消息去跑模型。
            return AgentRunOutcome(
                run_id=task.run_id,
                status=MANUAL_REQUIRED,
                execution=None,
                error_class=RUN_RECORD_MISSING,
                claimed=False,
            )
        if task.may_have_side_effects:
            # 续跑任务不在本 Worker 的职责范围内；没有领取租约就要如实说没跑。
            return self._stop(
                task.run_id,
                status=MANUAL_REQUIRED,
                error_class=RESUME_TASK_NOT_WIRED,
                event_type=RUN_UNKNOWN,
                claimed=False,
            )
        claimed = self._store.claim_run(
            task.run_id,
            worker_id=self._worker_id,
            lease_until_epoch_ms=self._clock_ms() + self._lease_ms,
        )
        if claimed is None:
            # 别人持有有效租约，或这个 Run 已经不该被领取。两种情况都不该执行。
            return AgentRunOutcome(
                run_id=task.run_id,
                status=record.status,
                execution=None,
                error_class=None,
                claimed=False,
            )
        self._emit(
            task.run_id,
            WORKER_CLAIMED,
            {
                "worker_id": self._worker_id,
                "attempt": str(claimed.attempt),
                "task_id": task.task_id,
                "deadline_epoch_ms": str(claimed.deadline_epoch_ms),
            },
        )
        if self._clock_ms() >= claimed.deadline_epoch_ms:
            return self._stop(
                task.run_id,
                status=FAILED,
                error_class=DEADLINE_EXCEEDED,
                event_type=RUN_FAILED,
                claimed=True,
            )
        try:
            context = self._loader.load(task, claimed)
        except AgentRunContextError as error:
            return self._stop(
                task.run_id,
                status=FAILED,
                error_class=error.error_class,
                event_type=RUN_FAILED,
                claimed=True,
            )
        try:
            execution = self._executor.execute(context, runner=runner)
        except AgentRunExecutionError as error:
            return self._stop(
                task.run_id,
                status=FAILED,
                error_class=error.error_class,
                event_type=RUN_FAILED,
                claimed=True,
            )
        except Exception as error:  # noqa: BLE001 - Worker 边界：任何异常都要落成事实
            return self._stop(
                task.run_id,
                status=MANUAL_REQUIRED,
                error_class=f"{UNEXPECTED_ERROR}:{type(error).__name__}",
                event_type=RUN_UNKNOWN,
                claimed=True,
            )
        status = _RUN_STATUS_BY_CHAT_STATUS.get(execution.status, FAILED)
        error_class = None if status in {COMPLETED, WAITING_APPROVAL} else EXECUTION_FAILED
        self._save(claimed, status=status, error_class=error_class)
        if error_class is not None:
            self._emit(task.run_id, RUN_FAILED, {"status": status, "error_class": error_class})
        return AgentRunOutcome(
            run_id=task.run_id,
            status=status,
            execution=execution,
            error_class=error_class,
            claimed=True,
        )

    def _stop(
        self,
        run_id: str,
        *,
        status: str,
        error_class: str,
        event_type: str,
        claimed: bool,
    ) -> AgentRunOutcome:
        """Worker 自己终止这一轮：写事实、写事件，并如实报告是否真的开始跑过。

        状态不明的终止用 run_unknown 事件，普通失败用 run_failed：前者要求人工
        对账，后者只表示这一轮没成功。混用会让「要不要人来看」失去依据。
        """

        record = self._store.get_run(run_id)
        if record is not None:
            self._save(record, status=status, error_class=error_class)
        self._emit(run_id, event_type, {"status": status, "error_class": error_class})
        return AgentRunOutcome(
            run_id=run_id,
            status=status,
            execution=None,
            error_class=error_class,
            claimed=claimed,
        )

    def _save(self, record: AgentRunRecord, *, status: str, error_class: str | None) -> None:
        """写回终态。

        终态不保留租约与 worker_id：跑完的 Run 不持有租约，谁跑过它由
        worker_claimed 事件负责回答，两处不会互相矛盾。
        """

        stopped = record.advanced(
            status=status,
            updated_at_epoch_ms=self._clock_ms(),
            error_class=error_class,
            event_sequence=record.event_sequence,
        )
        self._store.save_run(stopped)


def _now_ms() -> int:
    return int(time.time() * 1000)

