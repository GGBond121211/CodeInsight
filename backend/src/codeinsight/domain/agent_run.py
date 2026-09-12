"""Agent Run 的状态机、后台任务契约与崩溃恢复分类。

为什么单独建一个模块：同一份状态会被 API、AgentRunWorker、ValidationWorker 和 SSE
回放同时读写。三处各写一遍字符串时，最先出错的不是业务逻辑，而是「谁在什么时候把
WAITING 当成 FAILED」——等待审批的 Run 被当成失败重跑，会重复产生 Patch。

三条硬边界（计划第 0.3 节）：

1. 等待态不是失败。WAITING_APPROVAL 与 WAITING_VALIDATION 表示「停下来等外部动作」，
   重启后必须能从事实 Store 恢复，不得进入任何失败分支。
2. 只有从零开始的只读 Run 可以重试。resume_after_approval 与 resume_after_validation
   可能已经产生副作用，状态不明时进入 UNKNOWN，等待人工对账，禁止自动重放。
3. 任务消息只传标识与版本。AgentRunTask 不携带对话正文、源码或凭据。
"""

from __future__ import annotations

from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------
QUEUED = "QUEUED"
RUNNING = "RUNNING"
WAITING_APPROVAL = "WAITING_APPROVAL"
WAITING_VALIDATION = "WAITING_VALIDATION"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
UNKNOWN = "UNKNOWN"
CANCELLED = "CANCELLED"
MANUAL_REQUIRED = "MANUAL_REQUIRED"

SUPPORTED_AGENT_RUN_STATUSES: frozenset[str] = frozenset(
    {
        QUEUED,
        RUNNING,
        WAITING_APPROVAL,
        WAITING_VALIDATION,
        COMPLETED,
        FAILED,
        UNKNOWN,
        CANCELLED,
        MANUAL_REQUIRED,
    }
)

# 终态里没有 UNKNOWN 和 MANUAL_REQUIRED：它们不是「结束」，而是「等人处理」。
TERMINAL_AGENT_RUN_STATUSES: frozenset[str] = frozenset(
    {COMPLETED, FAILED, CANCELLED}
)
WAITING_AGENT_RUN_STATUSES: frozenset[str] = frozenset(
    {WAITING_APPROVAL, WAITING_VALIDATION}
)
ATTENTION_AGENT_RUN_STATUSES: frozenset[str] = frozenset({UNKNOWN, MANUAL_REQUIRED})
# 「还没结束、也还没等人处理」的状态：恢复巡检要扫的就是这些。
OPEN_AGENT_RUN_STATUSES: frozenset[str] = frozenset(
    {QUEUED, RUNNING, WAITING_APPROVAL, WAITING_VALIDATION}
)

# 允许的状态迁移。空集合表示该状态只能由人工或对账流程改写，不由 Worker 推进。
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    QUEUED: frozenset({RUNNING, CANCELLED, FAILED, MANUAL_REQUIRED}),
    RUNNING: frozenset(
        {
            RUNNING,
            WAITING_APPROVAL,
            WAITING_VALIDATION,
            COMPLETED,
            FAILED,
            CANCELLED,
            UNKNOWN,
            MANUAL_REQUIRED,
        }
    ),
    WAITING_APPROVAL: frozenset(
        {QUEUED, RUNNING, CANCELLED, FAILED, UNKNOWN, MANUAL_REQUIRED}
    ),
    WAITING_VALIDATION: frozenset(
        {QUEUED, RUNNING, CANCELLED, FAILED, UNKNOWN, MANUAL_REQUIRED}
    ),
    UNKNOWN: frozenset({RUNNING, COMPLETED, FAILED, MANUAL_REQUIRED, CANCELLED}),
    MANUAL_REQUIRED: frozenset({RUNNING, COMPLETED, FAILED, CANCELLED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    CANCELLED: frozenset(),
}


def can_transition(source: str, target: str) -> bool:
    """判断一次状态迁移是否被允许。同状态自迁移只对 RUNNING 开放。"""

    if source not in SUPPORTED_AGENT_RUN_STATUSES:
        raise ValueError(f"不支持的 Agent Run 状态：{source}")
    if target not in SUPPORTED_AGENT_RUN_STATUSES:
        raise ValueError(f"不支持的 Agent Run 状态：{target}")
    return target in ALLOWED_TRANSITIONS[source]


def transition(source: str, target: str) -> str:
    """返回目标状态；不允许时抛错，避免调用方默默写下一个非法状态。"""

    if not can_transition(source, target):
        raise ValueError(f"不允许的 Agent Run 状态迁移：{source} → {target}")
    return target


def is_waiting(status: str) -> bool:
    """等待审批或等待验证。它既不是失败，也不是终态。"""

    return status in WAITING_AGENT_RUN_STATUSES


def needs_attention(status: str) -> bool:
    """状态不明或需要人工：前端必须显示可查询状态，不能伪装成仍在跑。"""

    return status in ATTENTION_AGENT_RUN_STATUSES


@dataclass(frozen=True)
class RunRequestOptions:
    """受理这一轮时请求方给出的参数。

    它们属于「这一轮被要求做什么」，所以随 Run 一起持久化：Worker 在另一个进程
    里只有 run_id，拿不到请求对象，这些参数不能只活在 API 进程的内存里。
    """

    validation_profile: str = "python_compile"
    result_limit: int = 5
    show_debug_reasoning: bool = False

    def __post_init__(self) -> None:
        if not self.validation_profile.strip():
            raise ValueError("validation_profile 不能为空")
        if self.result_limit < 1:
            raise ValueError("result_limit 至少为 1")


# ---------------------------------------------------------------------------
# 后台任务
# ---------------------------------------------------------------------------
TASK_AGENT_RUN = "agent_run"
TASK_RESUME_AFTER_APPROVAL = "resume_after_approval"
TASK_RESUME_AFTER_VALIDATION = "resume_after_validation"

SUPPORTED_TASK_KINDS: frozenset[str] = frozenset(
    {TASK_AGENT_RUN, TASK_RESUME_AFTER_APPROVAL, TASK_RESUME_AFTER_VALIDATION}
)

# 只有从零开始的 Run 可以自动重试。两种 resume 任务在崩溃时可能已经越过
# approval 或已经写入工作区，自动重放等于重复产生副作用。
RETRYABLE_TASK_KINDS: frozenset[str] = frozenset({TASK_AGENT_RUN})
SIDE_EFFECTING_TASK_KINDS: frozenset[str] = frozenset(
    {TASK_RESUME_AFTER_APPROVAL, TASK_RESUME_AFTER_VALIDATION}
)


@dataclass(frozen=True)
class AgentRunTask:
    """一次后台调度任务。只携带标识与版本，不携带对话正文或源码。"""

    task_id: str
    session_id: str
    turn_id: str
    run_id: str
    task_kind: str
    idempotency_key: str
    policy_version: str
    deadline_epoch_ms: int
    attempt: int = 1
    max_attempts: int = 2
    # 本轮被要求的参数。它们不是对话内容，而是「这一轮要做什么」，
    # 因此随任务一起走到 Worker，不必让 Worker 反查请求。
    options: RunRequestOptions = RunRequestOptions()

    def __post_init__(self) -> None:
        for value, label in (
            (self.task_id, "task_id"),
            (self.session_id, "session_id"),
            (self.turn_id, "turn_id"),
            (self.run_id, "run_id"),
            (self.idempotency_key, "idempotency_key"),
            (self.policy_version, "policy_version"),
        ):
            if not value.strip():
                raise ValueError(f"{label} 不能为空")
        if self.task_kind not in SUPPORTED_TASK_KINDS:
            raise ValueError(f"不支持的任务类型：{self.task_kind}")
        if self.attempt < 1:
            raise ValueError("attempt 从 1 开始")
        if self.max_attempts < 1:
            raise ValueError("max_attempts 至少为 1")
        if self.attempt > self.max_attempts:
            raise ValueError("attempt 不能超过 max_attempts")
        if self.deadline_epoch_ms <= 0:
            raise ValueError("deadline_epoch_ms 必须为正")

    @property
    def retryable(self) -> bool:
        """只读 Run 可以在有限的 attempt 内重试。"""

        return (
            self.task_kind in RETRYABLE_TASK_KINDS
            and self.attempt < self.max_attempts
        )

    @property
    def may_have_side_effects(self) -> bool:
        """续跑任务可能已经写过工作区，恢复时不能盲目重放。"""

        return self.task_kind in SIDE_EFFECTING_TASK_KINDS

    def has_expired(self, *, now_epoch_ms: int) -> bool:
        return now_epoch_ms >= self.deadline_epoch_ms

    def next_attempt(self, *, task_id: str | None = None) -> AgentRunTask:
        """派生下一次尝试；task_id 缺省时沿用同一个任务标识。"""

        if not self.retryable:
            raise ValueError("该任务不允许继续重试")
        return replace(
            self,
            task_id=task_id or self.task_id,
            attempt=self.attempt + 1,
        )

    def as_payload(self) -> dict[str, str | int]:
        """跨进程传递的最小载荷。任何对话正文、源码与凭据都不在这里。"""

        return {
            "task_id": self.task_id,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "run_id": self.run_id,
            "task_kind": self.task_kind,
            "idempotency_key": self.idempotency_key,
            "policy_version": self.policy_version,
            "deadline_epoch_ms": self.deadline_epoch_ms,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "validation_profile": self.options.validation_profile,
            "result_limit": self.options.result_limit,
            "show_debug_reasoning": self.options.show_debug_reasoning,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> AgentRunTask:
        missing = [
            key
            for key in (
                "task_id",
                "session_id",
                "turn_id",
                "run_id",
                "task_kind",
                "idempotency_key",
                "policy_version",
                "deadline_epoch_ms",
            )
            if key not in payload
        ]
        if missing:
            raise ValueError(f"任务载荷缺少字段：{sorted(missing)}")
        return cls(
            task_id=str(payload["task_id"]),
            session_id=str(payload["session_id"]),
            turn_id=str(payload["turn_id"]),
            run_id=str(payload["run_id"]),
            task_kind=str(payload["task_kind"]),
            idempotency_key=str(payload["idempotency_key"]),
            policy_version=str(payload["policy_version"]),
            deadline_epoch_ms=int(payload["deadline_epoch_ms"]),  # type: ignore[arg-type]
            attempt=int(payload.get("attempt", 1)),  # type: ignore[arg-type]
            max_attempts=int(payload.get("max_attempts", 2)),  # type: ignore[arg-type]
            options=RunRequestOptions(
                validation_profile=str(payload.get("validation_profile", "python_compile")),
                result_limit=int(payload.get("result_limit", 5)),  # type: ignore[arg-type]
                show_debug_reasoning=bool(payload.get("show_debug_reasoning", False)),
            ),
        )


@dataclass(frozen=True)
class RunOutput:
    """一次 Agent Run 尝试的产出，落成事实供别的进程读。

    它和 AgentRunRecord 的分工：Record 回答「这一轮跑到哪儿了」，RunOutput 回答
    「这一轮说出了什么」。分开是必要的——「已排队但没人执行」和「跑完了但答案没
    传回来」是两种不同的故障，混在一张表里就分不出来。

    只存公开产出：给用户看的正文与结果载荷。模型隐藏推理、原始供应商响应与
    凭据都不在这里（debug_reasoning 有自己的落点）。
    """

    run_id: str
    attempt: int
    task_type: str
    assistant_message: str
    result: dict[str, object] | None = None
    error_class: str | None = None
    updated_at_epoch_ms: int = 0

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if self.attempt < 1:
            raise ValueError("attempt 必须为正")
        if self.updated_at_epoch_ms < 0:
            raise ValueError("时间戳不能为负")


@dataclass(frozen=True)
class AgentRunRecord:
    """一次 Agent Run 的可持久化事实。

    它同时承担两个角色：告诉 API「这一轮现在处于什么状态」，以及告诉恢复流程
    「上一次是谁、在第几次 attempt 上、租约什么时候过期」。仅此一处真相，
    Celery result backend 与 Redis 队列都不再各自维护状态。
    """

    run_id: str
    turn_id: str
    session_id: str
    task_id: str
    task_kind: str
    status: str
    policy_version: str
    idempotency_key: str
    deadline_epoch_ms: int
    updated_at_epoch_ms: int
    attempt: int = 1
    max_attempts: int = 2
    worker_id: str | None = None
    lease_until_epoch_ms: int | None = None
    error_class: str | None = None
    event_sequence: int = 0
    options: RunRequestOptions = RunRequestOptions()
    # 续跑任务要用到的审批事实。令牌是一次性、短时的，本来也存在 approvals
    # 事实表里；这里保存的是同一个事实的引用，好让另一个进程的 Worker 找到它。
    patch_id: str | None = None
    approval_token: str | None = None

    def __post_init__(self) -> None:
        for value, label in (
            (self.run_id, "run_id"),
            (self.turn_id, "turn_id"),
            (self.session_id, "session_id"),
            (self.task_id, "task_id"),
            (self.policy_version, "policy_version"),
            (self.idempotency_key, "idempotency_key"),
        ):
            if not value.strip():
                raise ValueError(f"{label} 不能为空")
        if self.status not in SUPPORTED_AGENT_RUN_STATUSES:
            raise ValueError(f"不支持的 Agent Run 状态：{self.status}")
        if self.task_kind not in SUPPORTED_TASK_KINDS:
            raise ValueError(f"不支持的任务类型：{self.task_kind}")
        if self.attempt < 1 or self.max_attempts < self.attempt:
            raise ValueError("attempt 必须落在 1..max_attempts 之间")
        if self.deadline_epoch_ms <= 0 or self.updated_at_epoch_ms <= 0:
            raise ValueError("时间戳必须为正")
        if self.event_sequence < 0:
            raise ValueError("event_sequence 不能为负")

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_AGENT_RUN_STATUSES

    @property
    def waiting(self) -> bool:
        """等待审批或等待验证。停机重启后必须能从这里恢复，而不是当成失败。"""

        return is_waiting(self.status)

    @property
    def needs_attention(self) -> bool:
        return needs_attention(self.status)

    @property
    def lease_expired(self) -> bool:
        """未持有租约视为已过期；否则按截止时间判断。"""

        return self.lease_until_epoch_ms is None

    def lease_expired_at(self, *, now_epoch_ms: int) -> bool:
        return (
            self.lease_until_epoch_ms is None
            or self.lease_until_epoch_ms <= now_epoch_ms
        )

    def can_be_claimed(self, *, now_epoch_ms: int) -> bool:
        """只有还在队列里、或租约已过期的 Run 允许被领取。"""

        if self.status == QUEUED:
            return True
        return self.status == RUNNING and self.lease_expired_at(now_epoch_ms=now_epoch_ms)

    def as_public_dict(self) -> dict[str, object]:
        """前端可见字段。不含租约、任务载荷、approval token 或任何正文。"""

        return {
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "status": self.status,
            "task_kind": self.task_kind,
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "error_class": self.error_class,
            "event_sequence": self.event_sequence,
            "updated_at_epoch_ms": self.updated_at_epoch_ms,
        }

    def to_task(self) -> AgentRunTask:
        return AgentRunTask(
            task_id=self.task_id,
            session_id=self.session_id,
            turn_id=self.turn_id,
            run_id=self.run_id,
            task_kind=self.task_kind,
            idempotency_key=self.idempotency_key,
            policy_version=self.policy_version,
            deadline_epoch_ms=self.deadline_epoch_ms,
            attempt=self.attempt,
            max_attempts=self.max_attempts,
        )

    def advanced(
        self,
        *,
        status: str,
        updated_at_epoch_ms: int,
        error_class: str | None = None,
        worker_id: str | None = None,
        lease_until_epoch_ms: int | None = None,
        event_sequence: int | None = None,
        task_id: str | None = None,
        task_kind: str | None = None,
        attempt: int | None = None,
        max_attempts: int | None = None,
        patch_id: str | None = None,
        approval_token: str | None = None,
        deadline_epoch_ms: int | None = None,
    ) -> AgentRunRecord:
        """派生一个新状态；迁移非法时抛错，而不是写下自相矛盾的事实。"""

        checked = transition(self.status, status)
        return replace(
            self,
            task_id=task_id or self.task_id,
            # 任务类型必须跟着一起走：事实层说这还是一开始的 Run，恢复逻辑就会
            # 按「可以重新开始的 Run」去重放一次已经批准的续跑。
            task_kind=task_kind or self.task_kind,
            status=checked,
            attempt=self.attempt if attempt is None else attempt,
            max_attempts=self.max_attempts if max_attempts is None else max_attempts,
            # deadline 只在续跑时由发放 continuation 的那一方显式刷新；不改动时
            # 沿用原值，避免状态写入顺手把 Run 的执行预算改掉。
            deadline_epoch_ms=(
                self.deadline_epoch_ms
                if deadline_epoch_ms is None
                else deadline_epoch_ms
            ),
            # 这两个字段只在显式给出时改写：None 表示「沿用已有事实」，
            # 因为终态写入不应该顺手把补丁标识抹掉。
            patch_id=patch_id or self.patch_id,
            approval_token=approval_token or self.approval_token,
            worker_id=worker_id,
            lease_until_epoch_ms=lease_until_epoch_ms,
            error_class=error_class,
            updated_at_epoch_ms=updated_at_epoch_ms,
            event_sequence=(
                self.event_sequence if event_sequence is None else event_sequence
            ),
        )


# 恢复结论。RETRY 表示可以安全重放；UNKNOWN 表示不确定，交人工对账。
RECOVERY_RETRY = "RETRY"
RECOVERY_UNKNOWN = "UNKNOWN"
RECOVERY_MANUAL_REQUIRED = "MANUAL_REQUIRED"


def recovery_decision(
    task: AgentRunTask,
    *,
    lease_expired: bool,
    now_epoch_ms: int,
) -> str:
    """崩溃后如何恢复一次任务。

    只读 Run：租约过期就能重试；重试次数用完或已过 deadline 则交人工。
    续跑任务：只要租约过期就进入 UNKNOWN——它可能已经写过隔离工作区，
    「重放一次」和「重复一次副作用」在外部看来无法区分。
    """

    if not lease_expired:
        return RECOVERY_UNKNOWN
    if task.may_have_side_effects:
        return RECOVERY_UNKNOWN
    if task.has_expired(now_epoch_ms=now_epoch_ms) or not task.retryable:
        return RECOVERY_MANUAL_REQUIRED
    return RECOVERY_RETRY
