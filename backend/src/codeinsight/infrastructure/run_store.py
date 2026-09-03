"""Store 的内存实现。零外部依赖，用于单元测试与本地开发。

这些实现不是「临时凑数的假货」，而是**契约的可执行定义**：
MySQL 实现必须通过同一套测试。任何在内存实现里不成立的语义
（比如 CAS 冲突、审批一次性消费），在 MySQL 实现里也不许成立。

线程安全：用一把 ``threading.RLock`` 保护全部写路径。理由是 CAS 与
「一次性消费」都要求读-判断-写这三步不可被打断；FastAPI 的线程池里
两个请求同时进来是常态。锁的粒度粗（整个 store 一把），但这是内存
实现——真正的并发压力由 MySQL 的行锁承担，这里不值得做细粒度优化。
"""

from __future__ import annotations

import threading
from dataclasses import replace

from codeinsight.domain.change import (
    ChangeApproval,
    CodeGoal,
    ConversationSession,
    RunSnapshot,
    StateVersionConflictError,
)
from codeinsight.domain.trace import IdempotencyKey


class ApprovalNotFoundError(Exception):
    """令牌不存在。"""


class ApprovalAlreadyConsumedError(Exception):
    """令牌已被消费过。

    这是审批「一次性」这条性质被实际触发时抛出的异常。它出现在日志里
    通常意味着有重复的应用请求——值得排查，不该被静默忽略。
    """


class ApprovalExpiredError(Exception):
    """令牌已过期。"""


class InMemorySessionStore:
    """Session 与 Goal 的内存实现。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ConversationSession] = {}
        self._goals: dict[str, CodeGoal] = {}

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def save_session(self, session: ConversationSession) -> None:
        with self._lock:
            self._sessions[session.session_id] = session

    def get_goal(self, goal_id: str) -> CodeGoal | None:
        with self._lock:
            return self._goals.get(goal_id)

    def save_goal(self, goal: CodeGoal) -> None:
        with self._lock:
            self._goals[goal.goal_id] = goal

    def list_active_goals(self, session_id: str) -> tuple[CodeGoal, ...]:
        with self._lock:
            found: list[CodeGoal] = []
            for goal in self._goals.values():
                if goal.session_id != session_id:
                    continue
                if goal.status != "ACTIVE":
                    continue
                found.append(goal)
        found.sort(key=lambda item: item.goal_id)
        return tuple(found)


class InMemoryRunStore:
    """Run 快照的内存实现，带 CAS 语义。

    ``save_run`` 是本模块最重要的方法。它模拟的是 MySQL 里这条语句::

        UPDATE runs SET ..., state_version = :new
        WHERE run_id = :id AND state_version = :expected

    然后检查 rowcount。rowcount 为 0 说明有人在你读取之后改过它，
    你手上的状态已经过时——此时**必须拒绝，不能重试**。重试会用
    过期的判断去覆盖别人的正确结果。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, RunSnapshot] = {}

    def get_run(self, run_id: str) -> RunSnapshot | None:
        with self._lock:
            return self._runs.get(run_id)

    def create_run(self, run: RunSnapshot) -> None:
        with self._lock:
            if run.run_id in self._runs:
                raise StateVersionConflictError(
                    f"run {run.run_id} 已存在。重复创建同一个 run 意味着调用方"
                    "把「新建」当成了「恢复」，应改用 get_run + save_run。"
                )
            self._runs[run.run_id] = run

    def save_run(self, run: RunSnapshot, *, expected_version: int) -> None:
        with self._lock:
            current = self._runs.get(run.run_id)
            if current is None:
                raise StateVersionConflictError(f"run {run.run_id} 不存在，无法保存")
            if current.state_version != expected_version:
                raise StateVersionConflictError(
                    f"stateVersion 冲突：库中为 {current.state_version}，"
                    f"调用方期望 {expected_version}。"
                    "说明这个 Run 在你读取之后已被其他请求推进过。"
                    "正确做法是重新读取仓库事实后重新判断，而不是重试写入——"
                    "重试会用过期的判断覆盖别人的正确结果。"
                )
            if run.state_version <= expected_version:
                raise ValueError(
                    f"新快照的 state_version（{run.state_version}）"
                    f"必须大于 expected_version（{expected_version}）。"
                    "每次状态变更都要递增版本号，否则 CAS 形同虚设。"
                )
            self._runs[run.run_id] = run

    def list_runs_for_goal(self, goal_id: str) -> tuple[RunSnapshot, ...]:
        with self._lock:
            found: list[RunSnapshot] = []
            for run in self._runs.values():
                if run.goal_id == goal_id:
                    found.append(run)
        found.sort(key=lambda item: item.run_id)
        return tuple(found)


class InMemoryApprovalStore:
    """审批令牌的内存实现，带原子的一次性消费。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._approvals: dict[str, ChangeApproval] = {}

    def issue(self, approval: ChangeApproval) -> None:
        with self._lock:
            if approval.token in self._approvals:
                raise ValueError(f"令牌 {approval.token} 已存在，不能重复签发")
            self._approvals[approval.token] = approval

    def get(self, token: str) -> ChangeApproval | None:
        with self._lock:
            return self._approvals.get(token)

    def consume(self, token: str, *, now_epoch_ms: int) -> ChangeApproval:
        """原子消费一个令牌，返回被消费的令牌。

        并发调用时只有一个能成功，其余抛 ``ApprovalAlreadyConsumedError``。
        这条性质是「approval bypass 为 0」这条硬门槛的基础。
        """
        with self._lock:
            approval = self._approvals.get(token)
            if approval is None:
                raise ApprovalNotFoundError(f"令牌不存在：{token}")
            if approval.is_consumed:
                raise ApprovalAlreadyConsumedError(
                    f"令牌 {token} 已在 {approval.consumed_at_epoch_ms} 被消费。"
                    "审批是一次性的，同一次批准不能用来应用两次补丁。"
                )
            if now_epoch_ms >= approval.expires_at_epoch_ms:
                raise ApprovalExpiredError(
                    f"令牌 {token} 已于 {approval.expires_at_epoch_ms} 过期。"
                    "过期令牌不得使用——用户当时看到的 diff 可能已经不是现在这份。"
                )
            consumed = replace(approval, consumed_at_epoch_ms=now_epoch_ms)
            self._approvals[token] = consumed
            return consumed


class InMemoryIdempotencyStore:
    """幂等键的内存实现。

    ``register`` 返回 True 表示首次登记（调用方应执行动作），
    返回 False 表示已登记（调用方应跳过并返回既有结果）。
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[tuple[str, str], str] = {}

    def register(self, key: IdempotencyKey, *, result_ref: str) -> bool:
        if not result_ref.strip():
            raise ValueError("result_ref 不能为空——否则命中幂等时无法返回既有结果")
        composite = (key.scope, key.key)
        with self._lock:
            if composite in self._entries:
                return False
            self._entries[composite] = result_ref
            return True

    def lookup(self, key: IdempotencyKey) -> str | None:
        with self._lock:
            return self._entries.get((key.scope, key.key))
