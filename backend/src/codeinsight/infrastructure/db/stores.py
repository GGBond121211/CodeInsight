"""Store 的 MySQL 实现。

这些类实现的是与内存实现完全相同的 ``domain/ports.py`` 契约，并且跑同一套
语义测试（见 ``tests/integration/test_mysql_stores.py``）。

关键差异在于**并发保证的来源**：内存实现靠一把进程内的锁，MySQL 实现靠
数据库的行锁与唯一约束。后者在多进程、多实例部署下依然成立——这正是
从文件 Store 换成数据库的核心理由。

一处刻意的混用：``RunStore`` 用 ORM（为了 ``version_id_col``），
``ApprovalStore.consume`` 与 ``IdempotencyStore.register`` 用 Core 的
``update()`` / ``insert()``。理由是这两处需要的是「带条件的单条语句 +
看 rowcount」，ORM 的 unit-of-work 会把它变成先 SELECT 再 UPDATE，
中间那个窗口正是并发要钻的空子。**用哪层取决于需要哪种保证，不是风格偏好。**
"""

from __future__ import annotations

import json
import time

from sqlalchemy import Engine, and_, delete, func, insert, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.exc import StaleDataError

from codeinsight.domain.agent_run import (
    OPEN_AGENT_RUN_STATUSES,
    QUEUED,
    RUNNING,
    AgentRunRecord,
    RunRequestOptions,
)
from codeinsight.domain.change import (
    ChangeApproval,
    CodeGoal,
    ConversationSession,
    RunSnapshot,
    StateVersionConflictError,
    TenantScope,
    merge_session_facts,
)
from codeinsight.domain.memory import MemoryRecord
from codeinsight.domain.trace import AuditRecord, IdempotencyKey, RunEvent, TraceContext
from codeinsight.infrastructure.db.schema import (
    AgentRunRow,
    ApprovalRow,
    AuditRecordRow,
    GatewayCostRow,
    GoalRow,
    IdempotencyRow,
    MemoryRecordRow,
    RunEventRow,
    RunRow,
    SessionRow,
)
from codeinsight.infrastructure.event_log import EventSequenceError
from codeinsight.infrastructure.run_store import (
    AgentRunTurnConflictError,
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)

# 多值字段在库里的分隔符。选换行而不是逗号：文件路径里可能有逗号，
# 但不可能有换行。
# 空字符串常量：直接写两个字面量引号会被上层模板转义吞掉，
# 这里显式命名一次，读起来也更清楚。
EMPTY_TEXT = ""


SCOPE_SEPARATOR = "\n"


def _join_scope(values: tuple[str, ...]) -> str:
    parts: list[str] = []
    for value in values:
        parts.append(value)
    return SCOPE_SEPARATOR.join(parts)


def _split_scope(raw: str) -> tuple[str, ...]:
    if not raw:
        return ()
    parts: list[str] = []
    for line in raw.split(SCOPE_SEPARATOR):
        if line:
            parts.append(line)
    return tuple(parts)


def _load_json_map(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("payload 必须是 JSON 对象")
    result: dict[str, str] = {}
    for key, value in parsed.items():
        result[str(key)] = str(value)
    return result


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class MySqlMemoryStore:
    """Memory 的 MySQL 事实层；Redis 只缓存这里的读取结果。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save(self, record: MemoryRecord) -> None:
        with self._session_factory() as db:
            row = db.get(MemoryRecordRow, record.record_id)
            if row is None:
                row = MemoryRecordRow(record_id=record.record_id)
                db.add(row)
            row.tenant_id = record.tenant_id
            row.user_id = record.user_id
            row.repo_id = record.repo_id
            row.layer = record.layer
            row.owner_id = record.owner_id
            row.content = record.content
            row.token_estimate = record.token_estimate
            row.is_trusted = record.is_trusted
            row.source = record.source
            row.confidence = record.confidence
            row.expires_at_epoch_ms = record.expires_at_epoch_ms
            row.consent = record.consent
            row.status = record.status
            db.commit()

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
        statement = select(MemoryRecordRow).where(
            MemoryRecordRow.record_id == record_id,
            MemoryRecordRow.layer == layer,
            MemoryRecordRow.owner_id == owner_id,
            MemoryRecordRow.tenant_id == tenant_id,
            MemoryRecordRow.user_id == user_id,
            MemoryRecordRow.repo_id == repo_id,
        )
        with self._session_factory() as db:
            row = db.scalars(statement).first()
            if row is None:
                return None
            return _memory_from_row(row)

    def list(
        self,
        layer: str,
        owner_id: str,
        *,
        tenant_id: str,
        user_id: str,
        repo_id: str,
    ) -> tuple[MemoryRecord, ...]:
        statement = (
            select(MemoryRecordRow)
            .where(
                MemoryRecordRow.layer == layer,
                MemoryRecordRow.owner_id == owner_id,
                MemoryRecordRow.tenant_id == tenant_id,
                MemoryRecordRow.user_id == user_id,
                MemoryRecordRow.repo_id == repo_id,
            )
            .order_by(MemoryRecordRow.record_id)
        )
        with self._session_factory() as db:
            return tuple(_memory_from_row(row) for row in db.scalars(statement))

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
        statement = delete(MemoryRecordRow).where(
            MemoryRecordRow.record_id == record_id,
            MemoryRecordRow.layer == layer,
            MemoryRecordRow.owner_id == owner_id,
            MemoryRecordRow.tenant_id == tenant_id,
            MemoryRecordRow.user_id == user_id,
            MemoryRecordRow.repo_id == repo_id,
        )
        with self._session_factory() as db:
            db.execute(statement)
            db.commit()


def _session_from_row(row: SessionRow) -> ConversationSession:
    return ConversationSession(
        session_id=row.session_id,
        scope=TenantScope(tenant_id=row.tenant_id, user_id=row.user_id),
        repo_id=row.repo_id,
        repo_root=row.repo_root or EMPTY_TEXT,
        summary=row.summary,
        active_goal_id=row.active_goal_id,
    )


def _memory_from_row(row: MemoryRecordRow) -> MemoryRecord:
    return MemoryRecord(
        record_id=row.record_id,
        layer=row.layer,
        owner_id=row.owner_id,
        content=row.content,
        token_estimate=row.token_estimate,
        is_trusted=row.is_trusted,
        tenant_id=row.tenant_id,
        user_id=row.user_id,
        repo_id=row.repo_id,
        source=row.source,
        confidence=row.confidence,
        expires_at_epoch_ms=row.expires_at_epoch_ms,
        consent=row.consent,
        status=row.status,
    )


# ---------------------------------------------------------------------------
# Session / Goal
# ---------------------------------------------------------------------------


class MySqlSessionStore:
    """Session 与 Goal 的 MySQL 实现。

    注意这里**不保存对话轮次明细**。``ConversationSession.recent_turns``
    属于 Session Memory，会随压缩策略变化；把它写进关系表意味着每次
    调整压缩策略都要改表。当前只持久化 ``summary`` 与 ``active_goal_id``
    这两项跨 Run 必需的字段，轮次明细留在内存里。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_session(self, session_id: str) -> ConversationSession | None:
        with self._session_factory() as db:
            row = db.get(SessionRow, session_id)
            if row is None:
                return None
            return _session_from_row(row)

    def save_session(self, session: ConversationSession) -> None:
        with self._session_factory() as db:
            row = db.get(SessionRow, session.session_id)
            if row is None:
                row = SessionRow(session_id=session.session_id)
                db.add(row)
            else:
                session = merge_session_facts(_session_from_row(row), session)
            row.tenant_id = session.scope.tenant_id
            row.user_id = session.scope.user_id
            row.repo_id = session.repo_id
            row.summary = session.summary
            row.active_goal_id = session.active_goal_id
            row.repo_root = session.repo_root or None
            db.commit()

    def get_goal(self, goal_id: str) -> CodeGoal | None:
        with self._session_factory() as db:
            row = db.get(GoalRow, goal_id)
            if row is None:
                return None
            return _goal_from_row(row)

    def save_goal(self, goal: CodeGoal) -> None:
        with self._session_factory() as db:
            row = db.get(GoalRow, goal.goal_id)
            if row is None:
                row = GoalRow(goal_id=goal.goal_id)
                db.add(row)
            row.session_id = goal.session_id
            row.tenant_id = goal.scope.tenant_id
            row.user_id = goal.scope.user_id
            row.repo_id = goal.repo_id
            row.task_type = goal.task_type
            row.user_goal = goal.user_goal
            row.mode = goal.mode
            row.target_scope = _join_scope(goal.target_scope)
            row.validation_profile = goal.validation_profile
            row.status = goal.status
            db.commit()

    def list_active_goals(self, session_id: str) -> tuple[CodeGoal, ...]:
        statement = (
            select(GoalRow)
            .where(GoalRow.session_id == session_id, GoalRow.status == "ACTIVE")
            .order_by(GoalRow.goal_id)
        )
        with self._session_factory() as db:
            found: list[CodeGoal] = []
            for row in db.scalars(statement):
                found.append(_goal_from_row(row))
            return tuple(found)


def _goal_from_row(row: GoalRow) -> CodeGoal:
    return CodeGoal(
        goal_id=row.goal_id,
        session_id=row.session_id,
        scope=TenantScope(tenant_id=row.tenant_id, user_id=row.user_id),
        repo_id=row.repo_id,
        task_type=row.task_type,
        user_goal=row.user_goal,
        mode=row.mode,
        target_scope=_split_scope(row.target_scope),
        validation_profile=row.validation_profile,
        status=row.status,
    )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


class MySqlRunStore:
    """Run 快照的 MySQL 实现，CAS 由 ``version_id_col`` 提供。

    ``save_run`` 的执行过程：

        1. 按 run_id 取出 ORM 行（此时 SQLAlchemy 记下它的 state_version）
        2. 校验它等于 expected_version，不等直接拒绝（不发 SQL）
        3. 改字段并 commit，SQLAlchemy 生成
           ``UPDATE runs SET ... WHERE run_id = ? AND state_version = <旧值>``
        4. rowcount 为 0 → SQLAlchemy 抛 StaleDataError → 转成
           StateVersionConflictError

    第 2 步的显式校验不是多余的：它捕捉「调用方手上的版本本身就过时」
    这种情况，能给出更清楚的错误信息；第 4 步捕捉的是「校验通过之后、
    UPDATE 执行之前」被别人抢先改掉的竞态。两层都需要。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def get_run(self, run_id: str) -> RunSnapshot | None:
        with self._session_factory() as db:
            row = db.get(RunRow, run_id)
            if row is None:
                return None
            return _run_from_row(row)

    def create_run(self, run: RunSnapshot) -> None:
        with self._session_factory() as db:
            row = RunRow(run_id=run.run_id)
            _apply_run_to_row(run, row)
            db.add(row)
            try:
                db.commit()
            except IntegrityError as error:
                db.rollback()
                raise StateVersionConflictError(
                    f"run {run.run_id} 已存在。重复创建同一个 run 意味着调用方"
                    "把「新建」当成了「恢复」，应改用 get_run + save_run。"
                ) from error

    def save_run(self, run: RunSnapshot, *, expected_version: int) -> None:
        if run.state_version <= expected_version:
            raise ValueError(
                f"新快照的 state_version（{run.state_version}）"
                f"必须大于 expected_version（{expected_version}）。"
                "每次状态变更都要递增版本号，否则 CAS 形同虚设。"
            )
        with self._session_factory() as db:
            row = db.get(RunRow, run.run_id)
            if row is None:
                raise StateVersionConflictError(f"run {run.run_id} 不存在，无法保存")
            if row.state_version != expected_version:
                raise StateVersionConflictError(
                    f"stateVersion 冲突：库中为 {row.state_version}，"
                    f"调用方期望 {expected_version}。"
                    "说明这个 Run 在你读取之后已被其他请求推进过。"
                    "正确做法是重新读取仓库事实后重新判断，而不是重试写入——"
                    "重试会用过期的判断覆盖别人的正确结果。"
                )
            _apply_run_to_row(run, row)
            try:
                db.commit()
            except StaleDataError as error:
                db.rollback()
                raise StateVersionConflictError(
                    f"stateVersion 冲突：UPDATE 影响 0 行，"
                    f"run {run.run_id} 在本事务读取之后已被其他请求改动。"
                    "重新读取仓库事实后重新判断，不要重试写入。"
                ) from error

    def list_runs_for_goal(self, goal_id: str) -> tuple[RunSnapshot, ...]:
        statement = select(RunRow).where(RunRow.goal_id == goal_id).order_by(RunRow.run_id)
        with self._session_factory() as db:
            found: list[RunSnapshot] = []
            for row in db.scalars(statement):
                found.append(_run_from_row(row))
            return tuple(found)


def _apply_run_to_row(run: RunSnapshot, row: RunRow) -> None:
    row.session_id = run.session_id
    row.goal_id = run.goal_id
    row.tenant_id = run.scope.tenant_id
    row.user_id = run.scope.user_id
    row.status = run.status
    row.state_version = run.state_version
    row.step_count = run.step_count
    row.max_steps = run.max_steps
    row.next_action = run.next_action
    row.deadline_epoch_ms = run.deadline_epoch_ms
    row.token_budget = run.token_budget
    row.tokens_used = run.tokens_used
    row.repair_attempts = run.repair_attempts
    row.max_repair_attempts = run.max_repair_attempts
    row.stuck_reason = run.stuck_reason
    row.active_tool = run.active_tool


def _run_from_row(row: RunRow) -> RunSnapshot:
    return RunSnapshot(
        run_id=row.run_id,
        session_id=row.session_id,
        goal_id=row.goal_id,
        scope=TenantScope(tenant_id=row.tenant_id, user_id=row.user_id),
        status=row.status,
        state_version=row.state_version,
        step_count=row.step_count,
        max_steps=row.max_steps,
        next_action=row.next_action,
        deadline_epoch_ms=row.deadline_epoch_ms,
        token_budget=row.token_budget,
        tokens_used=row.tokens_used,
        repair_attempts=row.repair_attempts,
        max_repair_attempts=row.max_repair_attempts,
        stuck_reason=row.stuck_reason,
        active_tool=row.active_tool,
    )


# ---------------------------------------------------------------------------
# 事件日志
# ---------------------------------------------------------------------------


class MySqlEventLog:
    """事件日志的 MySQL 实现。

    序号唯一性由 ``UNIQUE(run_id, sequence)`` 保证。应用层的连续性检查
    只是为了给出更清楚的错误信息——真正拦住并发重复写入的是数据库约束。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def append(self, event: RunEvent) -> None:
        with self._session_factory() as db:
            expected = self._next_sequence(db, event.run_id)
            if event.sequence != expected:
                raise EventSequenceError(
                    f"run {event.run_id} 的下一个事件序号应为 {expected}，"
                    f"实际收到 {event.sequence}。"
                    "序号必须连续——有洞会让回放和断线续传都不可信。"
                )
            row = RunEventRow(
                event_id=event.event_id,
                run_id=event.run_id,
                sequence=event.sequence,
                event_type=event.event_type,
                occurred_at_epoch_ms=event.occurred_at_epoch_ms,
                payload_json=json.dumps(event.payload, ensure_ascii=False),
                trace_id=None,
                span_id=None,
                parent_span_id=None,
            )
            if event.trace is not None:
                row.trace_id = event.trace.trace_id
                row.span_id = event.trace.span_id
                row.parent_span_id = event.trace.parent_span_id
            db.add(row)
            try:
                db.commit()
            except IntegrityError as error:
                db.rollback()
                raise EventSequenceError(
                    f"run {event.run_id} 的序号 {event.sequence} 已被占用。"
                    "并发写入撞上了唯一约束——调用方应重新读取当前序号后再追加。"
                ) from error

    def read_events(self, run_id: str, *, after_sequence: int = 0) -> tuple[RunEvent, ...]:
        if after_sequence < 0:
            raise ValueError("after_sequence 不能为负")
        statement = (
            select(RunEventRow)
            .where(RunEventRow.run_id == run_id, RunEventRow.sequence > after_sequence)
            .order_by(RunEventRow.sequence)
        )
        with self._session_factory() as db:
            found: list[RunEvent] = []
            for row in db.scalars(statement):
                found.append(_event_from_row(row))
            return tuple(found)

    def next_sequence(self, run_id: str) -> int:
        with self._session_factory() as db:
            return self._next_sequence(db, run_id)

    def count(self, run_id: str) -> int:
        return self.next_sequence(run_id) - 1

    def _next_sequence(self, db: Session, run_id: str) -> int:
        statement = (
            select(RunEventRow.sequence)
            .where(RunEventRow.run_id == run_id)
            .order_by(RunEventRow.sequence.desc())
            .limit(1)
        )
        highest = db.scalars(statement).first()
        if highest is None:
            return 1
        return highest + 1


def _event_from_row(row: RunEventRow) -> RunEvent:
    trace: TraceContext | None = None
    if row.trace_id is not None and row.span_id is not None:
        trace = TraceContext(
            trace_id=row.trace_id,
            span_id=row.span_id,
            parent_span_id=row.parent_span_id,
        )
    return RunEvent(
        event_id=row.event_id,
        run_id=row.run_id,
        sequence=row.sequence,
        event_type=row.event_type,
        occurred_at_epoch_ms=row.occurred_at_epoch_ms,
        payload=_load_json_map(row.payload_json),
        trace=trace,
    )


class MySqlAuditLog:
    """审计流的 MySQL 实现。与事件表分表，且**不提供任何删除方法**。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def record(self, entry: AuditRecord) -> None:
        with self._session_factory() as db:
            row = AuditRecordRow(
                audit_id=entry.audit_id,
                run_id=entry.run_id,
                event_type=entry.event_type,
                actor=entry.actor,
                occurred_at_epoch_ms=entry.occurred_at_epoch_ms,
                subject=entry.subject,
                outcome=entry.outcome,
                details_json=json.dumps(entry.details, ensure_ascii=False),
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError as error:
                db.rollback()
                raise ValueError(
                    f"审计记录 {entry.audit_id} 已存在，不能重复写入"
                ) from error

    def read_records(self, run_id: str) -> tuple[AuditRecord, ...]:
        statement = (
            select(AuditRecordRow)
            .where(AuditRecordRow.run_id == run_id)
            .order_by(AuditRecordRow.occurred_at_epoch_ms, AuditRecordRow.audit_id)
        )
        with self._session_factory() as db:
            found: list[AuditRecord] = []
            for row in db.scalars(statement):
                found.append(
                    AuditRecord(
                        audit_id=row.audit_id,
                        run_id=row.run_id,
                        event_type=row.event_type,
                        actor=row.actor,
                        occurred_at_epoch_ms=row.occurred_at_epoch_ms,
                        subject=row.subject,
                        outcome=row.outcome,
                        details=_load_json_map(row.details_json),
                    )
                )
            return tuple(found)


# ---------------------------------------------------------------------------
# 审批与幂等
# ---------------------------------------------------------------------------


class MySqlApprovalStore:
    """审批令牌的 MySQL 实现。

    ``consume`` 用一条带条件的 UPDATE 完成，而不是先 SELECT 判断再写：

        UPDATE approvals SET consumed_at_epoch_ms = :now
        WHERE token = :token AND consumed_at_epoch_ms IS NULL

    rowcount 为 1 表示这次消费成功，为 0 表示令牌不存在或已被消费。
    先读后写会留下一个窗口，两个并发请求可以都读到「未消费」然后都通过。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def issue(self, approval: ChangeApproval) -> None:
        with self._session_factory() as db:
            row = ApprovalRow(
                token=approval.token,
                run_id=approval.run_id,
                diff_hash=approval.diff_hash,
                base_fingerprint=approval.base_fingerprint,
                approval_scope=_join_scope(approval.scope),
                expires_at_epoch_ms=approval.expires_at_epoch_ms,
                consumed_at_epoch_ms=approval.consumed_at_epoch_ms,
            )
            db.add(row)
            try:
                db.commit()
            except IntegrityError as error:
                db.rollback()
                raise ValueError(
                    f"令牌 {approval.token} 已存在，不能重复签发"
                ) from error

    def get(self, token: str) -> ChangeApproval | None:
        with self._session_factory() as db:
            row = db.get(ApprovalRow, token)
            if row is None:
                return None
            return _approval_from_row(row)

    def consume(self, token: str, *, now_epoch_ms: int) -> ChangeApproval:
        with self._session_factory() as db:
            # 先判存在与过期，是为了区分三种失败原因并给出准确的异常。
            # 真正的互斥由下面那条带条件的 UPDATE 保证，不依赖这次读取。
            row = db.get(ApprovalRow, token)
            if row is None:
                raise ApprovalNotFoundError(f"令牌不存在：{token}")
            if now_epoch_ms >= row.expires_at_epoch_ms:
                raise ApprovalExpiredError(
                    f"令牌 {token} 已于 {row.expires_at_epoch_ms} 过期。"
                    "过期令牌不得使用——用户当时看到的 diff 可能已经不是现在这份。"
                )
            statement = (
                update(ApprovalRow)
                .where(
                    ApprovalRow.token == token,
                    ApprovalRow.consumed_at_epoch_ms.is_(None),
                )
                .values(consumed_at_epoch_ms=now_epoch_ms)
            )
            result = db.execute(statement)
            if result.rowcount != 1:
                db.rollback()
                raise ApprovalAlreadyConsumedError(
                    f"令牌 {token} 已被消费。"
                    "审批是一次性的，同一次批准不能用来应用两次补丁。"
                )
            db.commit()
            refreshed = db.get(ApprovalRow, token)
            if refreshed is None:
                raise ApprovalNotFoundError(f"令牌在消费后消失：{token}")
            db.refresh(refreshed)
            return _approval_from_row(refreshed)


def _approval_from_row(row: ApprovalRow) -> ChangeApproval:
    return ChangeApproval(
        token=row.token,
        run_id=row.run_id,
        diff_hash=row.diff_hash,
        base_fingerprint=row.base_fingerprint,
        scope=_split_scope(row.approval_scope),
        expires_at_epoch_ms=row.expires_at_epoch_ms,
        consumed_at_epoch_ms=row.consumed_at_epoch_ms,
    )


class MySqlIdempotencyStore:
    """幂等键的 MySQL 实现。

    ``register`` 直接 INSERT，撞主键就返回 False。**不先查再插**——
    先查再插的两步之间同样有窗口，两个并发请求会都拿到 True，
    于是同一个业务动作被执行两次。让主键冲突来告诉你「已经有人登记过了」。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def register(self, key: IdempotencyKey, *, result_ref: str) -> bool:
        if not result_ref.strip():
            raise ValueError("result_ref 不能为空——否则命中幂等时无法返回既有结果")
        statement = insert(IdempotencyRow).values(
            scope=key.scope,
            key_value=key.key,
            result_ref=result_ref,
        )
        with self._session_factory() as db:
            try:
                db.execute(statement)
                db.commit()
            except IntegrityError:
                db.rollback()
                return False
            return True

    def lookup(self, key: IdempotencyKey) -> str | None:
        statement = select(IdempotencyRow.result_ref).where(
            IdempotencyRow.scope == key.scope,
            IdempotencyRow.key_value == key.key,
        )
        with self._session_factory() as db:
            return db.scalars(statement).first()


class MySqlGatewayCostStore:
    """版本化 attempt 成本记录；价格更新不会覆盖历史行。"""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def record(self, cost) -> None:
        row = GatewayCostRow(
            attempt_id=cost.attempt_id,
            request_id=cost.request_id,
            tenant_id=cost.tenant_id,
            user_id=cost.user_id,
            scene=cost.scene,
            prompt_version=cost.prompt_version,
            provider=cost.provider,
            model_tier=cost.model_tier,
            model=cost.model,
            input_tokens=cost.input_tokens,
            output_tokens=cost.output_tokens,
            cached_tokens=cost.cached_tokens,
            price_version=cost.price_version,
            total_stars=cost.total_stars,
            latency_milliseconds=cost.latency_milliseconds,
            ttft_milliseconds=cost.ttft_milliseconds,
            fallback_reason=cost.fallback_reason,
            error_class=cost.error_class,
        )
        with self._session_factory() as db:
            db.add(row)
            db.commit()

    def list_for_request(self, request_id: str):
        from codeinsight.infrastructure.model_gateway import CostRecord

        statement = (
            select(GatewayCostRow)
            .where(GatewayCostRow.request_id == request_id)
            .order_by(GatewayCostRow.attempt_id)
        )
        with self._session_factory() as db:
            return tuple(
                CostRecord(
                    request_id=row.request_id,
                    attempt_id=row.attempt_id,
                    tenant_id=row.tenant_id,
                    user_id=row.user_id,
                    scene=row.scene,
                    prompt_version=row.prompt_version,
                    provider=row.provider,
                    model_tier=row.model_tier,
                    model=row.model,
                    input_tokens=row.input_tokens,
                    output_tokens=row.output_tokens,
                    cached_tokens=row.cached_tokens,
                    price_version=row.price_version,
                    total_stars=row.total_stars,
                    latency_milliseconds=row.latency_milliseconds,
                    ttft_milliseconds=row.ttft_milliseconds,
                    fallback_reason=row.fallback_reason,
                    error_class=row.error_class,
                )
                for row in db.scalars(statement)
            )


class MySqlAgentRunStore:
    """Agent Run 事实的 MySQL 实现。与内存实现跑同一套语义测试。

    claim_run 用一条带条件的 UPDATE 加 rowcount 完成 CAS，与 ApprovalStore.consume
    同一种做法：先 SELECT 再判断再 UPDATE，会让两个 Worker 同时读到 QUEUED，
    两边都以为自己领到了。租约过期是「上一个 Worker 没了」的唯一信号，
    所以它和状态一起写在 WHERE 里，由数据库判定。
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def save_run(self, record: AgentRunRecord) -> None:
        with self._session_factory() as db:
            row = db.get(AgentRunRow, record.run_id)
            if row is None:
                row = AgentRunRow(run_id=record.run_id)
                db.add(row)
            _apply_agent_run_to_row(row, record)
            try:
                db.commit()
            except IntegrityError as error:
                db.rollback()
                raise AgentRunTurnConflictError(
                    f"turn {record.turn_id} 已经属于另一个 Agent Run。"
                    "同一条用户消息重发时应当命中已有 Run，而不是新开一条。"
                ) from error

    def get_run(self, run_id: str) -> AgentRunRecord | None:
        with self._session_factory() as db:
            row = db.get(AgentRunRow, run_id)
            if row is None:
                return None
            return _agent_run_from_row(row)

    def find_run_by_turn(self, turn_id: str) -> AgentRunRecord | None:
        statement = select(AgentRunRow).where(AgentRunRow.turn_id == turn_id)
        with self._session_factory() as db:
            row = db.scalars(statement).first()
            if row is None:
                return None
            return _agent_run_from_row(row)

    def list_open_runs(self, *, limit: int = 50) -> tuple[AgentRunRecord, ...]:
        statement = (
            select(AgentRunRow)
            .where(AgentRunRow.status.in_(sorted(OPEN_AGENT_RUN_STATUSES)))
            .order_by(AgentRunRow.updated_at_epoch_ms)
            .limit(limit)
        )
        with self._session_factory() as db:
            return tuple(_agent_run_from_row(row) for row in db.scalars(statement))

    def count_queued_runs(self) -> int:
        statement = (
            select(func.count())
            .select_from(AgentRunRow)
            .where(AgentRunRow.status == QUEUED)
        )
        with self._session_factory() as db:
            return int(db.scalar(statement) or 0)

    def claim_run(
        self, run_id: str, *, worker_id: str, lease_until_epoch_ms: int
    ) -> AgentRunRecord | None:
        """QUEUED 随时可领；RUNNING 只有在租约过期后才允许被接管。"""

        now_epoch_ms = int(time.time() * 1000)
        statement = (
            update(AgentRunRow)
            .where(
                AgentRunRow.run_id == run_id,
                or_(
                    AgentRunRow.status == QUEUED,
                    and_(
                        AgentRunRow.status == RUNNING,
                        or_(
                            AgentRunRow.lease_until_epoch_ms.is_(None),
                            AgentRunRow.lease_until_epoch_ms <= now_epoch_ms,
                        ),
                    ),
                ),
            )
            .values(
                status=RUNNING,
                worker_id=worker_id,
                lease_until_epoch_ms=lease_until_epoch_ms,
                updated_at_epoch_ms=now_epoch_ms,
            )
        )
        with self._session_factory() as db:
            result = db.execute(statement)
            if result.rowcount != 1:
                db.rollback()
                return None
            db.commit()
            row = db.get(AgentRunRow, run_id)
            if row is None:
                return None
            return _agent_run_from_row(row)


def _apply_agent_run_to_row(row: AgentRunRow, record: AgentRunRecord) -> None:
    row.turn_id = record.turn_id
    row.session_id = record.session_id
    row.task_id = record.task_id
    row.task_kind = record.task_kind
    row.status = record.status
    row.policy_version = record.policy_version
    row.idempotency_key = record.idempotency_key
    row.attempt = record.attempt
    row.max_attempts = record.max_attempts
    row.worker_id = record.worker_id
    row.lease_until_epoch_ms = record.lease_until_epoch_ms
    row.deadline_epoch_ms = record.deadline_epoch_ms
    row.error_class = record.error_class
    row.event_sequence = record.event_sequence
    row.updated_at_epoch_ms = record.updated_at_epoch_ms
    row.validation_profile = record.options.validation_profile
    row.result_limit = record.options.result_limit
    row.show_debug_reasoning = record.options.show_debug_reasoning
    row.patch_id = record.patch_id
    row.approval_token = record.approval_token


def _agent_run_from_row(row: AgentRunRow) -> AgentRunRecord:
    return AgentRunRecord(
        run_id=row.run_id,
        turn_id=row.turn_id,
        session_id=row.session_id,
        task_id=row.task_id,
        task_kind=row.task_kind,
        status=row.status,
        policy_version=row.policy_version,
        idempotency_key=row.idempotency_key,
        deadline_epoch_ms=row.deadline_epoch_ms,
        updated_at_epoch_ms=row.updated_at_epoch_ms,
        attempt=row.attempt,
        max_attempts=row.max_attempts,
        worker_id=row.worker_id,
        lease_until_epoch_ms=row.lease_until_epoch_ms,
        error_class=row.error_class,
        event_sequence=row.event_sequence,
        options=RunRequestOptions(
            validation_profile=row.validation_profile,
            result_limit=row.result_limit,
            show_debug_reasoning=row.show_debug_reasoning,
        ),
        patch_id=row.patch_id,
        approval_token=row.approval_token,
    )


def truncate_all_tables(engine: Engine) -> None:
    """清空全部业务表。**仅供集成测试在用例之间隔离状态。**

    用 DELETE 而不是 TRUNCATE：TRUNCATE 在 MySQL 里是 DDL，会隐式提交
    事务，让「整个测试跑在一个可回滚事务里」这种隔离方式失效。
    """
    from codeinsight.infrastructure.db.schema import Base

    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(delete(table))
