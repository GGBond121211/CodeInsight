"""MySQL Store 的集成测试。未配置测试库时整体跳过（见 conftest.py）。

**这套测试与 tests/unit/infrastructure/test_run_store.py 断言同一组语义。**
两处都过才说明「换存储不改变行为」这件事真的成立。

内存实现靠进程内的一把锁，MySQL 实现靠数据库的行锁与唯一约束。
后者在多进程、多实例部署下依然成立——这正是从文件 Store 换成数据库的
核心理由。这里验证的就是这个「依然成立」。
"""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session, sessionmaker

from codeinsight.domain.change import (
    CHECKING,
    COMPLETED,
    GOAL_ABANDONED,
    MODE_ISOLATED_WRITE,
    RUNNING,
    ChangeApproval,
    CodeGoal,
    ConversationSession,
    RunSnapshot,
    StateVersionConflictError,
    TenantScope,
)
from codeinsight.domain.ports import (
    ApprovalStore,
    AuditLog,
    EventLog,
    IdempotencyStore,
    RunStore,
    SessionStore,
)
from codeinsight.domain.trace import (
    APPROVAL_GRANTED,
    RUN_STARTED,
    TOOL_CALLED,
    AuditRecord,
    IdempotencyKey,
    RunEvent,
    TraceContext,
)
from codeinsight.infrastructure.db.stores import (
    MySqlApprovalStore,
    MySqlAuditLog,
    MySqlEventLog,
    MySqlIdempotencyStore,
    MySqlRunStore,
    MySqlSessionStore,
)
from codeinsight.infrastructure.event_log import EventSequenceError
from codeinsight.infrastructure.run_store import (
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
)

SCOPE = TenantScope()
TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


def _run(**overrides: object) -> RunSnapshot:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "session_id": "sess-1",
        "goal_id": "goal-1",
        "scope": SCOPE,
    }
    defaults.update(overrides)
    return RunSnapshot(**defaults)  # type: ignore[arg-type]


def _goal(goal_id: str = "goal-1", **overrides: object) -> CodeGoal:
    defaults: dict[str, object] = {
        "goal_id": goal_id,
        "session_id": "sess-1",
        "scope": SCOPE,
        "repo_id": "repo-1",
        "task_type": "change",
        "user_goal": "把重试次数改成可配置",
    }
    defaults.update(overrides)
    return CodeGoal(**defaults)  # type: ignore[arg-type]


def _approval(**overrides: object) -> ChangeApproval:
    defaults: dict[str, object] = {
        "token": "tok-1",
        "run_id": "run-1",
        "diff_hash": "hash-a",
        "base_fingerprint": "base-1",
        "scope": ("src/shop/pricing.py",),
        "expires_at_epoch_ms": 5_000,
    }
    defaults.update(overrides)
    return ChangeApproval(**defaults)  # type: ignore[arg-type]


def _event(sequence: int, event_type: str = TOOL_CALLED, **overrides: object) -> RunEvent:
    defaults: dict[str, object] = {
        "event_id": f"ev-{sequence}",
        "run_id": "run-1",
        "sequence": sequence,
        "event_type": event_type,
        "occurred_at_epoch_ms": 1_000 + sequence,
    }
    defaults.update(overrides)
    return RunEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol 一致性
# ---------------------------------------------------------------------------


def test_mysql_stores_satisfy_their_protocols(
    session_factory: sessionmaker[Session],
) -> None:
    assert isinstance(MySqlSessionStore(session_factory), SessionStore)
    assert isinstance(MySqlRunStore(session_factory), RunStore)
    assert isinstance(MySqlEventLog(session_factory), EventLog)
    assert isinstance(MySqlAuditLog(session_factory), AuditLog)
    assert isinstance(MySqlApprovalStore(session_factory), ApprovalStore)
    assert isinstance(MySqlIdempotencyStore(session_factory), IdempotencyStore)


# ---------------------------------------------------------------------------
# Run CAS —— 与内存实现断言同一组语义
# ---------------------------------------------------------------------------


def test_create_then_get_roundtrip(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.status == RUNNING
    assert loaded.state_version == 1


def test_get_unknown_run_returns_none(session_factory: sessionmaker[Session]) -> None:
    assert MySqlRunStore(session_factory).get_run("nope") is None


def test_duplicate_create_is_refused(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    with pytest.raises(StateVersionConflictError, match="已存在"):
        store.create_run(_run())


def test_save_with_matching_version_succeeds(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    store.save_run(_run().transition_to(CHECKING), expected_version=1)
    reloaded = store.get_run("run-1")
    assert reloaded is not None
    assert reloaded.status == CHECKING
    assert reloaded.state_version == 2


def test_save_with_stale_version_is_refused(session_factory: sessionmaker[Session]) -> None:
    """并发 resume 里后到的那个必须失败，而不是覆盖。"""
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    store.save_run(_run().transition_to(CHECKING), expected_version=1)
    with pytest.raises(StateVersionConflictError, match="冲突"):
        store.save_run(_run().transition_to(COMPLETED), expected_version=1)


def test_stale_write_does_not_change_stored_state(
    session_factory: sessionmaker[Session],
) -> None:
    """验收条件的直接检验：失败的写入必须**没有**改动库里的状态。"""
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    store.save_run(_run().transition_to(COMPLETED), expected_version=1)
    with pytest.raises(StateVersionConflictError):
        store.save_run(_run().transition_to(CHECKING), expected_version=1)
    final = store.get_run("run-1")
    assert final is not None
    assert final.status == COMPLETED
    assert final.state_version == 2


def test_save_must_increment_version(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    store.create_run(_run())
    with pytest.raises(ValueError, match="必须大于"):
        store.save_run(_run(), expected_version=1)


def test_save_unknown_run_is_refused(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    with pytest.raises(StateVersionConflictError, match="不存在"):
        store.save_run(_run(state_version=2), expected_version=1)


def test_cas_where_clause_is_actually_on_the_update(
    session_factory: sessionmaker[Session],
) -> None:
    """绕过应用层的前置校验，验证 UPDATE 语句本身带了版本条件。

    做法：两个独立 Session 各自读到 version 1，一个先提交成功，
    另一个提交时 rowcount 为 0，SQLAlchemy 抛 StaleDataError。
    如果 version_id_col 没生效，第二个会静默覆盖第一个。
    """
    from codeinsight.infrastructure.db.schema import RunRow

    store = MySqlRunStore(session_factory)
    store.create_run(_run())

    first = session_factory()
    second = session_factory()
    try:
        row_a = first.get(RunRow, "run-1")
        row_b = second.get(RunRow, "run-1")
        assert row_a is not None and row_b is not None
        assert row_a.state_version == 1
        assert row_b.state_version == 1

        row_a.status = CHECKING
        row_a.state_version = 2
        first.commit()

        row_b.status = COMPLETED
        row_b.state_version = 2
        from sqlalchemy.orm.exc import StaleDataError

        with pytest.raises(StaleDataError):
            second.commit()
        second.rollback()
    finally:
        first.close()
        second.close()

    final = store.get_run("run-1")
    assert final is not None
    assert final.status == CHECKING


def test_list_runs_for_goal_is_deterministic(session_factory: sessionmaker[Session]) -> None:
    store = MySqlRunStore(session_factory)
    store.create_run(_run(run_id="run-2"))
    store.create_run(_run(run_id="run-1"))
    store.create_run(_run(run_id="run-3", goal_id="goal-2"))
    ids: list[str] = []
    for run in store.list_runs_for_goal("goal-1"):
        ids.append(run.run_id)
    assert ids == ["run-1", "run-2"]


def test_all_run_fields_survive_a_roundtrip(session_factory: sessionmaker[Session]) -> None:
    """字段落库再读回必须完全一致——漏映射一个字段不会报错，只会静默丢值。"""
    store = MySqlRunStore(session_factory)
    original = _run(
        status="STUCK",
        step_count=3,
        max_steps=8,
        next_action="等待用户确认范围",
        deadline_epoch_ms=9_999_999_999_999,
        token_budget=120_000,
        tokens_used=4_321,
        repair_attempts=1,
        max_repair_attempts=2,
        stuck_reason="连续三次工具调用未改变状态",
        active_tool="apply_patch_isolated",
    )
    store.create_run(original)
    assert store.get_run("run-1") == original


# ---------------------------------------------------------------------------
# Session / Goal
# ---------------------------------------------------------------------------


def test_session_roundtrip(session_factory: sessionmaker[Session]) -> None:
    store = MySqlSessionStore(session_factory)
    session = ConversationSession(
        session_id="sess-1", scope=SCOPE, repo_id="repo-1", summary="已定位到 3 处"
    )
    store.save_session(session)
    loaded = store.get_session("sess-1")
    assert loaded is not None
    assert loaded.summary == "已定位到 3 处"


def test_save_session_is_idempotent(session_factory: sessionmaker[Session]) -> None:
    """重复保存同一个 session 应当是更新，不是主键冲突。"""
    store = MySqlSessionStore(session_factory)
    session = ConversationSession(session_id="sess-1", scope=SCOPE, repo_id="repo-1")
    store.save_session(session)
    store.save_session(session.with_active_goal("goal-1"))
    loaded = store.get_session("sess-1")
    assert loaded is not None
    assert loaded.active_goal_id == "goal-1"


def test_same_session_continues_the_same_goal(
    session_factory: sessionmaker[Session],
) -> None:
    """验收条件：同一 session 的第二次提问继续同一个 Goal。"""
    store = MySqlSessionStore(session_factory)
    store.save_goal(_goal("goal-1"))
    store.save_session(
        ConversationSession(
            session_id="sess-1", scope=SCOPE, repo_id="repo-1"
        ).with_active_goal("goal-1")
    )
    reloaded = store.get_session("sess-1")
    assert reloaded is not None
    assert reloaded.active_goal_id == "goal-1"
    assert store.get_goal("goal-1") is not None


def test_list_active_goals_excludes_finished_ones(
    session_factory: sessionmaker[Session],
) -> None:
    store = MySqlSessionStore(session_factory)
    store.save_goal(_goal("goal-1"))
    store.save_goal(_goal("goal-2", status=GOAL_ABANDONED))
    store.save_goal(_goal("goal-3", session_id="sess-2"))
    active: list[str] = []
    for goal in store.list_active_goals("sess-1"):
        active.append(goal.goal_id)
    assert active == ["goal-1"]


def test_goal_target_scope_survives_roundtrip(
    session_factory: sessionmaker[Session],
) -> None:
    """多值字段用换行分隔存 TEXT；路径里的逗号不该把它拆坏。"""
    store = MySqlSessionStore(session_factory)
    goal = _goal(
        mode=MODE_ISOLATED_WRITE,
        validation_profile="pytest-fast",
        target_scope=("src/a,b.py", "src/shop/pricing.py"),
    )
    store.save_goal(goal)
    loaded = store.get_goal("goal-1")
    assert loaded is not None
    assert loaded.target_scope == ("src/a,b.py", "src/shop/pricing.py")


def test_chinese_text_survives_roundtrip(session_factory: sessionmaker[Session]) -> None:
    """utf8mb4 的实际检验。字符集配错这里就会是乱码或截断。"""
    store = MySqlSessionStore(session_factory)
    goal = _goal(user_goal="把「重试次数」改成可配置项 🔧 兼容旧调用方")
    store.save_goal(goal)
    loaded = store.get_goal("goal-1")
    assert loaded is not None
    assert loaded.user_goal == "把「重试次数」改成可配置项 🔧 兼容旧调用方"


# ---------------------------------------------------------------------------
# 事件日志
# ---------------------------------------------------------------------------


def test_events_append_and_read_in_order(session_factory: sessionmaker[Session]) -> None:
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    log.append(_event(2))
    log.append(_event(3))
    assert log.next_sequence("run-1") == 4
    sequences: list[int] = []
    for event in log.read_events("run-1"):
        sequences.append(event.sequence)
    assert sequences == [1, 2, 3]


def test_gap_in_sequence_is_refused(session_factory: sessionmaker[Session]) -> None:
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    with pytest.raises(EventSequenceError, match="连续"):
        log.append(_event(3))


def test_duplicate_sequence_is_refused(session_factory: sessionmaker[Session]) -> None:
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    with pytest.raises(EventSequenceError):
        log.append(_event(1, event_id="ev-dup"))


def test_read_after_sequence_resumes_from_position(
    session_factory: sessionmaker[Session],
) -> None:
    """SSE 断线重连（增补 R-1）：客户端回传 Last-Event-ID，服务端据此续传。"""
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    log.append(_event(2))
    log.append(_event(3))
    resumed = log.read_events("run-1", after_sequence=1)
    assert len(resumed) == 2
    assert resumed[0].sequence == 2


def test_runs_have_independent_sequences(session_factory: sessionmaker[Session]) -> None:
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    log.append(_event(1, RUN_STARTED, run_id="run-2", event_id="ev-b1"))
    assert log.count("run-1") == 1
    assert log.count("run-2") == 1


def test_event_payload_and_trace_survive_roundtrip(
    session_factory: sessionmaker[Session],
) -> None:
    log = MySqlEventLog(session_factory)
    trace = TraceContext(trace_id=TRACE_ID, span_id=SPAN_ID)
    log.append(
        _event(
            1,
            RUN_STARTED,
            payload={"task_type": "change", "tool_name": "读取文件"},
            trace=trace,
        )
    )
    loaded = log.read_events("run-1")[0]
    assert loaded.payload == {"task_type": "change", "tool_name": "读取文件"}
    assert loaded.trace is not None
    assert loaded.trace.trace_id == TRACE_ID


def test_event_without_trace_reads_back_as_none(
    session_factory: sessionmaker[Session],
) -> None:
    log = MySqlEventLog(session_factory)
    log.append(_event(1, RUN_STARTED))
    assert log.read_events("run-1")[0].trace is None


# ---------------------------------------------------------------------------
# 审计流
# ---------------------------------------------------------------------------


def _audit(**overrides: object) -> AuditRecord:
    defaults: dict[str, object] = {
        "audit_id": "aud-1",
        "run_id": "run-1",
        "event_type": APPROVAL_GRANTED,
        "actor": "local",
        "occurred_at_epoch_ms": 1_000,
        "subject": "hash-a",
        "outcome": "granted",
    }
    defaults.update(overrides)
    return AuditRecord(**defaults)  # type: ignore[arg-type]


def test_audit_records_roundtrip(session_factory: sessionmaker[Session]) -> None:
    log = MySqlAuditLog(session_factory)
    log.record(_audit())
    records = log.read_records("run-1")
    assert len(records) == 1
    assert records[0].outcome == "granted"


def test_duplicate_audit_id_is_refused(session_factory: sessionmaker[Session]) -> None:
    log = MySqlAuditLog(session_factory)
    log.record(_audit())
    with pytest.raises(ValueError, match="已存在"):
        log.record(_audit())


def test_audit_log_exposes_no_delete_method(session_factory: sessionmaker[Session]) -> None:
    """审计记录的价值在于它一直在；提供清理接口早晚会有人调用。"""
    log = MySqlAuditLog(session_factory)
    assert not hasattr(log, "delete")
    assert not hasattr(log, "purge")


def test_audit_and_events_are_stored_separately(
    session_factory: sessionmaker[Session],
) -> None:
    """写审计不会出现在事件表里，反之亦然（增补 R-5）。"""
    events = MySqlEventLog(session_factory)
    audits = MySqlAuditLog(session_factory)
    events.append(_event(1, APPROVAL_GRANTED))
    audits.record(_audit())
    assert events.count("run-1") == 1
    assert len(audits.read_records("run-1")) == 1


# ---------------------------------------------------------------------------
# 审批一次性消费
# ---------------------------------------------------------------------------


def test_approval_consume_marks_it_used(session_factory: sessionmaker[Session]) -> None:
    store = MySqlApprovalStore(session_factory)
    store.issue(_approval())
    consumed = store.consume("tok-1", now_epoch_ms=1_000)
    assert consumed.consumed_at_epoch_ms == 1_000
    persisted = store.get("tok-1")
    assert persisted is not None
    assert persisted.is_consumed


def test_second_consume_is_refused(session_factory: sessionmaker[Session]) -> None:
    """「approval bypass 为 0」这条硬门槛的直接检验。"""
    store = MySqlApprovalStore(session_factory)
    store.issue(_approval())
    store.consume("tok-1", now_epoch_ms=1_000)
    with pytest.raises(ApprovalAlreadyConsumedError, match="一次性"):
        store.consume("tok-1", now_epoch_ms=1_100)


def test_expired_approval_is_refused(session_factory: sessionmaker[Session]) -> None:
    store = MySqlApprovalStore(session_factory)
    store.issue(_approval(expires_at_epoch_ms=1_000))
    with pytest.raises(ApprovalExpiredError, match="过期"):
        store.consume("tok-1", now_epoch_ms=1_000)


def test_unknown_token_is_refused(session_factory: sessionmaker[Session]) -> None:
    with pytest.raises(ApprovalNotFoundError):
        MySqlApprovalStore(session_factory).consume("nope", now_epoch_ms=1_000)


def test_duplicate_issue_is_refused(session_factory: sessionmaker[Session]) -> None:
    store = MySqlApprovalStore(session_factory)
    store.issue(_approval())
    with pytest.raises(ValueError, match="重复签发"):
        store.issue(_approval())


def test_approval_bindings_survive_roundtrip(
    session_factory: sessionmaker[Session],
) -> None:
    """四个绑定字段落库后必须完好，否则范围校验会基于错的值通过。"""
    store = MySqlApprovalStore(session_factory)
    store.issue(_approval(scope=("src/a.py", "src/b.py")))
    loaded = store.get("tok-1")
    assert loaded is not None
    assert loaded.scope == ("src/a.py", "src/b.py")
    assert loaded.is_valid_for(
        run_id="run-1",
        diff_hash="hash-a",
        base_fingerprint="base-1",
        now_epoch_ms=1_000,
    )


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


def test_first_register_returns_true(session_factory: sessionmaker[Session]) -> None:
    store = MySqlIdempotencyStore(session_factory)
    key = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    assert store.register(key, result_ref="patch-1") is True


def test_second_register_returns_false(session_factory: sessionmaker[Session]) -> None:
    """由主键冲突判定，而不是先查再插——先查再插的窗口里两个请求都会拿到 True。"""
    store = MySqlIdempotencyStore(session_factory)
    key = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    store.register(key, result_ref="patch-1")
    assert store.register(key, result_ref="patch-1") is False
    assert store.lookup(key) == "patch-1"


def test_different_patches_get_independent_keys(
    session_factory: sessionmaker[Session],
) -> None:
    store = MySqlIdempotencyStore(session_factory)
    first = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    second = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-b")
    assert store.register(first, result_ref="patch-1") is True
    assert store.register(second, result_ref="patch-2") is True


def test_same_key_in_different_scopes_does_not_collide(
    session_factory: sessionmaker[Session],
) -> None:
    """复合主键 (scope, key_value) 的实际检验。"""
    store = MySqlIdempotencyStore(session_factory)
    request_key = IdempotencyKey(scope="request", key="same")
    action_key = IdempotencyKey(scope="action", key="same")
    assert store.register(request_key, result_ref="a") is True
    assert store.register(action_key, result_ref="b") is True
    assert store.lookup(request_key) == "a"
    assert store.lookup(action_key) == "b"


def test_lookup_unknown_key_returns_none(session_factory: sessionmaker[Session]) -> None:
    store = MySqlIdempotencyStore(session_factory)
    assert store.lookup(IdempotencyKey(scope="action", key="nope")) is None


# ---------------------------------------------------------------------------
# 端到端：重复 resume 不改变已完成状态
# ---------------------------------------------------------------------------


def test_repeated_resume_does_not_reapply(session_factory: sessionmaker[Session]) -> None:
    """Step 2 的核心验收条件，四个组件一起验。

    场景：一次写类任务已经批准、应用、完成。之后同一个请求被重发三次。
    正确结果是：状态不变、审批不能再用、幂等键不再放行、事件不重复。
    """
    runs = MySqlRunStore(session_factory)
    approvals = MySqlApprovalStore(session_factory)
    keys = MySqlIdempotencyStore(session_factory)
    events = MySqlEventLog(session_factory)

    runs.create_run(_run())
    approvals.issue(_approval())
    action_key = IdempotencyKey.for_action(
        goal_id="goal-1", action="apply_patch", fingerprint="hash-a"
    )

    # 第一次：正常走完
    approvals.consume("tok-1", now_epoch_ms=1_000)
    assert keys.register(action_key, result_ref="patch-1") is True
    events.append(_event(1, RUN_STARTED))
    runs.save_run(_run().transition_to(COMPLETED), expected_version=1)

    # 之后重发三次，每一道闸门都必须挡住
    for _ in range(3):
        with pytest.raises(ApprovalAlreadyConsumedError):
            approvals.consume("tok-1", now_epoch_ms=1_100)
        assert keys.register(action_key, result_ref="patch-1") is False
        with pytest.raises(StateVersionConflictError):
            runs.save_run(_run().transition_to(COMPLETED), expected_version=1)

    final = runs.get_run("run-1")
    assert final is not None
    assert final.status == COMPLETED
    assert final.state_version == 2
    assert events.count("run-1") == 1
