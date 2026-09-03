"""Session / Goal / Run 状态机的回归测试。

这些测试保护的是 Step 2 的核心承诺：**状态机不允许走到不该到的地方**。
其中「终态不可迁出」和「步数上限不可越过」两条如果被放松，
就等于允许一个已回滚的任务被重新执行，或允许 Agent 无限消耗预算。
"""

from __future__ import annotations

import pytest

from codeinsight.domain.change import (
    CHECKING,
    COMPLETED,
    FAILED,
    GOAL_ACTIVE,
    LEGAL_RUN_TRANSITIONS,
    MODE_ISOLATED_WRITE,
    MODE_READ_ONLY,
    REVIEW_REQUIRED,
    ROLLED_BACK,
    RUNNING,
    STUCK,
    SUPPORTED_RUN_STATUSES,
    TERMINAL_RUN_STATUSES,
    WAITING_APPROVAL,
    WAITING_USER,
    ChangeApproval,
    CodeGoal,
    ConversationSession,
    GoalRouting,
    IllegalTransitionError,
    RunSnapshot,
    TenantScope,
    WorkspaceRun,
    assert_legal_transition,
    is_interruptible,
)

SCOPE = TenantScope()


def _run(**overrides: object) -> RunSnapshot:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "session_id": "sess-1",
        "goal_id": "goal-1",
        "scope": SCOPE,
    }
    defaults.update(overrides)
    return RunSnapshot(**defaults)  # type: ignore[arg-type]


def _goal(**overrides: object) -> CodeGoal:
    defaults: dict[str, object] = {
        "goal_id": "goal-1",
        "session_id": "sess-1",
        "scope": SCOPE,
        "repo_id": "repo-1",
        "task_type": "change",
        "user_goal": "把重试次数改成可配置",
    }
    defaults.update(overrides)
    return CodeGoal(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 状态机结构
# ---------------------------------------------------------------------------


def test_every_status_has_a_transition_entry() -> None:
    """漏掉一条会让 assert_legal_transition 抛 KeyError 而不是给出解释。"""
    assert set(LEGAL_RUN_TRANSITIONS) == set(SUPPORTED_RUN_STATUSES)


def test_terminal_states_have_no_outgoing_transitions() -> None:
    for status in TERMINAL_RUN_STATUSES:
        assert LEGAL_RUN_TRANSITIONS[status] == frozenset()


def test_transition_targets_are_all_known_statuses() -> None:
    for targets in LEGAL_RUN_TRANSITIONS.values():
        for target in targets:
            assert target in SUPPORTED_RUN_STATUSES


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RUNNING, WAITING_APPROVAL),
        (RUNNING, CHECKING),
        (WAITING_APPROVAL, RUNNING),
        (WAITING_APPROVAL, ROLLED_BACK),
        (CHECKING, RUNNING),
        (CHECKING, COMPLETED),
        (CHECKING, REVIEW_REQUIRED),
        (REVIEW_REQUIRED, ROLLED_BACK),
        (STUCK, ROLLED_BACK),
    ],
)
def test_legal_transitions_are_accepted(current: str, target: str) -> None:
    assert_legal_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (WAITING_USER, WAITING_APPROVAL),  # 用户还没回话就跳去等审批
        (WAITING_USER, COMPLETED),  # 等用户输入时不能自己完成
        (CHECKING, WAITING_APPROVAL),  # 检查阶段不该回头要审批
        (RUNNING, ROLLED_BACK),  # 回滚必须经由明确的失败或拒绝路径
    ],
)
def test_illegal_transitions_are_refused(current: str, target: str) -> None:
    with pytest.raises(IllegalTransitionError):
        assert_legal_transition(current, target)


@pytest.mark.parametrize("terminal", sorted(TERMINAL_RUN_STATUSES))
def test_terminal_state_cannot_transition_out(terminal: str) -> None:
    """最重要的一条：已完成或已回滚的 Run 不允许被重新执行。"""
    run = _run(status=terminal, stuck_reason=None)
    with pytest.raises(IllegalTransitionError, match="终态"):
        run.transition_to(RUNNING)


def test_unknown_status_is_refused() -> None:
    with pytest.raises(IllegalTransitionError, match="未知"):
        assert_legal_transition(RUNNING, "ALMOST_DONE")


# ---------------------------------------------------------------------------
# stateVersion
# ---------------------------------------------------------------------------


def test_transition_increments_state_version() -> None:
    """每次状态变更都必须递增版本号，否则 CAS 无从判断。"""
    run = _run()
    moved = run.transition_to(CHECKING)
    assert moved.state_version == run.state_version + 1
    assert moved.status == CHECKING


def test_step_increments_state_version() -> None:
    run = _run()
    stepped = run.with_step(tool_name="search_repository")
    assert stepped.state_version == run.state_version + 1
    assert stepped.step_count == 1


def test_snapshots_are_immutable() -> None:
    run = _run()
    with pytest.raises(Exception):
        run.status = COMPLETED  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 步数与修复次数上限
# ---------------------------------------------------------------------------


def test_step_over_limit_becomes_stuck_not_error() -> None:
    """超限是可预期的业务结果，应转 STUCK 并说明原因，而不是抛异常。"""
    run = _run(max_steps=2, step_count=2)
    result = run.with_step()
    assert result.status == STUCK
    assert result.stuck_reason is not None
    assert "上限" in result.stuck_reason


def test_step_count_cannot_be_constructed_over_limit() -> None:
    with pytest.raises(ValueError, match="超过上限"):
        _run(max_steps=3, step_count=4)


def test_stuck_requires_a_reason() -> None:
    """没有原因的 STUCK 无法排障，构造时就要拦住。"""
    with pytest.raises(ValueError, match="STUCK"):
        _run(status=STUCK)


def test_transition_to_stuck_without_reason_is_refused() -> None:
    run = _run()
    with pytest.raises(ValueError, match="原因"):
        run.transition_to(STUCK)


def test_repair_attempts_exhausted_goes_to_review() -> None:
    """修复次数用尽必须转人工，不能无限自动重试。"""
    run = _run(status=CHECKING, max_repair_attempts=2, repair_attempts=2)
    result = run.with_repair_attempt()
    assert result.status == REVIEW_REQUIRED


def test_repair_attempt_within_limit_stays_put() -> None:
    run = _run(status=CHECKING, max_repair_attempts=2, repair_attempts=0)
    result = run.with_repair_attempt()
    assert result.status == CHECKING
    assert result.repair_attempts == 1
    assert result.repairs_remaining == 1


def test_terminal_run_cannot_take_a_step() -> None:
    run = _run(status=COMPLETED)
    with pytest.raises(IllegalTransitionError):
        run.with_step()


# ---------------------------------------------------------------------------
# 取消语义分级（增补 R-1）
# ---------------------------------------------------------------------------


def test_read_only_tool_is_interruptible() -> None:
    assert is_interruptible("search_repository") is True
    assert _run(active_tool="read_file").can_cancel_immediately() is True


def test_write_class_tool_is_not_interruptible() -> None:
    """写类工具中途打断会留下半应用的补丁，必须等它结束并对账。"""
    assert is_interruptible("apply_patch_isolated") is False
    assert _run(active_tool="apply_patch_isolated").can_cancel_immediately() is False


def test_unregistered_tool_defaults_to_not_interruptible() -> None:
    """默认方向很重要：猜错的代价是留下说不清的中间状态。"""
    assert is_interruptible("some_new_tool") is False
    assert _run(active_tool="some_new_tool").can_cancel_immediately() is False


def test_idle_run_can_always_be_cancelled() -> None:
    assert _run(active_tool=None).can_cancel_immediately() is True


def test_transition_clears_active_tool() -> None:
    """状态迁移后旧的 active_tool 不该残留，否则取消判断会看错工具。"""
    run = _run(active_tool="apply_patch_isolated")
    assert run.transition_to(CHECKING).active_tool is None


# ---------------------------------------------------------------------------
# Goal 与模式
# ---------------------------------------------------------------------------


def test_read_only_is_the_default_mode() -> None:
    """从 1.0 继承的地基：默认不写。"""
    assert _goal().mode == MODE_READ_ONLY
    assert _goal().allows_write is False


def test_write_mode_requires_validation_profile() -> None:
    """允许写但没有固定检查，等于放开了没有验证的修改。"""
    with pytest.raises(ValueError, match="validation_profile"):
        _goal(mode=MODE_ISOLATED_WRITE)


def test_write_mode_with_profile_is_accepted() -> None:
    goal = _goal(mode=MODE_ISOLATED_WRITE, validation_profile="pytest-fast")
    assert goal.allows_write is True


def test_unknown_task_type_is_refused() -> None:
    with pytest.raises(ValueError, match="task_type"):
        _goal(task_type="refactor_everything")


def test_goal_defaults_to_active() -> None:
    assert _goal().status == GOAL_ACTIVE


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


def test_session_turns_must_be_contiguous() -> None:
    session = ConversationSession(session_id="s", scope=SCOPE, repo_id="r")
    session = session.with_turn("user", "改一下重试逻辑")
    session = session.with_turn("assistant", "已定位到 3 处")
    assert len(session.recent_turns) == 2
    assert session.recent_turns[1].sequence == 2


def test_session_with_turn_returns_new_instance() -> None:
    original = ConversationSession(session_id="s", scope=SCOPE, repo_id="r")
    updated = original.with_turn("user", "问题")
    assert original.recent_turns == ()
    assert len(updated.recent_turns) == 1


# ---------------------------------------------------------------------------
# Goal 路由
# ---------------------------------------------------------------------------


def test_continue_routing_must_name_a_goal() -> None:
    """默认继续当前 Goal，但必须指名是哪一个。"""
    with pytest.raises(ValueError, match="goal_id"):
        GoalRouting(decision=GoalRouting.CONTINUE)


def test_new_routing_must_not_carry_existing_goal() -> None:
    with pytest.raises(ValueError, match="既有"):
        GoalRouting(decision=GoalRouting.NEW, goal_id="goal-1")


def test_new_routing_without_goal_is_accepted() -> None:
    routing = GoalRouting(decision=GoalRouting.NEW, reason="用户换了新问题")
    assert routing.goal_id is None


# ---------------------------------------------------------------------------
# 审批令牌的范围绑定
# ---------------------------------------------------------------------------


def _approval(**overrides: object) -> ChangeApproval:
    defaults: dict[str, object] = {
        "token": "tok-1",
        "run_id": "run-1",
        "diff_hash": "hash-a",
        "base_fingerprint": "base-1",
        "scope": ("src/shop/pricing.py",),
        "expires_at_epoch_ms": 2_000,
    }
    defaults.update(overrides)
    return ChangeApproval(**defaults)  # type: ignore[arg-type]


def test_approval_valid_when_everything_matches() -> None:
    approval = _approval()
    assert approval.is_valid_for(
        run_id="run-1",
        diff_hash="hash-a",
        base_fingerprint="base-1",
        now_epoch_ms=1_000,
    )


@pytest.mark.parametrize(
    ("run_id", "diff_hash", "base_fingerprint", "now"),
    [
        ("run-2", "hash-a", "base-1", 1_000),  # 跨任务复用
        ("run-1", "hash-b", "base-1", 1_000),  # 批准 A 却应用 B
        ("run-1", "hash-a", "base-2", 1_000),  # 基线已漂移
        ("run-1", "hash-a", "base-1", 2_000),  # 已过期
    ],
)
def test_approval_invalid_when_any_binding_differs(
    run_id: str, diff_hash: str, base_fingerprint: str, now: int
) -> None:
    """四个绑定字段缺一不可，任何一项不符即失效。"""
    assert not _approval().is_valid_for(
        run_id=run_id,
        diff_hash=diff_hash,
        base_fingerprint=base_fingerprint,
        now_epoch_ms=now,
    )


def test_consumed_approval_is_never_valid() -> None:
    approval = _approval(consumed_at_epoch_ms=500)
    assert approval.is_consumed
    assert not approval.is_valid_for(
        run_id="run-1",
        diff_hash="hash-a",
        base_fingerprint="base-1",
        now_epoch_ms=1_000,
    )


def test_approval_must_declare_scope() -> None:
    with pytest.raises(ValueError, match="范围"):
        _approval(scope=())


# ---------------------------------------------------------------------------
# Workspace 隔离
# ---------------------------------------------------------------------------


def test_workspace_path_cannot_equal_source_repo() -> None:
    """原仓库默认只读是从 1.0 继承的硬边界，构造时就要拦住。"""
    with pytest.raises(ValueError, match="只读"):
        WorkspaceRun(
            workspace_id="ws-1",
            run_id="run-1",
            source_repo_path="/repos/shop",
            workspace_path="/repos/shop",
        )


def test_workspace_without_checkpoint_has_none() -> None:
    workspace = WorkspaceRun(
        workspace_id="ws-1",
        run_id="run-1",
        source_repo_path="/repos/shop",
        workspace_path="/tmp/ws-1",
    )
    assert workspace.latest_checkpoint is None


# ---------------------------------------------------------------------------
# 租户维度（增补 R-3）
# ---------------------------------------------------------------------------


def test_tenant_scope_has_single_valued_defaults() -> None:
    """当前恒为单值、不做鉴权，但维度必须已在数据模型里。"""
    scope = TenantScope()
    assert scope.tenant_id == "default"
    assert scope.user_id == "local"


def test_tenant_scope_rejects_blank_values() -> None:
    with pytest.raises(ValueError):
        TenantScope(tenant_id="  ")


# ---------------------------------------------------------------------------
# 完整流程串联
# ---------------------------------------------------------------------------


def test_happy_path_write_flow() -> None:
    """RUNNING → 等审批 → 应用 → 检查 → 完成。"""
    run = _run()
    run = run.with_step(tool_name="search_repository")
    run = run.with_step(tool_name="generate_patch")
    run = run.transition_to(WAITING_APPROVAL)
    assert run.can_cancel_immediately() is True

    run = run.transition_to(RUNNING)
    run = run.with_step(tool_name="apply_patch_isolated")
    assert run.can_cancel_immediately() is False

    run = run.transition_to(CHECKING)
    run = run.transition_to(COMPLETED)
    assert run.is_terminal
    assert run.state_version == 8


def test_check_failure_repair_then_review_flow() -> None:
    """检查失败 → 有限修复两轮 → 用尽后转人工。"""
    run = _run(status=CHECKING, max_repair_attempts=2)
    run = run.with_repair_attempt()
    run = run.transition_to(RUNNING)
    run = run.transition_to(CHECKING)
    run = run.with_repair_attempt()
    assert run.repairs_remaining == 0

    run = run.with_repair_attempt()
    assert run.status == REVIEW_REQUIRED


def test_approval_denied_leads_to_rollback() -> None:
    run = _run(status=WAITING_APPROVAL)
    run = run.transition_to(ROLLED_BACK)
    assert run.is_terminal
    with pytest.raises(IllegalTransitionError):
        run.transition_to(FAILED)
