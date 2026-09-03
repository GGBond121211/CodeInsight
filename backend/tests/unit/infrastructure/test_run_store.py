"""Store 内存实现的回归测试。

**这套测试是 Store 契约的定义**。MySQL 实现落地后必须复用它——
任何在内存实现里不成立的语义，在 MySQL 实现里也不许成立。

最核心的两条：
    1. stateVersion CAS —— 防止并发 resume 把补丁应用两次
    2. 审批一次性消费 —— 「approval bypass 为 0」这条硬门槛的基础
"""

from __future__ import annotations

import threading

import pytest

from codeinsight.domain.change import (
    CHECKING,
    COMPLETED,
    GOAL_ABANDONED,
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
    EventLog,
    IdempotencyStore,
    RunStore,
    SessionStore,
)
from codeinsight.domain.trace import IdempotencyKey
from codeinsight.infrastructure.event_log import InMemoryEventLog
from codeinsight.infrastructure.run_store import (
    ApprovalAlreadyConsumedError,
    ApprovalExpiredError,
    ApprovalNotFoundError,
    InMemoryApprovalStore,
    InMemoryIdempotencyStore,
    InMemoryRunStore,
    InMemorySessionStore,
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


def _goal(goal_id: str = "goal-1", **overrides: object) -> CodeGoal:
    defaults: dict[str, object] = {
        "goal_id": goal_id,
        "session_id": "sess-1",
        "scope": SCOPE,
        "repo_id": "repo-1",
        "task_type": "change",
        "user_goal": "改一下重试逻辑",
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


# ---------------------------------------------------------------------------
# Protocol 一致性
#
# runtime_checkable Protocol 只检查方法是否存在，不检查签名。这组测试的
# 价值在于：换成 MySQL 实现时，漏掉一个方法会立刻被发现。
# ---------------------------------------------------------------------------


def test_in_memory_implementations_satisfy_their_protocols() -> None:
    assert isinstance(InMemorySessionStore(), SessionStore)
    assert isinstance(InMemoryRunStore(), RunStore)
    assert isinstance(InMemoryApprovalStore(), ApprovalStore)
    assert isinstance(InMemoryIdempotencyStore(), IdempotencyStore)
    assert isinstance(InMemoryEventLog(), EventLog)


# ---------------------------------------------------------------------------
# stateVersion CAS
# ---------------------------------------------------------------------------


def test_create_then_get_roundtrip() -> None:
    store = InMemoryRunStore()
    store.create_run(_run())
    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.status == RUNNING


def test_get_unknown_run_returns_none() -> None:
    assert InMemoryRunStore().get_run("nope") is None


def test_duplicate_create_is_refused() -> None:
    """重复创建说明调用方把「恢复」当成了「新建」。"""
    store = InMemoryRunStore()
    store.create_run(_run())
    with pytest.raises(StateVersionConflictError, match="已存在"):
        store.create_run(_run())


def test_save_with_matching_version_succeeds() -> None:
    store = InMemoryRunStore()
    store.create_run(_run())
    moved = _run().transition_to(CHECKING)
    store.save_run(moved, expected_version=1)
    reloaded = store.get_run("run-1")
    assert reloaded is not None
    assert reloaded.status == CHECKING
    assert reloaded.state_version == 2


def test_save_with_stale_version_is_refused() -> None:
    """最核心的一条：并发 resume 里后到的那个必须失败，而不是覆盖。"""
    store = InMemoryRunStore()
    store.create_run(_run())
    store.save_run(_run().transition_to(CHECKING), expected_version=1)

    stale = _run().transition_to(COMPLETED)  # 基于 version 1 做的判断
    with pytest.raises(StateVersionConflictError, match="冲突"):
        store.save_run(stale, expected_version=1)


def test_conflict_message_tells_caller_not_to_retry() -> None:
    """错误信息必须指向「重新读取事实」，而不是「重试写入」。"""
    store = InMemoryRunStore()
    store.create_run(_run())
    store.save_run(_run().transition_to(CHECKING), expected_version=1)
    with pytest.raises(StateVersionConflictError) as error:
        store.save_run(_run().transition_to(COMPLETED), expected_version=1)
    assert "重新读取" in str(error.value)


def test_save_must_increment_version() -> None:
    """版本号不递增，CAS 就形同虚设。"""
    store = InMemoryRunStore()
    store.create_run(_run())
    with pytest.raises(ValueError, match="必须大于"):
        store.save_run(_run(), expected_version=1)


def test_save_unknown_run_is_refused() -> None:
    store = InMemoryRunStore()
    with pytest.raises(StateVersionConflictError, match="不存在"):
        store.save_run(_run(state_version=2), expected_version=1)


def test_concurrent_resume_lets_exactly_one_win() -> None:
    """八个线程同时基于 version 1 推进，只能有一个成功。

    这个测试直接对应验收条件「重复 resume 不改变已完成状态」。
    如果它变红，意味着补丁可能被应用多次。
    """
    store = InMemoryRunStore()
    store.create_run(_run())

    successes: list[int] = []
    conflicts: list[int] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        try:
            store.save_run(_run().transition_to(CHECKING), expected_version=1)
        except StateVersionConflictError:
            with lock:
                conflicts.append(index)
        else:
            with lock:
                successes.append(index)

    threads: list[threading.Thread] = []
    for index in range(8):
        threads.append(threading.Thread(target=attempt, args=(index,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(conflicts) == 7
    final = store.get_run("run-1")
    assert final is not None
    assert final.state_version == 2


def test_list_runs_for_goal_is_deterministic() -> None:
    store = InMemoryRunStore()
    store.create_run(_run(run_id="run-2"))
    store.create_run(_run(run_id="run-1"))
    store.create_run(_run(run_id="run-3", goal_id="goal-2"))
    ids: list[str] = []
    for run in store.list_runs_for_goal("goal-1"):
        ids.append(run.run_id)
    assert ids == ["run-1", "run-2"]


# ---------------------------------------------------------------------------
# Session / Goal
# ---------------------------------------------------------------------------


def test_session_roundtrip() -> None:
    store = InMemorySessionStore()
    session = ConversationSession(session_id="sess-1", scope=SCOPE, repo_id="repo-1")
    store.save_session(session)
    assert store.get_session("sess-1") == session


def test_get_unknown_session_returns_none() -> None:
    assert InMemorySessionStore().get_session("nope") is None


def test_list_active_goals_excludes_finished_ones() -> None:
    store = InMemorySessionStore()
    store.save_goal(_goal("goal-1"))
    store.save_goal(_goal("goal-2", status=GOAL_ABANDONED))
    store.save_goal(_goal("goal-3", session_id="sess-2"))
    active: list[str] = []
    for goal in store.list_active_goals("sess-1"):
        active.append(goal.goal_id)
    assert active == ["goal-1"]


def test_same_session_continues_the_same_goal() -> None:
    """验收条件：同一 session 的第二次提问继续同一个 Goal。"""
    store = InMemorySessionStore()
    store.save_goal(_goal("goal-1"))
    session = ConversationSession(
        session_id="sess-1", scope=SCOPE, repo_id="repo-1"
    ).with_active_goal("goal-1")
    store.save_session(session)

    reloaded = store.get_session("sess-1")
    assert reloaded is not None
    assert reloaded.active_goal_id == "goal-1"
    assert store.get_goal(reloaded.active_goal_id) is not None


# ---------------------------------------------------------------------------
# 审批一次性消费
# ---------------------------------------------------------------------------


def test_approval_consume_marks_it_used() -> None:
    store = InMemoryApprovalStore()
    store.issue(_approval())
    consumed = store.consume("tok-1", now_epoch_ms=1_000)
    assert consumed.is_consumed
    assert consumed.consumed_at_epoch_ms == 1_000


def test_second_consume_is_refused() -> None:
    """「一次性」的直接体现：同一次批准不能应用两次补丁。"""
    store = InMemoryApprovalStore()
    store.issue(_approval())
    store.consume("tok-1", now_epoch_ms=1_000)
    with pytest.raises(ApprovalAlreadyConsumedError, match="一次性"):
        store.consume("tok-1", now_epoch_ms=1_100)


def test_expired_approval_is_refused() -> None:
    store = InMemoryApprovalStore()
    store.issue(_approval(expires_at_epoch_ms=1_000))
    with pytest.raises(ApprovalExpiredError, match="过期"):
        store.consume("tok-1", now_epoch_ms=1_000)


def test_unknown_token_is_refused() -> None:
    with pytest.raises(ApprovalNotFoundError):
        InMemoryApprovalStore().consume("nope", now_epoch_ms=1_000)


def test_duplicate_issue_is_refused() -> None:
    store = InMemoryApprovalStore()
    store.issue(_approval())
    with pytest.raises(ValueError, match="重复签发"):
        store.issue(_approval())


def test_concurrent_consume_lets_exactly_one_win() -> None:
    """十个线程抢同一个令牌，只能有一个拿到。

    这条性质是「approval bypass 为 0」这条硬门槛的基础。
    """
    store = InMemoryApprovalStore()
    store.issue(_approval())

    winners: list[int] = []
    losers: list[int] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        try:
            store.consume("tok-1", now_epoch_ms=1_000)
        except ApprovalAlreadyConsumedError:
            with lock:
                losers.append(index)
        else:
            with lock:
                winners.append(index)

    threads: list[threading.Thread] = []
    for index in range(10):
        threads.append(threading.Thread(target=attempt, args=(index,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1
    assert len(losers) == 9


# ---------------------------------------------------------------------------
# 幂等
# ---------------------------------------------------------------------------


def test_first_register_returns_true() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    assert store.register(key, result_ref="patch-1") is True


def test_second_register_returns_false() -> None:
    """False 表示「已登记，跳过执行并返回既有结果」。"""
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    store.register(key, result_ref="patch-1")
    assert store.register(key, result_ref="patch-1") is False
    assert store.lookup(key) == "patch-1"


def test_different_patches_get_independent_keys() -> None:
    """指纹不同的两个补丁都应被允许执行，不能被误判为重复。"""
    store = InMemoryIdempotencyStore()
    first = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    second = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-b")
    assert store.register(first, result_ref="patch-1") is True
    assert store.register(second, result_ref="patch-2") is True


def test_same_key_in_different_scopes_does_not_collide() -> None:
    """三种粒度的键空间必须隔离，否则会互相误命中。"""
    store = InMemoryIdempotencyStore()
    request_key = IdempotencyKey(scope="request", key="same")
    action_key = IdempotencyKey(scope="action", key="same")
    assert store.register(request_key, result_ref="a") is True
    assert store.register(action_key, result_ref="b") is True
    assert store.lookup(request_key) == "a"
    assert store.lookup(action_key) == "b"


def test_lookup_unknown_key_returns_none() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey(scope="action", key="nope")
    assert store.lookup(key) is None


def test_register_requires_a_result_ref() -> None:
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey(scope="action", key="k")
    with pytest.raises(ValueError, match="result_ref"):
        store.register(key, result_ref="")


def test_concurrent_register_lets_exactly_one_win() -> None:
    """并发登记同一个业务动作，只有一个应当去真正执行它。"""
    store = InMemoryIdempotencyStore()
    key = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")

    results: list[bool] = []
    lock = threading.Lock()

    def attempt() -> None:
        outcome = store.register(key, result_ref="patch-1")
        with lock:
            results.append(outcome)

    threads: list[threading.Thread] = []
    for _ in range(10):
        threads.append(threading.Thread(target=attempt))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results.count(True) == 1
    assert results.count(False) == 9
