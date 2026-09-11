"""Q-011 Task 1：Agent Run 状态机与后台任务契约。

这些用例锁住三条会被写错一次就再也回不来的边界：
等待态不是失败、只有只读 Run 能重试、任务载荷不含正文。
"""

from __future__ import annotations

import pytest

from codeinsight.domain import agent_run


def _task(**overrides: object) -> agent_run.AgentRunTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "run_id": "run-1",
        "task_kind": agent_run.TASK_AGENT_RUN,
        "idempotency_key": "turn-1",
        "policy_version": "agent-run-v1",
        "deadline_epoch_ms": 10_000,
        "attempt": 1,
        "max_attempts": 2,
    }
    values.update(overrides)
    return agent_run.AgentRunTask(**values)  # type: ignore[arg-type]


def _record(**overrides: object) -> agent_run.AgentRunRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "turn_id": "turn-1",
        "session_id": "session-1",
        "task_id": "task-1",
        "task_kind": agent_run.TASK_AGENT_RUN,
        "status": agent_run.QUEUED,
        "policy_version": "agent-run-v1",
        "idempotency_key": "turn-1",
        "deadline_epoch_ms": 10_000,
        "updated_at_epoch_ms": 1,
    }
    values.update(overrides)
    return agent_run.AgentRunRecord(**values)  # type: ignore[arg-type]


def test_waiting_statuses_are_not_failures() -> None:
    """等待审批和等待验证是「停下来等外部动作」，不是失败也不是终态。"""

    for status in (agent_run.WAITING_APPROVAL, agent_run.WAITING_VALIDATION):
        assert agent_run.is_waiting(status)
        assert status not in agent_run.TERMINAL_AGENT_RUN_STATUSES
        assert not agent_run.needs_attention(status)

    assert agent_run.is_waiting(agent_run.RUNNING) is False
    assert agent_run.is_waiting(agent_run.COMPLETED) is False


def test_unknown_and_manual_required_are_attention_not_terminal() -> None:
    for status in (agent_run.UNKNOWN, agent_run.MANUAL_REQUIRED):
        assert agent_run.needs_attention(status)
        assert status not in agent_run.TERMINAL_AGENT_RUN_STATUSES


def test_running_to_waiting_approval_is_allowed_but_terminal_cannot_move() -> None:
    assert agent_run.can_transition(agent_run.RUNNING, agent_run.WAITING_APPROVAL)
    assert agent_run.can_transition(agent_run.WAITING_APPROVAL, agent_run.QUEUED)
    assert agent_run.can_transition(agent_run.WAITING_VALIDATION, agent_run.RUNNING)

    for status in agent_run.TERMINAL_AGENT_RUN_STATUSES:
        assert agent_run.ALLOWED_TRANSITIONS[status] == frozenset()
        with pytest.raises(ValueError):
            agent_run.transition(status, agent_run.RUNNING)


def test_unknown_rejects_a_worker_status_it_never_produced() -> None:
    with pytest.raises(ValueError):
        agent_run.transition(agent_run.QUEUED, agent_run.WAITING_VALIDATION)
    with pytest.raises(ValueError, match="不支持的 Agent Run 状态"):
        agent_run.transition("PENDING", agent_run.RUNNING)


def test_task_rejects_unknown_kind_and_bad_attempts() -> None:
    with pytest.raises(ValueError, match="不支持的任务类型"):
        _task(task_kind="reindex_everything")
    with pytest.raises(ValueError, match="attempt 从 1 开始"):
        _task(attempt=0)
    with pytest.raises(ValueError, match="不能超过 max_attempts"):
        _task(attempt=3, max_attempts=2)
    with pytest.raises(ValueError, match="deadline_epoch_ms"):
        _task(deadline_epoch_ms=0)


def test_only_the_read_only_kind_may_be_retried() -> None:
    """续跑任务可能已经产生副作用，不能靠重试掩盖。"""

    read_only = _task()
    assert read_only.retryable
    assert read_only.next_attempt().attempt == 2
    assert read_only.may_have_side_effects is False

    exhausted = _task(attempt=2)
    assert exhausted.retryable is False
    with pytest.raises(ValueError, match="不允许继续重试"):
        exhausted.next_attempt()

    for kind in (
        agent_run.TASK_RESUME_AFTER_APPROVAL,
        agent_run.TASK_RESUME_AFTER_VALIDATION,
    ):
        resume = _task(task_kind=kind)
        assert resume.may_have_side_effects
        assert resume.retryable is False


def test_task_payload_carries_ids_and_versions_only() -> None:
    payload = _task().as_payload()

    assert payload["run_id"] == "run-1"
    assert agent_run.AgentRunTask.from_payload(dict(payload)) == _task()
    for forbidden in ("messages", "prompt", "content", "api_key", "answer"):
        assert forbidden not in payload


def test_task_payload_round_trip_rejects_a_trimmed_message() -> None:
    payload = _task().as_payload()
    payload.pop("run_id")

    with pytest.raises(ValueError, match="任务载荷缺少字段"):
        agent_run.AgentRunTask.from_payload(dict(payload))


def test_recovery_retries_read_only_but_never_replays_a_resume_task() -> None:
    read_only = _task()
    assert (
        agent_run.recovery_decision(
            read_only, lease_expired=True, now_epoch_ms=100
        )
        == agent_run.RECOVERY_RETRY
    )

    resume = _task(task_kind=agent_run.TASK_RESUME_AFTER_APPROVAL)
    assert (
        agent_run.recovery_decision(
            resume, lease_expired=True, now_epoch_ms=100
        )
        == agent_run.RECOVERY_UNKNOWN
    )

    assert (
        agent_run.recovery_decision(
            read_only, lease_expired=False, now_epoch_ms=100
        )
        == agent_run.RECOVERY_UNKNOWN
    )
    assert (
        agent_run.recovery_decision(
            _task(attempt=2), lease_expired=True, now_epoch_ms=100
        )
        == agent_run.RECOVERY_MANUAL_REQUIRED
    )
    assert (
        agent_run.recovery_decision(
            read_only, lease_expired=True, now_epoch_ms=20_000
        )
        == agent_run.RECOVERY_MANUAL_REQUIRED
    )


def test_record_claim_requires_a_free_or_expired_lease() -> None:
    queued = _record()
    assert queued.can_be_claimed(now_epoch_ms=100)

    leased = _record(
        status=agent_run.RUNNING,
        worker_id="worker-a",
        lease_until_epoch_ms=500,
    )
    assert leased.can_be_claimed(now_epoch_ms=100) is False
    assert leased.can_be_claimed(now_epoch_ms=600) is True
    assert leased.lease_expired_at(now_epoch_ms=500) is True

    done = _record(status=agent_run.COMPLETED)
    assert done.can_be_claimed(now_epoch_ms=10**9) is False


def test_record_transitions_are_checked_and_sequence_is_carried() -> None:
    queued = _record(event_sequence=4)
    running = queued.advanced(
        status=agent_run.RUNNING,
        updated_at_epoch_ms=2,
        worker_id="worker-a",
        lease_until_epoch_ms=900,
        event_sequence=5,
    )

    assert running.status == agent_run.RUNNING
    assert running.worker_id == "worker-a"
    assert running.event_sequence == 5
    assert running.waiting is False

    with pytest.raises(ValueError, match="不允许的 Agent Run 状态迁移"):
        queued.advanced(status=agent_run.WAITING_VALIDATION, updated_at_epoch_ms=3)

    assert running.as_public_dict()["status"] == agent_run.RUNNING
    assert "lease_until_epoch_ms" not in running.as_public_dict()


def test_record_converts_back_into_the_task_it_was_scheduled_from() -> None:
    record = _record(attempt=2, max_attempts=3, status=agent_run.QUEUED)
    task = record.to_task()

    assert task.task_id == record.task_id
    assert task.run_id == record.run_id
    assert task.attempt == 2
    assert task.max_attempts == 3
