"""Q-011 Task 3：投递契约的单元测试。

锁住的三条：事实先落地再投递、同一条消息不会产生第二个 Run、
投递失败必须留下一个可查询的失败状态。
"""

from __future__ import annotations

import time

from codeinsight.application.agent_run_dispatcher import (
    DISPATCH_FAILED,
    DISPATCH_UNKNOWN,
    AgentRunDispatcher,
    AgentRunTransport,
    InMemoryAgentRunTransport,
)
from codeinsight.domain.agent_run import (
    FAILED,
    MANUAL_REQUIRED,
    QUEUED,
    RECOVERY_MANUAL_REQUIRED,
    RECOVERY_RETRY,
    RECOVERY_UNKNOWN,
    TASK_AGENT_RUN,
    TASK_RESUME_AFTER_APPROVAL,
    AgentRunTask,
    recovery_decision,
)
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _task(**overrides: object) -> AgentRunTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "session_id": "session-1",
        "turn_id": "turn-1",
        "run_id": "run-1",
        "task_kind": TASK_AGENT_RUN,
        "idempotency_key": "turn-1",
        "policy_version": "agent-run-v1",
        "deadline_epoch_ms": 10_000,
    }
    values.update(overrides)
    return AgentRunTask(**values)  # type: ignore[arg-type]


class _StorePeekingTransport(InMemoryAgentRunTransport):
    """在投递的那一刻回看 Store：Run 必须先落地，再谈投递。"""

    def __init__(self, store: InMemoryAgentRunStore) -> None:
        super().__init__()
        self._store = store
        self.seen_status: list[str | None] = []

    def publish(self, task: AgentRunTask) -> str:
        record = self._store.get_run(task.run_id)
        self.seen_status.append(None if record is None else record.status)
        return super().publish(task)


def test_transport_protocol_is_satisfied() -> None:
    assert isinstance(InMemoryAgentRunTransport(), AgentRunTransport)


def test_run_is_written_before_the_message_is_published() -> None:
    store = InMemoryAgentRunStore()
    transport = _StorePeekingTransport(store)
    dispatcher = AgentRunDispatcher(store=store, transport=transport, clock_ms=lambda: 1_000)

    record = dispatcher.dispatch(_task())

    assert transport.seen_status == [QUEUED]
    assert record.status == QUEUED
    assert store.get_run("run-1") == record


def test_resending_the_same_turn_does_not_run_a_second_task() -> None:
    store = InMemoryAgentRunStore()
    transport = InMemoryAgentRunTransport()
    dispatcher = AgentRunDispatcher(store=store, transport=transport, clock_ms=lambda: 1_000)

    first = dispatcher.dispatch(_task())
    second = dispatcher.dispatch(_task(task_id="task-2"))

    assert first == second
    assert len(transport.published) == 1
    assert len(store.list_open_runs()) == 1


def test_failed_dispatch_leaves_a_visible_failure() -> None:
    """「已排队但没人执行」是最难发现的状态，必须变成可查询的失败。"""

    store = InMemoryAgentRunStore()
    transport = InMemoryAgentRunTransport(failure=RuntimeError("broker down"))
    dispatcher = AgentRunDispatcher(store=store, transport=transport, clock_ms=lambda: 1_000)

    record = dispatcher.dispatch(_task())

    assert record.status == FAILED
    assert record.error_class == DISPATCH_FAILED
    persisted = store.get_run("run-1")
    assert persisted is not None
    assert persisted.status == FAILED


def test_ambiguous_dispatch_of_a_side_effecting_task_needs_manual_review() -> None:
    """续跑任务投递超时，无法区分「没送到」和「送到了但没回执」。"""

    store = InMemoryAgentRunStore()
    transport = InMemoryAgentRunTransport(failure=TimeoutError("no ack"))
    dispatcher = AgentRunDispatcher(store=store, transport=transport, clock_ms=lambda: 1_000)

    record = dispatcher.dispatch(_task(task_kind=TASK_RESUME_AFTER_APPROVAL))

    assert record.status == MANUAL_REQUIRED
    assert record.error_class == DISPATCH_UNKNOWN


def test_task_payload_carries_ids_and_version_only() -> None:
    payload = _task().as_payload()

    assert set(payload) == {
        "task_id",
        "session_id",
        "turn_id",
        "run_id",
        "task_kind",
        "idempotency_key",
        "policy_version",
        "deadline_epoch_ms",
        "attempt",
        "max_attempts",
    }
    for forbidden in (
        "user_message",
        "question",
        "messages",
        "source",
        "source_code",
        "prompt",
        "api_key",
        "diff",
    ):
        assert forbidden not in payload


def test_only_one_worker_can_hold_the_lease() -> None:
    store = InMemoryAgentRunStore()
    dispatcher = AgentRunDispatcher(
        store=store, transport=InMemoryAgentRunTransport(), clock_ms=lambda: 1_000
    )
    dispatcher.dispatch(_task())

    lease = _now_ms() + 60_000
    assert store.claim_run("run-1", worker_id="w-1", lease_until_epoch_ms=lease) is not None
    assert store.claim_run("run-1", worker_id="w-2", lease_until_epoch_ms=lease) is None


def test_expired_lease_decides_between_retry_and_manual() -> None:
    """只读 Run 过期可重试；续跑 Run 状态不明时交人工，不自动重放。"""

    read_only = _task()
    assert recovery_decision(read_only, lease_expired=True, now_epoch_ms=1_000) == RECOVERY_RETRY
    assert (
        recovery_decision(read_only, lease_expired=True, now_epoch_ms=10_000)
        == RECOVERY_MANUAL_REQUIRED
    )
    assert recovery_decision(read_only, lease_expired=False, now_epoch_ms=1) == RECOVERY_UNKNOWN

    side_effecting = _task(task_kind=TASK_RESUME_AFTER_APPROVAL)
    assert (
        recovery_decision(side_effecting, lease_expired=True, now_epoch_ms=1_000)
        == RECOVERY_UNKNOWN
    )

