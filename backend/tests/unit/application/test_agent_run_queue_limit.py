"""Q-011 Task 11：队列容量与背压的单元测试。

锁住四条：不配上限就是原来的行为；满了要明确拒绝并留下可查的事实；
上限只数「还在等领取」的 Run；以及为什么续跑不受它限制。
"""

from __future__ import annotations

import time

import pytest

from codeinsight.application.agent_run_dispatcher import (
    QUEUE_FULL,
    AgentRunDispatcher,
    InMemoryAgentRunTransport,
    QueueFullRejected,
)
from codeinsight.domain.agent_run import (
    FAILED,
    QUEUED,
    TASK_AGENT_RUN,
    TASK_RESUME_AFTER_APPROVAL,
    WAITING_APPROVAL,
    AgentRunRecord,
    AgentRunTask,
)
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore


def _now_ms() -> int:
    return int(time.time() * 1000)


def _task(index: int, *, kind: str = TASK_AGENT_RUN) -> AgentRunTask:
    return AgentRunTask(
        task_id=f"task-{index}",
        session_id=f"session-{index}",
        turn_id=f"turn-{index}",
        run_id=f"run-{index}",
        task_kind=kind,
        idempotency_key=f"turn-{index}",
        policy_version="agent-run-v1",
        deadline_epoch_ms=10_000,
    )


def _dispatcher(*, queue_limit: int | None) -> tuple[AgentRunDispatcher, InMemoryAgentRunStore]:
    store = InMemoryAgentRunStore()
    transport = InMemoryAgentRunTransport()
    return (
        AgentRunDispatcher(store=store, transport=transport, queue_limit=queue_limit),
        store,
    )


def test_without_a_limit_every_run_is_taken() -> None:
    dispatcher, store = _dispatcher(queue_limit=None)

    for index in (1, 2, 3):
        assert dispatcher.dispatch(_task(index)).status == QUEUED

    assert store.count_queued_runs() == 3


def test_a_full_queue_is_rejected_and_leaves_a_queryable_fact() -> None:
    dispatcher, store = _dispatcher(queue_limit=1)
    dispatcher.dispatch(_task(1))

    with pytest.raises(QueueFullRejected):
        dispatcher.dispatch(_task(2))

    rejected = store.get_run("run-2")
    assert rejected is not None
    # 「被拒了多少」必须是一笔可以查询的账，而不是只在日志里出现过的字符串。
    assert rejected.status == FAILED
    assert rejected.error_class == QUEUE_FULL
    # 被拒的那一轮没有进入等待队列，所以上限不会被自己的拒绝撑大。
    assert store.count_queued_runs() == 1


def test_the_limit_counts_only_runs_still_waiting_to_be_claimed() -> None:
    dispatcher, store = _dispatcher(queue_limit=1)
    dispatcher.dispatch(_task(1))

    # 被 Worker 领走之后，等待队列就空了：占着上限的是「在等」，不是「在跑」。
    claimed = store.claim_run(
        "run-1", worker_id="worker-a", lease_until_epoch_ms=_now_ms() + 60_000
    )
    assert claimed is not None

    assert dispatcher.dispatch(_task(2)).status == QUEUED


def test_a_continuation_is_not_blocked_by_a_full_queue() -> None:
    dispatcher, store = _dispatcher(queue_limit=1)
    dispatcher.dispatch(_task(1))

    # 另一条 Run 早就停在等审批：它已经被受理过，卡住它只会把 Run 永久
    # 挂在等待态，因此上限只挡新受理，不挡续跑。
    store.save_run(
        AgentRunRecord(
            run_id="run-9",
            turn_id="turn-9",
            session_id="session-9",
            task_id="task-9",
            task_kind=TASK_AGENT_RUN,
            status=WAITING_APPROVAL,
            policy_version="agent-run-v1",
            idempotency_key="turn-9",
            deadline_epoch_ms=10_000,
            updated_at_epoch_ms=_now_ms(),
        )
    )

    resumed = dispatcher.dispatch_continuation(
        _task(9, kind=TASK_RESUME_AFTER_APPROVAL),
        expected_status=WAITING_APPROVAL,
        patch_id="patch-1",
    )

    assert resumed.status == QUEUED


def test_an_invalid_limit_is_refused_at_construction() -> None:
    with pytest.raises(ValueError):
        AgentRunDispatcher(
            store=InMemoryAgentRunStore(),
            transport=InMemoryAgentRunTransport(),
            queue_limit=0,
        )
