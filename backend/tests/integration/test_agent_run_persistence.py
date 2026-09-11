"""Q-011 Task 2：Agent Run 持久化的 MySQL 集成测试。

未配置测试库时整体跳过（见 conftest.py）。单元测试已经用 SQLite 证明了那些
SQL 语句本身能执行；这里补的是只有在真实 MySQL 上才成立的部分——InnoDB 行锁、
UNIQUE 约束下两个 Worker 抢同一个 Run、以及重启后的读取。

本轮没有可用的 MySQL 实例（3307 上没有服务），因此这些用例处于
「已写好、等环境」的状态，不能算作已通过的证据。
"""

from __future__ import annotations

import threading
import time

from sqlalchemy.orm import Session, sessionmaker

from codeinsight.domain.agent_run import QUEUED, RUNNING, AgentRunRecord
from codeinsight.domain.ports import AgentRunStore
from codeinsight.domain.trace import TASK_QUEUED, TOOL_CALLED, WORKER_CLAIMED, RunEvent
from codeinsight.infrastructure.db.stores import MySqlAgentRunStore, MySqlEventLog
from codeinsight.infrastructure.event_log import EventSequenceError
from codeinsight.infrastructure.run_store import AgentRunTurnConflictError


def _now_ms() -> int:
    return int(time.time() * 1000)


def _record(**overrides: object) -> AgentRunRecord:
    defaults: dict[str, object] = {
        "run_id": "run-1",
        "turn_id": "turn-1",
        "session_id": "sess-1",
        "task_id": "task-1",
        "task_kind": "agent_run",
        "status": QUEUED,
        "policy_version": "agent-run-v1",
        "idempotency_key": "turn-1",
        "deadline_epoch_ms": 10_000,
        "updated_at_epoch_ms": 1_000,
    }
    defaults.update(overrides)
    return AgentRunRecord(**defaults)  # type: ignore[arg-type]


def _event(sequence: int, event_type: str) -> RunEvent:
    return RunEvent(
        event_id=f"ev-{sequence}",
        run_id="run-1",
        sequence=sequence,
        event_type=event_type,
        occurred_at_epoch_ms=1_000 + sequence,
    )


def test_agent_run_store_satisfies_its_protocol(
    session_factory: sessionmaker[Session],
) -> None:
    assert isinstance(MySqlAgentRunStore(session_factory), AgentRunStore)


def test_queued_run_survives_a_restart(session_factory: sessionmaker[Session]) -> None:
    """场景 1：进程重启后仍读得到那条还没有跑完的 Run。"""

    MySqlAgentRunStore(session_factory).save_run(_record())

    opened_again = MySqlAgentRunStore(session_factory).get_run("run-1")
    assert opened_again is not None
    assert opened_again.status == QUEUED
    assert opened_again.turn_id == "turn-1"


def test_turn_id_is_unique(session_factory: sessionmaker[Session]) -> None:
    """唯一约束由数据库兜底，而不是靠调用方先查一遍。"""

    store = MySqlAgentRunStore(session_factory)
    store.save_run(_record())
    raised = False
    try:
        store.save_run(_record(run_id="run-2"))
    except AgentRunTurnConflictError:
        raised = True
    assert raised is True


def test_worker_event_then_crash_keeps_last_state(
    session_factory: sessionmaker[Session],
) -> None:
    """场景 2：Worker 写下事件后崩溃，重建后可读最后状态。"""

    MySqlAgentRunStore(session_factory).save_run(
        _record(status=RUNNING, worker_id="worker-1", lease_until_epoch_ms=1)
    )
    log = MySqlEventLog(session_factory)
    log.append(_event(1, TASK_QUEUED))
    log.append(_event(2, WORKER_CLAIMED))

    last = MySqlAgentRunStore(session_factory).get_run("run-1")
    assert last is not None
    assert last.status == RUNNING
    assert [event.event_type for event in MySqlEventLog(session_factory).read_events("run-1")] == [
        TASK_QUEUED,
        WORKER_CLAIMED,
    ]


def test_duplicate_event_sequence_is_refused(session_factory: sessionmaker[Session]) -> None:
    """场景 3：UNIQUE(run_id, sequence) 是序号唯一性的最后防线。"""

    log = MySqlEventLog(session_factory)
    log.append(_event(1, TASK_QUEUED))
    refused = False
    try:
        log.append(_event(1, TASK_QUEUED))
    except EventSequenceError:
        refused = True
    assert refused is True


def test_events_replay_after_a_sequence(session_factory: sessionmaker[Session]) -> None:
    """场景 4：SSE 断线重连按 after_sequence 续传。"""

    log = MySqlEventLog(session_factory)
    for sequence, event_type in ((1, TASK_QUEUED), (2, WORKER_CLAIMED), (3, TOOL_CALLED)):
        log.append(_event(sequence, event_type))

    tail = log.read_events("run-1", after_sequence=1)
    assert [event.sequence for event in tail] == [2, 3]


def test_claim_lets_exactly_one_worker_in(session_factory: sessionmaker[Session]) -> None:
    """两个 Worker 同时抢同一个 Run，只有一个能拿到租约。"""

    store = MySqlAgentRunStore(session_factory)
    store.save_run(_record())

    winners: list[str] = []
    lock = threading.Lock()
    barrier = threading.Barrier(2)

    def attempt(worker_id: str) -> None:
        barrier.wait(timeout=10)
        claimed = store.claim_run(
            "run-1", worker_id=worker_id, lease_until_epoch_ms=_now_ms() + 60_000
        )
        if claimed is not None:
            with lock:
                winners.append(claimed.worker_id or "")

    threads = [threading.Thread(target=attempt, args=(name,)) for name in ("w-1", "w-2")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(winners) == 1

