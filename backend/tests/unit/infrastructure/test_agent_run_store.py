"""Q-011 Task 2：Agent Run 事实层与事件回放的契约测试。

同一组断言跑两份实现：

    InMemoryAgentRunStore  单进程模式与单元测试用的实现。
    MySqlAgentRunStore     由本机 SQLite 文件库承载。

第二项不是「MySQL 已经验证」的证据。它证明的是那些 SQL 语句本身——
条件 UPDATE 的 CAS、UNIQUE(turn_id)、after_sequence 续传——在真实 SQL 引擎上
成立，而不是只在 Python 字典里成立。MySQL 的 InnoDB 行锁、并发与字符集行为
仍以 tests/integration/test_agent_run_persistence.py 为准（未配测试库时跳过）。
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from codeinsight.domain.agent_run import (
    COMPLETED,
    FAILED,
    MANUAL_REQUIRED,
    QUEUED,
    RUNNING,
    WAITING_APPROVAL,
    WAITING_VALIDATION,
    AgentRunRecord,
)
from codeinsight.domain.ports import AgentRunStore
from codeinsight.domain.trace import TASK_QUEUED, TOOL_CALLED, WORKER_CLAIMED, RunEvent
from codeinsight.infrastructure.db.schema import Base
from codeinsight.infrastructure.db.stores import MySqlAgentRunStore, MySqlEventLog
from codeinsight.infrastructure.event_log import EventSequenceError
from codeinsight.infrastructure.run_store import (
    AgentRunTurnConflictError,
    InMemoryAgentRunStore,
)


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


def _event(sequence: int, event_type: str, *, run_id: str = "run-1") -> RunEvent:
    return RunEvent(
        event_id=f"ev-{run_id}-{sequence}",
        run_id=run_id,
        sequence=sequence,
        event_type=event_type,
        occurred_at_epoch_ms=1_000 + sequence,
    )


def _sqlite_factory(path: Path) -> sessionmaker[Session]:
    """用 SQLite 文件库承载 MySQL 实现。用文件而不是内存库，重启语义才成立。"""

    engine = create_engine(f"sqlite+pysqlite:///{path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


@pytest.fixture(params=["memory", "sql"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> AgentRunStore:
    if request.param == "memory":
        return InMemoryAgentRunStore()
    return MySqlAgentRunStore(_sqlite_factory(tmp_path / "agent-run.db"))


# 契约与基础读写


def test_both_implementations_satisfy_the_protocol(store: AgentRunStore) -> None:
    assert isinstance(store, AgentRunStore)


def test_round_trip_preserves_every_field(store: AgentRunStore) -> None:
    record = _record(
        status=RUNNING,
        attempt=2,
        max_attempts=3,
        worker_id="worker-7",
        lease_until_epoch_ms=9_000,
        event_sequence=4,
        updated_at_epoch_ms=1_234,
    )
    store.save_run(record)
    assert store.get_run("run-1") == record


def test_unknown_run_is_not_found(store: AgentRunStore) -> None:
    assert store.get_run("nope") is None
    assert store.find_run_by_turn("nope") is None


def test_find_run_by_turn(store: AgentRunStore) -> None:
    store.save_run(_record())
    found = store.find_run_by_turn("turn-1")
    assert found is not None
    assert found.run_id == "run-1"


def test_one_turn_cannot_own_two_runs(store: AgentRunStore) -> None:
    """同一条用户消息只对应一个 Run——重发不该跑两遍模型。"""

    store.save_run(_record())
    with pytest.raises(AgentRunTurnConflictError):
        store.save_run(_record(run_id="run-2"))


def test_updating_the_same_run_is_allowed(store: AgentRunStore) -> None:
    store.save_run(_record())
    store.save_run(_record(status=RUNNING, updated_at_epoch_ms=2_000))
    loaded = store.get_run("run-1")
    assert loaded is not None
    assert loaded.status == RUNNING


# 未完成 Run 的巡检清单


def _seed_mixed_states(store: AgentRunStore) -> None:
    store.save_run(
        _record(run_id="r-done", turn_id="t-5", status=COMPLETED, updated_at_epoch_ms=5)
    )
    store.save_run(_record(run_id="r-failed", turn_id="t-6", status=FAILED, updated_at_epoch_ms=6))
    store.save_run(
        _record(run_id="r-manual", turn_id="t-7", status=MANUAL_REQUIRED, updated_at_epoch_ms=7)
    )
    store.save_run(_record(run_id="r-queued", turn_id="t-1", status=QUEUED, updated_at_epoch_ms=30))
    store.save_run(
        _record(
            run_id="r-running",
            turn_id="t-2",
            status=RUNNING,
            updated_at_epoch_ms=10,
            worker_id="worker-1",
            lease_until_epoch_ms=_now_ms() + 60_000,
        )
    )
    store.save_run(
        _record(run_id="r-approval", turn_id="t-3", status=WAITING_APPROVAL, updated_at_epoch_ms=20)
    )
    store.save_run(
        _record(
            run_id="r-validation",
            turn_id="t-4",
            status=WAITING_VALIDATION,
            updated_at_epoch_ms=25,
        )
    )


def test_open_runs_exclude_finished_and_attention(store: AgentRunStore) -> None:
    """终态与「等人处理」都不进巡检清单；等待态必须进——它不是失败。"""

    _seed_mixed_states(store)
    opened = store.list_open_runs()
    assert [item.run_id for item in opened] == [
        "r-running",
        "r-approval",
        "r-validation",
        "r-queued",
    ]


def test_open_runs_respect_the_limit(store: AgentRunStore) -> None:
    _seed_mixed_states(store)
    opened = store.list_open_runs(limit=2)
    assert [item.run_id for item in opened] == ["r-running", "r-approval"]


# 租约 CAS


def test_claim_queued_run_takes_a_lease(store: AgentRunStore) -> None:
    store.save_run(_record())
    claimed = store.claim_run(
        "run-1", worker_id="worker-1", lease_until_epoch_ms=_now_ms() + 60_000
    )
    assert claimed is not None
    assert claimed.status == RUNNING
    assert claimed.worker_id == "worker-1"
    assert claimed.lease_until_epoch_ms is not None


def test_second_claim_is_refused_while_the_lease_holds(store: AgentRunStore) -> None:
    store.save_run(_record())
    lease = _now_ms() + 60_000
    assert store.claim_run("run-1", worker_id="w-1", lease_until_epoch_ms=lease) is not None
    assert store.claim_run("run-1", worker_id="w-2", lease_until_epoch_ms=lease) is None


def test_expired_lease_allows_takeover(store: AgentRunStore) -> None:
    """租约过期是「上一个 Worker 没了」的唯一信号。"""

    store.save_run(
        _record(
            status=RUNNING,
            worker_id="dead-worker",
            lease_until_epoch_ms=1,
            updated_at_epoch_ms=2,
        )
    )
    claimed = store.claim_run("run-1", worker_id="w-2", lease_until_epoch_ms=_now_ms() + 60_000)
    assert claimed is not None
    assert claimed.worker_id == "w-2"


def test_waiting_and_finished_runs_cannot_be_claimed(store: AgentRunStore) -> None:
    """等待审批的 Run 被重新领取，等于用户还没批准就又跑了一遍。"""

    store.save_run(_record(run_id="r-wait", turn_id="t-1", status=WAITING_APPROVAL))
    store.save_run(_record(run_id="r-done", turn_id="t-2", status=COMPLETED))
    lease = _now_ms() + 60_000
    assert store.claim_run("r-wait", worker_id="w", lease_until_epoch_ms=lease) is None
    assert store.claim_run("r-done", worker_id="w", lease_until_epoch_ms=lease) is None


def test_claim_unknown_run_returns_none(store: AgentRunStore) -> None:
    assert store.claim_run("nope", worker_id="w", lease_until_epoch_ms=_now_ms() + 1) is None


# 重启与事件回放：只有真正落盘的实现才有「重启」可言


def test_queued_run_survives_a_restart(tmp_path: Path) -> None:
    """计划场景 1：API 建完 QUEUED Run 就重启，新进程仍读得到它。"""

    path = tmp_path / "restart.db"
    MySqlAgentRunStore(_sqlite_factory(path)).save_run(_record())

    opened_again = MySqlAgentRunStore(_sqlite_factory(path)).get_run("run-1")
    assert opened_again is not None
    assert opened_again.status == QUEUED
    assert opened_again.turn_id == "turn-1"


def test_worker_crash_keeps_the_last_persisted_state(tmp_path: Path) -> None:
    """计划场景 2：Worker 写下 worker_claimed 后崩溃，重建后读到的是最后状态。"""

    path = tmp_path / "crash.db"
    log = MySqlEventLog(_sqlite_factory(path))
    log.append(_event(1, TASK_QUEUED))
    log.append(_event(2, WORKER_CLAIMED))

    reopened = MySqlEventLog(_sqlite_factory(path)).read_events("run-1")
    assert [event.event_type for event in reopened] == [TASK_QUEUED, WORKER_CLAIMED]


def test_duplicate_sequence_is_refused(tmp_path: Path) -> None:
    """计划场景 3：同一个序号写两次只能留下一条。"""

    path = tmp_path / "duplicate.db"
    log = MySqlEventLog(_sqlite_factory(path))
    log.append(_event(1, TASK_QUEUED))
    with pytest.raises(EventSequenceError):
        log.append(_event(1, TASK_QUEUED))


def test_events_replay_after_a_sequence(tmp_path: Path) -> None:
    """计划场景 4：断线重连按 after_sequence 续传，不重不漏。"""

    path = tmp_path / "replay.db"
    log = MySqlEventLog(_sqlite_factory(path))
    for sequence, event_type in ((1, TASK_QUEUED), (2, WORKER_CLAIMED), (3, TOOL_CALLED)):
        log.append(_event(sequence, event_type))

    tail = log.read_events("run-1", after_sequence=1)
    assert [event.sequence for event in tail] == [2, 3]

