"""Q-011 Task 5：AgentRunWorker 生命周期的单元测试。

用假的加载器与执行体，锁住 Worker 自己的判断：租约、截止时间、失败分类，
以及「说不清的终止必须交人工」这条边界。
"""

from __future__ import annotations

import time
from dataclasses import replace

from codeinsight.agent.agent_run_worker import (
    RESUME_TASK_NOT_WIRED,
    RUN_RECORD_MISSING,
    AgentRunWorker,
)
from codeinsight.application.agent_run_executor import (
    CONTEXT_SESSION_NOT_FOUND,
    DEADLINE_EXCEEDED,
    EXECUTION_FAILED,
    AgentRunContext,
    AgentRunContextError,
    AgentRunExecutor,
)
from codeinsight.domain.agent_run import (
    COMPLETED,
    FAILED,
    MANUAL_REQUIRED,
    QUEUED,
    RUNNING,
    TASK_AGENT_RUN,
    TASK_RESUME_AFTER_APPROVAL,
    WAITING_APPROVAL,
    AgentRunRecord,
    AgentRunTask,
    RunRequestOptions,
)
from codeinsight.infrastructure.chat_runtime import ChatExecution
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore

# Store 内部用自己的真实时钟判断租约，因此这里的时间基准也必须贴近真实时间，
# 否则「还没过期」的租约在 Store 眼里已经是 1970 年的过期租约。
NOW_MS = int(time.time() * 1000)


def _task(**overrides: object) -> AgentRunTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "session_id": "s-1",
        "turn_id": "turn-1",
        "run_id": "run-1",
        "task_kind": TASK_AGENT_RUN,
        "idempotency_key": "turn-1",
        "policy_version": "chat-agent-run-v1",
        "deadline_epoch_ms": NOW_MS + 60_000,
    }
    values.update(overrides)
    return AgentRunTask(**values)  # type: ignore[arg-type]


def _record(**overrides: object) -> AgentRunRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "turn_id": "turn-1",
        "session_id": "s-1",
        "task_id": "task-1",
        "task_kind": TASK_AGENT_RUN,
        "status": QUEUED,
        "policy_version": "chat-agent-run-v1",
        "idempotency_key": "turn-1",
        "deadline_epoch_ms": NOW_MS + 60_000,
        "updated_at_epoch_ms": NOW_MS,
    }
    values.update(overrides)
    return AgentRunRecord(**values)  # type: ignore[arg-type]


def _context(run_id: str = "run-1") -> AgentRunContext:
    return AgentRunContext(
        run_id=run_id,
        turn_id="turn-1",
        session_id="s-1",
        task_id="task-1",
        repo_id="repo-1",
        repo_root="C:/repo",
        repo_fingerprint="fp:repo",
        user_message="checkout 如何校验输入？",
        task_type="explain",
        confidence=0.9,
        rule="test",
        options=RunRequestOptions(),
    )


class _StubLoader:
    def __init__(self, *, context: AgentRunContext | None = None, error: str | None = None) -> None:
        self._context = context or _context()
        self._error = error

    def load(self, task: AgentRunTask, record: AgentRunRecord) -> AgentRunContext:
        if self._error is not None:
            raise AgentRunContextError(self._error)
        return self._context


def _build(
    *,
    loader: object | None = None,
    record: AgentRunRecord | None = None,
    now_ms: int = NOW_MS,
) -> tuple[AgentRunWorker, InMemoryAgentRunStore, list[tuple[str, str, dict[str, str]]]]:
    store = InMemoryAgentRunStore()
    stored = record or _record()
    store.save_run(stored)
    events: list[tuple[str, str, dict[str, str]]] = []
    emit = lambda run_id, event_type, payload: events.append((run_id, event_type, payload))  # noqa: E731
    worker = AgentRunWorker(
        store=store,
        loader=loader or _StubLoader(),  # type: ignore[arg-type]
        executor=AgentRunExecutor(emit=emit),
        emit=emit,
        worker_id="worker-test",
        clock_ms=lambda: now_ms,
    )
    return worker, store, events


def _ok_runner(context: AgentRunContext) -> ChatExecution:
    return ChatExecution(status="COMPLETED", assistant_message="答")


# ---------------------------------------------------------------------------
# 领取
# ---------------------------------------------------------------------------


def test_worker_claims_the_run_and_records_it() -> None:
    worker, store, events = _build()

    outcome = worker.handle(_task(), runner=_ok_runner)

    assert outcome.claimed is True
    assert outcome.execution is not None
    assert [event_type for _, event_type, _ in events] == ["worker_claimed", "context_loaded"]
    assert events[0][2]["worker_id"] == "worker-test"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.status == COMPLETED


def test_second_worker_cannot_claim_a_leased_run() -> None:
    """重复投递的第二条消息不该跑第二遍模型。"""

    held = _record(status=RUNNING, worker_id="w-1", lease_until_epoch_ms=NOW_MS + 600_000)
    worker, _, events = _build(record=held)

    outcome = worker.handle(_task(), runner=_ok_runner)

    assert outcome.claimed is False
    assert outcome.execution is None
    assert events == []


def test_task_without_a_run_record_is_stopped() -> None:
    worker, _, _ = _build()
    worker._store._runs.clear()  # type: ignore[attr-defined]

    outcome = worker.handle(_task(), runner=_ok_runner)

    assert outcome.claimed is False
    assert outcome.status == MANUAL_REQUIRED
    assert outcome.error_class == RUN_RECORD_MISSING


def test_resume_task_is_refused_instead_of_guessed() -> None:
    """续跑任务会写隔离工作区，本轮实现明确停住而不是猜着跑。"""

    worker, store, events = _build()

    outcome = worker.handle(
        _task(task_kind=TASK_RESUME_AFTER_APPROVAL, run_id="run-1"), runner=_ok_runner
    )

    assert outcome.claimed is False
    assert outcome.status == MANUAL_REQUIRED
    assert outcome.error_class == RESUME_TASK_NOT_WIRED
    assert outcome.claimed is False
    assert events[0][1] == "run_unknown"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.status == MANUAL_REQUIRED


def test_expired_deadline_stops_before_the_model_runs() -> None:
    called: list[str] = []

    def runner(context: AgentRunContext) -> ChatExecution:
        called.append(context.run_id)
        return ChatExecution(status="COMPLETED", assistant_message="答")

    worker, store, events = _build(
        record=_record(deadline_epoch_ms=NOW_MS - 1), now_ms=NOW_MS
    )

    outcome = worker.handle(_task(), runner=runner)

    assert called == []
    assert outcome.status == FAILED
    assert outcome.error_class == DEADLINE_EXCEEDED
    assert ("run-1", "run_failed", {"status": FAILED, "error_class": DEADLINE_EXCEEDED}) in events


# ---------------------------------------------------------------------------
# 失败分类
# ---------------------------------------------------------------------------


def test_context_error_fails_the_run_with_its_class() -> None:
    worker, store, events = _build(loader=_StubLoader(error=CONTEXT_SESSION_NOT_FOUND))

    outcome = worker.handle(_task(), runner=_ok_runner)

    assert outcome.status == FAILED
    assert outcome.error_class == CONTEXT_SESSION_NOT_FOUND
    assert events[-1][1] == "run_failed"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.error_class == CONTEXT_SESSION_NOT_FOUND


def test_failed_execution_marks_the_run_failed() -> None:
    def runner(context: AgentRunContext) -> ChatExecution:
        return ChatExecution(status="FAILED", assistant_message=None, error="执行失败")

    worker, store, events = _build()

    outcome = worker.handle(_task(), runner=runner)

    assert outcome.status == FAILED
    assert outcome.error_class == EXECUTION_FAILED
    assert events[-1][1] == "run_failed"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.status == FAILED


def test_waiting_approval_is_not_a_failure() -> None:
    """等待审批不是失败：它要能被恢复，而不是被当成没跑成。"""

    def runner(context: AgentRunContext) -> ChatExecution:
        return ChatExecution(status="WAITING_APPROVAL", assistant_message="请确认")

    worker, store, events = _build()

    outcome = worker.handle(_task(), runner=runner)

    assert outcome.status == WAITING_APPROVAL
    assert outcome.error_class is None
    assert outcome.ok is True
    assert [event_type for _, event_type, _ in events] == ["worker_claimed", "context_loaded"]
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.waiting is True


def test_unexpected_failure_needs_a_human() -> None:
    """说不清的终止必须交人工，而不是伪装成一个普通的失败。"""

    def runner(context: AgentRunContext) -> ChatExecution:
        raise KeyError("意外")

    worker, store, events = _build()

    outcome = worker.handle(_task(), runner=runner)

    assert outcome.status == MANUAL_REQUIRED
    assert outcome.error_class == "UNEXPECTED_ERROR:KeyError"
    assert events[-1][1] == "run_unknown"
    assert events[-1][2]["error_class"] == "UNEXPECTED_ERROR:KeyError"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.needs_attention is True


def test_run_can_be_reclaimed_after_the_lease_expires() -> None:
    """租约过期是「上一个 Worker 没了」的唯一信号。"""

    store = InMemoryAgentRunStore()
    store.save_run(
        _record(
            status=RUNNING,
            worker_id="dead-worker",
            lease_until_epoch_ms=NOW_MS - 1,
        )
    )
    events: list[tuple[str, str, dict[str, str]]] = []
    emit = lambda run_id, event_type, payload: events.append((run_id, event_type, payload))  # noqa: E731
    worker = AgentRunWorker(
        store=store,
        loader=_StubLoader(),
        executor=AgentRunExecutor(emit=emit),
        emit=emit,
        worker_id="worker-new",
        clock_ms=lambda: NOW_MS,
    )

    outcome = worker.handle(_task(), runner=_ok_runner)

    assert outcome.claimed is True
    assert events[0][2]["worker_id"] == "worker-new"
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.status == COMPLETED


def test_same_run_id_can_be_advanced_by_a_second_attempt() -> None:
    """恢复不是「换一个 Run」，而是同一个 run_id 的下一次尝试。"""

    worker, store, _ = _build()
    worker.handle(_task(), runner=_ok_runner)

    resumed = replace(_task(), attempt=2)
    outcome = worker.handle(resumed, runner=_ok_runner)

    # 终态不允许再被推进：第二次尝试拿不到租约，也不该改写终态。
    assert outcome.claimed is False
    stored = store.get_run("run-1")
    assert stored is not None
    assert stored.status == COMPLETED
    assert stored.attempt == 1

