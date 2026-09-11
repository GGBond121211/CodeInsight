"""Q-011 Task 3：Celery 投递与 Worker 注册的单元测试。

不需要 broker：这些用例验证的是「消息里装了什么」「注册了哪些任务名」，
以及 Worker 配置仍是 acks_late 与 prefetch=1。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from codeinsight.agent.agent_run_worker import AgentRunOutcome
from codeinsight.agent.worker_tasks import build_agent_run_task, build_validation_task
from codeinsight.application.agent_run_dispatcher import AgentRunTransport
from codeinsight.application.validation_coordinator import VALIDATION_PASSED, ValidationOutcome
from codeinsight.domain.agent_run import TASK_AGENT_RUN, AgentRunTask
from codeinsight.infrastructure.celery_agent_dispatcher import (
    AGENT_RUN_TASK_NAME,
    VALIDATION_TASK_NAME,
    CeleryAgentRunTransport,
)


def _task() -> AgentRunTask:
    return AgentRunTask(
        task_id="task-1",
        session_id="session-1",
        turn_id="turn-1",
        run_id="run-1",
        task_kind=TASK_AGENT_RUN,
        idempotency_key="turn-1",
        policy_version="agent-run-v1",
        deadline_epoch_ms=10_000,
    )


class _FakeCelery:
    """记录 send_task 的调用，不连 broker。"""

    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []

    def send_task(self, name: str, **options: object) -> SimpleNamespace:
        self.sent.append({"name": name, **options})
        return SimpleNamespace(id=f"msg-{len(self.sent)}")


class _FakeConf:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def update(self, **values: object) -> None:
        self.values.update(values)


class _RecordingCelery:
    """假的 Celery app：记下 @app.task(name=...) 注册了什么。"""

    def __init__(self) -> None:
        self.conf = _FakeConf()
        self.registered: dict[str, object] = {}

    def task(self, *, name: str, **options: object):
        def decorator(function):
            self.registered[name] = function
            return function

        return decorator


def test_transport_satisfies_the_protocol_and_sends_ids_only() -> None:
    app = _FakeCelery()
    transport = CeleryAgentRunTransport(app)  # type: ignore[arg-type]
    assert isinstance(transport, AgentRunTransport)

    message_id = transport.publish(_task())

    assert message_id == "msg-1"
    assert len(app.sent) == 1
    sent = app.sent[0]
    assert sent["name"] == AGENT_RUN_TASK_NAME
    assert sent["kwargs"] == _task().as_payload()


def test_agent_run_and_validation_are_registered_as_separate_tasks() -> None:
    """两个职责共用一个任务名，会让「跑模型」和「跑容器」共用同一套超时与并发。"""

    app = _RecordingCelery()
    build_agent_run_task(app)  # type: ignore[arg-type]
    build_validation_task(app)  # type: ignore[arg-type]

    assert set(app.registered) == {AGENT_RUN_TASK_NAME, VALIDATION_TASK_NAME}


def test_worker_keeps_late_acks_and_single_prefetch() -> None:
    app = _RecordingCelery()
    build_agent_run_task(app)  # type: ignore[arg-type]

    assert app.conf.values["task_acks_late"] is True
    assert app.conf.values["task_reject_on_worker_lost"] is True
    assert app.conf.values["worker_prefetch_multiplier"] == 1
    assert app.conf.values["task_serializer"] == "json"


class _FakeValidationService:
    """假服务：只回答「谁被要求跑哪一条校验」。"""

    def __init__(self, *, unknown: set[str] | None = None) -> None:
        self.handled: list[str] = []
        self.after: list[ValidationOutcome] = []
        self.unknown = unknown or set()
        self.validation_worker = SimpleNamespace(handle=self._handle)
        self.validation_coordinator = SimpleNamespace(after_validation=self._after)

    def _handle(self, task_id: str) -> ValidationOutcome:
        if task_id in self.unknown:
            raise KeyError(f"校验任务不存在：{task_id}")
        self.handled.append(task_id)
        return ValidationOutcome(
            task_id=task_id,
            run_id="run-1",
            claimed=True,
            status=VALIDATION_PASSED,
            profile="python_compile",
        )

    def _after(self, outcome: ValidationOutcome) -> None:
        self.after.append(outcome)


class _FakeAgentService:
    """假服务：记录交给 AgentRunWorker 的那条消息。"""

    def __init__(self) -> None:
        self.tasks: list[AgentRunTask] = []

    def run_agent_task(self, task: AgentRunTask) -> AgentRunOutcome:
        self.tasks.append(task)
        return AgentRunOutcome(
            run_id=task.run_id,
            status="COMPLETED",
            execution=None,
            error_class=None,
            claimed=True,
        )


def test_validation_task_runs_only_the_registered_task() -> None:
    """任务体只认 task_id：profile 与路径从登记事实里读，不从消息里读。"""

    app = _RecordingCelery()
    service = _FakeValidationService()
    build_validation_task(app, service_factory=lambda: service)  # type: ignore[arg-type]
    task = app.registered[VALIDATION_TASK_NAME]

    result = task("validation-1")

    assert service.handled == ["validation-1"]
    assert [outcome.task_id for outcome in service.after] == ["validation-1"]
    assert result["status"] == VALIDATION_PASSED
    assert result["profile"] == "python_compile"
    assert result["claimed"] is True


def test_validation_task_never_fakes_a_result_for_an_unknown_task() -> None:
    app = _RecordingCelery()
    service = _FakeValidationService(unknown={"missing"})
    build_validation_task(app, service_factory=lambda: service)  # type: ignore[arg-type]
    task = app.registered[VALIDATION_TASK_NAME]

    with pytest.raises(KeyError):
        task("missing")

    # 没跑成的任务不进 continuation：Run 侧不该收到一个凭空的结论。
    assert service.after == []


def test_agent_run_task_delegates_to_the_run_worker() -> None:
    app = _RecordingCelery()
    service = _FakeAgentService()
    build_agent_run_task(app, service_factory=lambda: service)  # type: ignore[arg-type]
    task = app.registered[AGENT_RUN_TASK_NAME]

    result = task(**_task().as_payload())

    assert [received.run_id for received in service.tasks] == ["run-1"]
    assert service.tasks[0].task_kind == TASK_AGENT_RUN
    assert result == {
        "run_id": "run-1",
        "claimed": True,
        "status": "COMPLETED",
        "error_class": None,
    }
