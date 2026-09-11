"""Q-011 Task 3：Celery 投递与 Worker 注册的单元测试。

不需要 broker：这些用例验证的是「消息里装了什么」「注册了哪些任务名」，
以及 Worker 配置仍是 acks_late 与 prefetch=1。
"""

from __future__ import annotations

from types import SimpleNamespace

from codeinsight.agent.worker_tasks import build_agent_run_task, build_validation_task
from codeinsight.application.agent_run_dispatcher import AgentRunTransport
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

