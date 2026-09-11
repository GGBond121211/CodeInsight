"""Celery + Redis 承载后台任务：Agent Run 与固定 Sandbox validation。

两种任务的边界不同，所以是两个 task name，而不是一个任务里分叉：

    codeinsight.run_agent                  完整 Agent Run（AgentRunWorker 承载）
    codeinsight.run_registered_validation  固定的 Docker Sandbox 校验（ValidationWorker 承载）

合成一个任务，会让「跑模型」和「跑容器」共用一套超时、并发与重试设置，
而这两件事的最优取值恰好相反：模型调用慢且贵，容器校验快且可重放。
"""

from __future__ import annotations

import os
from collections.abc import Callable

from celery import Celery

from codeinsight.application.conversation_service import ConversationService
from codeinsight.domain.agent_run import AgentRunTask
from codeinsight.infrastructure.celery_agent_dispatcher import (
    AGENT_RUN_TASK_NAME,
    VALIDATION_TASK_NAME,
)
from codeinsight.infrastructure.otel import get_telemetry

# Worker 侧的服务工厂。测试注入假服务；部署时按环境装配。
ServiceFactory = Callable[[], ConversationService]


def _configured_app(app: Celery | None = None) -> Celery:
    """建（或配置）Celery app。acks_late 与 prefetch=1 是这里的硬要求。"""

    selected = app or Celery(
        "codeinsight",
        broker=os.environ.get("CODEINSIGHT_CELERY_BROKER_URL", "redis://127.0.0.1:6380/1"),
        backend=os.environ.get("CODEINSIGHT_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6380/2"),
    )
    selected.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )
    return selected


def build_worker_conversation_service() -> ConversationService:
    """Worker 进程侧的服务组装。

    用与 API 相同的装配（create_app），不另写一份：两条装配路径一旦出现差别，
    Worker 执行的就不是用户受理时承诺的那个流程。续跑通道换成 broker——本进程只
    执行这一次尝试，API 进程里的那一轮状态不该被它改写。

    缺 MySQL 配置时直接失败：静默退回内存 Store，会让 Worker 在一个空队列上工作
    却报告一切正常，这比报错更难查。
    """

    from codeinsight.api.app import create_app
    from codeinsight.infrastructure.celery_agent_dispatcher import (
        CeleryAgentRunTransport,
        CeleryValidationTransport,
    )
    from codeinsight.infrastructure.db.engine import MySqlConfig

    if MySqlConfig.from_env() is None:
        raise RuntimeError(
            "未配置 CODEINSIGHT_MYSQL_*：Worker 需要可从事实恢复的 Run/Session Store。"
        )
    assembled = create_app(
        agent_run_transport=CeleryAgentRunTransport(celery_app),
        validation_transport=CeleryValidationTransport(celery_app),
    )
    return assembled.state.codeinsight_conversation


def build_validation_task(
    app: Celery | None = None, service_factory: ServiceFactory | None = None
):
    """注册 codeinsight.run_registered_validation：跑一条登记过的固定校验。

    消息里只有 task_id。profile 与要校验的隔离 workspace 都从队列事实里读——投递方
    在消息里写什么就执行什么，等于把「跑哪条命令」的决定权交给了消息本身。
    """

    selected = _configured_app(app)
    build_service = service_factory or build_worker_conversation_service

    @selected.task(name=VALIDATION_TASK_NAME)
    def run_registered_validation(task_id: str) -> dict[str, object]:
        service = build_service()
        with get_telemetry().span(
            "worker", "sandbox_validation", attributes={"validation_task": task_id}
        ):
            outcome = service.validation_worker.handle(task_id)
            service.validation_coordinator.after_validation(outcome)
        return {
            "task_id": outcome.task_id,
            "run_id": outcome.run_id,
            "claimed": outcome.claimed,
            "status": outcome.status,
            "profile": outcome.profile,
            "error_class": outcome.error_class,
        }

    return run_registered_validation


def build_agent_run_task(
    app: Celery | None = None, service_factory: ServiceFactory | None = None
):
    """注册 codeinsight.run_agent：消息到达即执行，不在这里自己造终态。

    领取租约、重建上下文、写状态都在 AgentRunWorker 里，与进程内路径是同一段代码。
    这个任务体只做翻译：把消息还原成 AgentRunTask，然后交出执行权。任务载荷只有
    标识与版本，正文从事实 Store 读。
    """

    selected = _configured_app(app)
    build_service = service_factory or build_worker_conversation_service

    @selected.task(name=AGENT_RUN_TASK_NAME)
    def run_agent(**payload: object) -> dict[str, object]:
        task = AgentRunTask.from_payload(payload)
        outcome = build_service().run_agent_task(task)
        return {
            "run_id": outcome.run_id,
            "claimed": outcome.claimed,
            "status": outcome.status,
            "error_class": outcome.error_class,
        }

    return run_agent


celery_app = _configured_app()
run_registered_validation = build_validation_task(celery_app)
run_agent = build_agent_run_task(celery_app)

