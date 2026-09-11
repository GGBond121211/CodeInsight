"""Celery + Redis 承载后台任务：Agent Run 与固定 Sandbox validation。

两种任务的边界不同，所以是两个 task name，而不是一个任务里分叉：

    codeinsight.run_agent                  完整 Agent Run（AgentRunWorker 承载）
    codeinsight.run_registered_validation  固定的 Docker Sandbox 校验（ValidationWorker 承载）

合成一个任务，会让「跑模型」和「跑容器」共用一套超时、并发与重试设置，
而这两件事的最优取值恰好相反：模型调用慢且贵，容器校验快且可重放。
"""

from __future__ import annotations

import os
import time

from celery import Celery

from codeinsight.domain.agent_run import MANUAL_REQUIRED, AgentRunTask
from codeinsight.domain.ports import AgentRunStore
from codeinsight.infrastructure.celery_agent_dispatcher import AGENT_RUN_TASK_NAME
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.sandbox import DockerSandbox

# 领取租约的默认长度。Worker 崩掉后，超过这个时间租约才允许被别人接管：
# 太短会把还在跑的 Run 抢走，太长会让恢复变慢。具体取值由并发实验决定（Task 11）。
DEFAULT_LEASE_MS = 60_000

# 执行体还没接上时的公开错误类别。Task 5 会用只读 Tool Loop 替换这一段。
ERROR_AGENT_RUN_NOT_IMPLEMENTED = "AGENT_RUN_NOT_IMPLEMENTED"


def build_agent_run_store() -> AgentRunStore:
    """Worker 侧的 Run 事实层。

    没配 MySQL 就直接失败，不退回内存实现：任务一旦交给另一个进程，进程内存里的
    Store 就什么都证明不了，恢复巡检会看到一个空队列而报不出任何错。
    """

    from codeinsight.infrastructure.db.engine import (
        MySqlConfig,
        create_db_engine,
        create_session_factory,
    )
    from codeinsight.infrastructure.db.stores import MySqlAgentRunStore

    mysql = MySqlConfig.from_env()
    if mysql is None:
        raise RuntimeError(
            "未配置 CODEINSIGHT_MYSQL_*：Agent Run Worker 需要可恢复的事实 Store。"
        )
    engine = create_db_engine(mysql)
    return MySqlAgentRunStore(create_session_factory(engine))


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


def build_validation_task(app: Celery | None = None):
    selected = _configured_app(app)

    @selected.task(name="codeinsight.run_registered_validation")
    def run_registered_validation(profile: str, workspace_path: str) -> dict[str, object]:
        with get_telemetry().span(
            "worker", "sandbox_validation", attributes={"validation_profile": profile}
        ):
            result = DockerSandbox().run(profile, workspace_path)
        return {
            "profile": result.profile,
            "passed": result.passed,
            "error_class": result.error_class,
            "message_excerpt": result.message_excerpt,
        }

    return run_registered_validation


def build_agent_run_task(app: Celery | None = None):
    """注册 codeinsight.run_agent：先领取 Run，再把工作交给执行体。

    领取这一步是幂等的最后一道闸：重复投递同一条消息时，第二个 Worker 拿不到租约，
    于是不会跑第二遍模型。任务载荷只有标识与版本，正文从 Session 里读。
    """

    selected = _configured_app(app)

    @selected.task(name=AGENT_RUN_TASK_NAME)
    def run_agent(**payload: object) -> dict[str, object]:
        task = AgentRunTask.from_payload(payload)
        store = build_agent_run_store()
        now_epoch_ms = int(time.time() * 1000)
        record = store.claim_run(
            task.run_id,
            worker_id=f"celery-{os.getpid()}",
            lease_until_epoch_ms=now_epoch_ms + DEFAULT_LEASE_MS,
        )
        if record is None:
            # 有人持有有效租约，或这个 Run 处于不该被领取的状态。
            return {"run_id": task.run_id, "claimed": False}

        # Task 5 在这里接上只读 Tool Loop。现在写下一个明确的「未实现」，而不是让
        # Run 停在 RUNNING 上等租约过期——说不清的状态必须能被人看见。
        stopped = record.advanced(
            status=MANUAL_REQUIRED,
            updated_at_epoch_ms=int(time.time() * 1000),
            error_class=ERROR_AGENT_RUN_NOT_IMPLEMENTED,
        )
        store.save_run(stopped)
        return {"run_id": task.run_id, "claimed": True, "status": stopped.status}

    return run_agent


celery_app = _configured_app()
run_registered_validation = build_validation_task(celery_app)
run_agent = build_agent_run_task(celery_app)

