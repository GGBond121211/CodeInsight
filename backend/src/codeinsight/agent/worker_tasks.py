"""Celery + Redis 承载固定 Sandbox validation 的 Worker 入口。"""

from __future__ import annotations

import os

from celery import Celery

from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.sandbox import DockerSandbox


def build_validation_task(app: Celery | None = None):
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


celery_app = Celery(
    "codeinsight",
    broker=os.environ.get("CODEINSIGHT_CELERY_BROKER_URL", "redis://127.0.0.1:6380/1"),
    backend=os.environ.get("CODEINSIGHT_CELERY_RESULT_BACKEND", "redis://127.0.0.1:6380/2"),
)
run_registered_validation = build_validation_task(celery_app)
