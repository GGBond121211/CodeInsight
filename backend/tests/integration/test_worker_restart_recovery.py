import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from redis import Redis
from redis.exceptions import RedisError

from codeinsight.agent.worker_tasks import build_validation_task
from codeinsight.infrastructure.task_queue import (
    InMemoryTaskQueue,
    RedisTaskQueue,
    TaskEnvelope,
)


def _task() -> TaskEnvelope:
    return TaskEnvelope(
        task_id="task-1",
        run_id="run-1",
        session_id="session-1",
        idempotency_key="run-1:python_compile",
        task_type="sandbox_validation",
        payload={"profile": "python_compile", "workspace_path": "C:/managed/workspace"},
        deadline_epoch_ms=10_000,
        max_attempts=2,
    )


def test_expired_worker_lease_is_recovered_without_duplicate_task() -> None:
    queue = InMemoryTaskQueue()
    queue.submit(_task())
    first = queue.claim("worker-a", now_epoch_ms=100, lease_ms=50)
    recovered = queue.claim("worker-b", now_epoch_ms=151, lease_ms=50)

    assert first is not None and first.attempt == 1
    assert recovered is not None and recovered.task_id == first.task_id
    assert recovered.attempt == 2
    assert len(queue.list_tasks()) == 1


def test_non_retryable_worker_failure_goes_to_manual_required() -> None:
    queue = InMemoryTaskQueue()
    queue.submit(_task())
    claimed = queue.claim("worker-a", now_epoch_ms=100, lease_ms=50)
    assert claimed is not None

    failed = queue.fail(claimed.task_id, "PERMISSION", retryable=False)

    assert failed.status == "MANUAL_REQUIRED"


def test_celery_task_accepts_only_registered_validation_profile() -> None:
    task = build_validation_task()
    assert task.name == "codeinsight.run_registered_validation"
    assert task.app.conf.task_acks_late is True
    assert task.app.conf.task_reject_on_worker_lost is True
    assert task.app.conf.worker_prefetch_multiplier == 1


def test_redis_queue_survives_adapter_restart_and_recovers_expired_lease() -> None:
    client = Redis.from_url("redis://127.0.0.1:6380/3", decode_responses=True)
    try:
        client.ping()
    except RedisError as error:
        pytest.skip(f"Redis unavailable: {error}")
    prefix = f"codeinsight:test:step7:{uuid.uuid4().hex}"
    before_restart = RedisTaskQueue(client, prefix=prefix)
    try:
        before_restart.submit(_task())
        first = before_restart.claim("worker-a", now_epoch_ms=100, lease_ms=50)
        assert first is not None and first.attempt == 1

        after_restart = RedisTaskQueue(client, prefix=prefix)
        recovered = after_restart.claim("worker-b", now_epoch_ms=151, lease_ms=50)

        assert recovered is not None
        assert recovered.task_id == "task-1"
        assert recovered.attempt == 2
        assert len(after_restart.list_tasks()) == 1
    finally:
        before_restart.clear_test_namespace()


def test_real_worker_process_kill_and_restart_recovers_same_task(tmp_path: Path) -> None:
    client = Redis.from_url("redis://127.0.0.1:6380/3", decode_responses=True)
    try:
        client.ping()
    except RedisError as error:
        pytest.skip(f"Redis unavailable: {error}")
    prefix = f"codeinsight:test:process:{uuid.uuid4().hex}"
    queue = RedisTaskQueue(client, prefix=prefix)
    queue.submit(_task())
    marker = tmp_path / "claimed.txt"
    project = Path(__file__).resolve().parents[3]
    pythonpath = str(project / "backend" / "src")
    script = (
        "import sys,time; from redis import Redis; "
        "from codeinsight.infrastructure.task_queue import RedisTaskQueue; "
        "q=RedisTaskQueue(Redis.from_url(sys.argv[1],decode_responses=True),prefix=sys.argv[2]); "
        "t=q.claim(sys.argv[3],now_epoch_ms=int(sys.argv[4]),lease_ms=100); "
        "open(sys.argv[5],'w',encoding='utf-8').write(t.task_id if t else 'none'); "
        "time.sleep(30)"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = pythonpath
    first = subprocess.Popen(
        [
            sys.executable,
            "-c",
            script,
            "redis://127.0.0.1:6380/3",
            prefix,
            "worker-a",
            "100",
            str(marker),
        ],
        env=environment,
    )
    try:
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert marker.read_text(encoding="utf-8") == "task-1"
        first.kill()
        first.wait(timeout=10)

        recovered = queue.claim("worker-b", now_epoch_ms=201, lease_ms=100)
        assert recovered is not None
        assert recovered.task_id == "task-1"
        assert recovered.attempt == 2
        assert queue.complete(recovered.task_id).status == "COMPLETED"
        assert len(queue.list_tasks()) == 1
    finally:
        if first.poll() is None:
            first.kill()
        queue.clear_test_namespace()
