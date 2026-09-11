"""Q-011 Task 11：Agent Run 执行队列的容量与背压实验。

这个脚本回答的是「队列能收多少、什么时候必须拒、拒了之后还剩什么」，
不是「模型有多快」。因此：

    真实的部分    MySQL 事实层、租约 CAS、Redis broker 投递、队列上限与拒绝。
    假的部分      模型调用（Fake Provider 不在这里——脚本根本不调模型，
                  它只把 Run 推进到终态，测的是排队与领取，不是推理）。

边界必须写在结果里：单进程多线程 Worker 不等于多进程 Worker，本机 Redis
不等于线上 Redis。没有跑过的组合在结果 JSON 的 not_executed 里如实列出。

用法（在 backend 目录下）：

    .venv\\Scripts\\python.exe tests\\evals\\agent_run_queue_benchmark.py
    .venv\\Scripts\\python.exe tests\\evals\\agent_run_queue_benchmark.py \\
        --scenario limit300-burst450
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from codeinsight.agent.worker_tasks import celery_app  # noqa: E402
from codeinsight.application.agent_run_dispatcher import (  # noqa: E402
    QUEUE_FULL,
    AgentRunDispatcher,
    QueueFullRejected,
)
from codeinsight.domain.agent_run import (  # noqa: E402
    COMPLETED,
    QUEUED,
    TASK_AGENT_RUN,
    AgentRunTask,
)
from codeinsight.infrastructure.celery_agent_dispatcher import (  # noqa: E402
    CeleryAgentRunTransport,
)
from codeinsight.infrastructure.db.engine import (  # noqa: E402
    ENV_HOST,
    ENV_PASSWORD,
    ENV_PORT,
    ENV_USER,
    MySqlConfig,
    create_all_tables,
    create_db_engine,
    create_session_factory,
)
from codeinsight.infrastructure.db.stores import (  # noqa: E402
    MySqlAgentRunStore,
    truncate_all_tables,
)

ENV_TEST_DATABASE = "CODEINSIGHT_MYSQL_TEST_DATABASE"
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "configs" / "agent_run_worker_pressure.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "Q011-AGENT-RUN-QUEUE"

LEASE_MS = 5_000


@dataclass(frozen=True)
class Scenario:
    """一个要跑的组合。字段直接来自 experiments/configs/agent_run_worker_pressure.yaml。"""

    name: str
    burst: int
    workers: int
    queue_limit: int | None
    occupancy_ms: int
    submit_threads: int = 16
    kind: str = "burst"


def _mysql_config() -> MySqlConfig:
    """和集成测试同一套变量：只连测试库，库名必须含 test。"""

    host = os.environ.get(ENV_HOST, "").strip()
    user = os.environ.get(ENV_USER, "").strip()
    database = os.environ.get(ENV_TEST_DATABASE, "").strip()
    if not host or not user or not database:
        raise SystemExit(
            f"未配置 {ENV_HOST} / {ENV_USER} / {ENV_TEST_DATABASE}："
            "这个实验要真 MySQL，不提供内存退路（内存实现证明不了租约 CAS）。"
        )
    if "test" not in database.lower():
        raise SystemExit(
            f"{ENV_TEST_DATABASE}={database!r} 的库名里没有 'test'。"
            "实验会清空全部表——拒绝对一个看起来不是测试库的目标执行。"
        )
    raw_port = os.environ.get(ENV_PORT, "").strip()
    return MySqlConfig(
        host=host,
        port=int(raw_port) if raw_port else 3306,
        user=user,
        password=os.environ.get(ENV_PASSWORD, ""),
        database=database,
    )


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(percentile * len(ordered) + 0.9999) - 1))
    return round(ordered[index], 2)


def _redis_latency_report(samples: int = 40) -> dict[str, Any]:
    """真实 Redis 的往返延迟。broker 就在这里，不能只写「应该很快」。"""

    import redis

    url = os.environ.get("CODEINSIGHT_REDIS_URL", "redis://127.0.0.1:6380/0")
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
    timings: list[float] = []
    errors: list[str] = []
    for index in range(samples):
        started = time.perf_counter()
        try:
            client.set(f"codeinsight:bench:q011:{index}", str(index), ex=30)
            client.get(f"codeinsight:bench:q011:{index}")
        except Exception as exc:  # pragma: no cover - 只有环境坏掉才会走到
            errors.append(type(exc).__name__)
            continue
        timings.append((time.perf_counter() - started) * 1000)
    for index in range(samples):
        client.delete(f"codeinsight:bench:q011:{index}")
    return {
        "url": url,
        "samples": len(timings),
        "errors": sorted(set(errors)),
        "p50_ms": _percentile(timings, 0.50),
        "p95_ms": _percentile(timings, 0.95),
        "max_ms": round(max(timings), 2) if timings else None,
    }


def _task(index: int, scenario: str) -> AgentRunTask:
    return AgentRunTask(
        task_id=f"task-{scenario}-{index}",
        session_id=f"session-{scenario}-{index}",
        turn_id=f"turn-{scenario}-{index}",
        run_id=f"run-{scenario}-{index}",
        task_kind=TASK_AGENT_RUN,
        idempotency_key=f"turn-{scenario}-{index}",
        policy_version="q011-queue-benchmark-v1",
        deadline_epoch_ms=int(time.time() * 1000) + 10 * 60 * 1000,
    )


class _Worker:
    """一个进程内 Worker：轮询等待领取的 Run，抢到租约就占住一段时间再收尾。

    它模仿的是真实 Worker 的两个动作：claim_run（租约 CAS）与写回终态；
    真正跑模型与工具的那一段被替换成固定占用时间，因为这里测的是排队。
    """

    def __init__(
        self,
        *,
        store: MySqlAgentRunStore,
        name: str,
        occupancy_ms: int,
        stop: threading.Event,
        claims: dict[str, int],
        claims_lock: threading.Lock,
    ) -> None:
        self._store = store
        self._name = name
        self._occupancy_ms = occupancy_ms
        self._stop = stop
        self._claims = claims
        self._claims_lock = claims_lock
        self.claim_times: list[tuple[str, int]] = []

    def run(self) -> None:
        while not self._stop.is_set():
            record = self._claim_next()
            if record is None:
                time.sleep(0.002)
                continue
            claimed_at_ms = int(time.time() * 1000)
            self.claim_times.append((record.run_id, claimed_at_ms))
            time.sleep(self._occupancy_ms / 1000)
            finished = record.advanced(
                status=COMPLETED, updated_at_epoch_ms=int(time.time() * 1000)
            )
            self._store.save_run(finished)

    def _claim_next(self):
        for candidate in self._store.list_open_runs(limit=400):
            if candidate.status != QUEUED:
                continue
            claimed = self._store.claim_run(
                candidate.run_id,
                worker_id=self._name,
                lease_until_epoch_ms=int(time.time() * 1000) + LEASE_MS,
            )
            if claimed is None:
                continue
            with self._claims_lock:
                self._claims[claimed.run_id] = self._claims.get(claimed.run_id, 0) + 1
            return claimed
        return None


def _run_burst(
    *,
    scenario: Scenario,
    store: MySqlAgentRunStore,
    dispatcher: AgentRunDispatcher,
) -> dict[str, Any]:
    created_ms: dict[str, int] = {}
    claims: dict[str, int] = {}
    claims_lock = threading.Lock()
    stop = threading.Event()
    rejections = 0
    accepted = 0
    submit_errors: list[str] = []
    dispatch_ms: list[float] = []
    dispatch_lock = threading.Lock()

    workers = [
        _Worker(
            store=store,
            name=f"worker-{index}",
            occupancy_ms=scenario.occupancy_ms,
            stop=stop,
            claims=claims,
            claims_lock=claims_lock,
        )
        for index in range(scenario.workers)
    ]
    worker_threads = [threading.Thread(target=worker.run, daemon=True) for worker in workers]
    for thread in worker_threads:
        thread.start()

    def submit(index: int) -> None:
        nonlocal rejections, accepted
        task = _task(index, scenario.name)
        started_ms = time.perf_counter()
        try:
            record = dispatcher.dispatch(task)
        except QueueFullRejected:
            with dispatch_lock:
                dispatch_ms.append((time.perf_counter() - started_ms) * 1000)
            rejections += 1
            return
        except Exception as exc:  # pragma: no cover - 只有环境坏掉才会走到
            submit_errors.append(type(exc).__name__)
            return
        with dispatch_lock:
            dispatch_ms.append((time.perf_counter() - started_ms) * 1000)
        if record.status == QUEUED:
            accepted += 1
            created_ms[record.run_id] = record.updated_at_epoch_ms
        else:
            rejections += 1

    max_queued = 0
    sampler_stop = threading.Event()

    def sample_queue() -> None:
        nonlocal max_queued
        while not sampler_stop.is_set():
            try:
                max_queued = max(max_queued, store.count_queued_runs())
            except Exception:  # pragma: no cover - 采样不该拖垮实验
                pass
            time.sleep(0.01)

    sampler = threading.Thread(target=sample_queue, daemon=True)
    sampler.start()

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(scenario.burst, scenario.submit_threads)) as pool:
        list(pool.map(submit, range(scenario.burst)))
    dispatch_seconds = time.perf_counter() - started

    # 排空：等所有被受理的 Run 都离开等待队列，而不是等一个拍脑袋的固定秒数。
    drain_deadline = time.perf_counter() + 120
    while time.perf_counter() < drain_deadline:
        if store.count_queued_runs() == 0:
            break
        time.sleep(0.02)
    drained_at = time.perf_counter()
    stop.set()
    sampler_stop.set()
    for thread in worker_threads:
        thread.join(timeout=10)
    sampler.join(timeout=5)

    claim_times = dict(
        item for worker in workers for item in worker.claim_times
    )
    queue_waits = [
        float(claim_times[run_id] - created_ms[run_id])
        for run_id in claim_times
        if run_id in created_ms
    ]
    unfinished = len(created_ms) - len(claim_times)
    duplicate_claims = sum(1 for count in claims.values() if count > 1)
    # 收尾时间：被受理的 Run 全部离开等待队列花了多久（含领取与占用）。
    makespan_seconds = drained_at - started
    return {
        "scenario": scenario.name,
        "queue_limit": scenario.queue_limit,
        "burst": scenario.burst,
        "workers": scenario.workers,
        "submit_threads": min(scenario.burst, scenario.submit_threads),
        "occupancy_ms": scenario.occupancy_ms,
        "accepted": accepted,
        "rejected": rejections,
        "rejected_error_class": QUEUE_FULL if rejections else None,
        "submit_errors": sorted(set(submit_errors)),
        "max_queued_observed": max_queued,
        "queue_wait_ms": {
            "p50": _percentile(queue_waits, 0.50),
            "p95": _percentile(queue_waits, 0.95),
            "p99": _percentile(queue_waits, 0.99),
            "max": round(max(queue_waits), 2) if queue_waits else None,
        },
        "unfinished_runs": unfinished,
        "duplicate_claims": duplicate_claims,
        "dispatch_seconds": round(dispatch_seconds, 3),
        "dispatch_ms": {
            "p50": _percentile(dispatch_ms, 0.50),
            "p95": _percentile(dispatch_ms, 0.95),
            "p99": _percentile(dispatch_ms, 0.99),
            "max": round(max(dispatch_ms), 2) if dispatch_ms else None,
        },
        "drain_seconds": round(makespan_seconds - dispatch_seconds, 3),
        "makespan_seconds": round(makespan_seconds, 3),
        "throughput_per_second": round(accepted / makespan_seconds, 1)
        if makespan_seconds > 0 and accepted
        else None,
    }


def _run_recovery(*, scenario: Scenario, store: MySqlAgentRunStore) -> dict[str, Any]:
    """Worker 在持有租约时消失：另一个 Worker 多久能把它接管回来。

    这里不杀进程，而是让第一段租约故意过期并且不再续期——对事实层来说，
    「Worker 死了」和「租约不续了」是同一件事，因为心跳就是租约。
    """

    dispatcher = AgentRunDispatcher(
        store=store,
        transport=CeleryAgentRunTransport(celery_app),
        queue_limit=None,
    )
    task = _task(0, scenario.name)
    dispatcher.dispatch(task)

    lease_ms = scenario.occupancy_ms
    dead = store.claim_run(
        task.run_id,
        worker_id="worker-dead",
        lease_until_epoch_ms=int(time.time() * 1000) + lease_ms,
    )
    assert dead is not None
    lease_expiry_ms = dead.lease_until_epoch_ms or 0

    while int(time.time() * 1000) <= lease_expiry_ms:
        time.sleep(0.005)

    takeover = store.claim_run(
        task.run_id,
        worker_id="worker-alive",
        lease_until_epoch_ms=int(time.time() * 1000) + LEASE_MS,
    )
    takeover_ms = int(time.time() * 1000)
    assert takeover is not None
    store.save_run(
        takeover.advanced(status=COMPLETED, updated_at_epoch_ms=takeover_ms)
    )
    return {
        "scenario": scenario.name,
        "kind": "recovery",
        "lease_ms": lease_ms,
        "recovery_ms": takeover_ms - lease_expiry_ms,
        "takeover_worker": takeover.worker_id,
        "status_reached": COMPLETED,
    }


def _load_scenarios(path: Path) -> tuple[dict[str, Any], list[Scenario]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    scenarios = [
        Scenario(
            name=item["name"],
            burst=int(item["burst"]),
            workers=int(item["workers"]),
            queue_limit=item.get("queue_limit"),
            occupancy_ms=int(item["occupancy_ms"]),
            submit_threads=int(item.get("submit_threads", 16)),
            kind=item.get("kind", "burst"),
        )
        for item in raw["scenarios"]
    ]
    return raw, scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scenario", action="append", default=None)
    parser.add_argument("--label", default="local")
    arguments = parser.parse_args(argv)

    raw, scenarios = _load_scenarios(arguments.config)
    if arguments.scenario:
        wanted = set(arguments.scenario)
        scenarios = [item for item in scenarios if item.name in wanted]
        missing = wanted - {item.name for item in scenarios}
        if missing:
            raise SystemExit(f"配置里没有这些 scenario：{sorted(missing)}")

    config = _mysql_config()
    engine = create_db_engine(config)
    create_all_tables(engine)
    session_factory = create_session_factory(engine)

    results: list[dict[str, Any]] = []
    for scenario in scenarios:
        truncate_all_tables(engine)
        store = MySqlAgentRunStore(session_factory)
        dispatcher = AgentRunDispatcher(
            store=store,
            transport=CeleryAgentRunTransport(celery_app),
            queue_limit=scenario.queue_limit,
        )
        print(
            f"[q011] 开始 {scenario.name}（burst={scenario.burst}, "
            f"limit={scenario.queue_limit}）"
        )
        if scenario.kind == "recovery":
            outcome = _run_recovery(scenario=scenario, store=store)
        else:
            outcome = _run_burst(scenario=scenario, store=store, dispatcher=dispatcher)
        print(f"[q011] {scenario.name} -> {json.dumps(outcome, ensure_ascii=False)}")
        results.append(outcome)

    redis_report = _redis_latency_report()
    engine.dispose()

    payload = {
        "experiment": "Q-011-TASK-11-AGENT-RUN-QUEUE",
        "label": arguments.label,
        "recorded_at": datetime.now(UTC).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mysql": config.safe_url,
            "worker_model": "单进程多线程（不是多进程 Worker）",
            "broker": "真实 Redis（Celery send_task）",
            "model": "未调用（本实验只测排队与领取）",
            "note": (
                "受理路径每次要 3-5 次 MySQL 往返，本机单次简单查询 p50 约 1.6ms、"
                "新建连接约 64ms；绝对延迟受这台机器的 Docker/WSL 环境影响，"
                "请看同一轮内的相对比较，不要当成线上容量。"
            ),
        },
        "config_source": str(arguments.config.relative_to(PROJECT_ROOT)),
        "config_matrix": raw.get("matrix", {}),
        "results": results,
        "redis": redis_report,
        "not_executed": raw.get("not_executed", []),
    }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = OUTPUT_ROOT / f"{stamp}-{arguments.label}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[q011] 结果写入 {target}")

    waits = [
        item["queue_wait_ms"]["p95"]
        for item in results
        if item.get("queue_wait_ms", {}).get("p95") is not None
    ]
    if waits:
        print(
            f"[q011] 各场景 queue wait P95：最大 {max(waits)} ms"
            f"（中位 {statistics.median(waits)} ms）"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
