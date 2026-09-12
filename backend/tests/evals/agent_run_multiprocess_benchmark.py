r"""Q-011 收尾：多进程 Worker 的竞争与容量实验（真 Celery 进程，假模型）。

Task 11 那份实验的 Worker 是「同进程 4 个线程」，所以「多进程租约竞争、真实 Celery
的 prefetch、逐进程资源归因」三项只能写「没测」。这份补上那三项：

    真的    独立 Worker 进程（celery worker --pool=solo）、真 MySQL 事实层、
            真 Redis broker、真租约 CAS、真重复投递、真杀进程。
    假的    模型与工具（不调用，占用时间固定），与 Task 11 保持一致——
            本实验的结论只覆盖调度与竞争，不能外推成模型能力。

两个场景：

    compete  并发竞争：两个进程同时抢同一批 Run，并且每条消息投两次，
             看重复投递会不会跑两遍（期望：跑一遍）。
    kill     进程被杀：杀掉持有租约的进程，然后按 broker 重投的方式再投一次，
             看租约期内的重投会不会重放（期望：不会），以及租约过期后
             另一个进程能不能接管（期望：能）。

用法（backend 目录）：

    .venv\Scripts\python.exe tests\evals\agent_run_multiprocess_benchmark.py --scenario compete
    .venv\Scripts\python.exe tests\evals\agent_run_multiprocess_benchmark.py --scenario kill

需要 CODEINSIGHT_MYSQL_HOST / CODEINSIGHT_MYSQL_USER / CODEINSIGHT_MYSQL_TEST_DATABASE
（库名必须含 test）与真 Redis（默认 redis://127.0.0.1:6380/5，用独立 db，不会碰开发队列）。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import redis

BACKEND_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_ROOT.parent
EVALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT / "src"))

from celery import Celery  # noqa: E402

from codeinsight.application.agent_run_dispatcher import AgentRunDispatcher  # noqa: E402
from codeinsight.domain.agent_run import (  # noqa: E402
    COMPLETED,
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
DEFAULT_BROKER = "redis://127.0.0.1:6380/5"
DEFAULT_QUEUE = "q011-mp-eval"
OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "results" / "Q011-AGENT-RUN-QUEUE"
WORKER_ENTRY = "q011_mp_worker_entry:celery_app"


def _mysql_config() -> MySqlConfig:
    """和集成测试同一套变量：只连测试库，库名必须含 test。"""

    host = os.environ.get(ENV_HOST, "").strip()
    user = os.environ.get(ENV_USER, "").strip()
    database = os.environ.get(ENV_TEST_DATABASE, "").strip()
    if not host or not user or not database:
        raise SystemExit(
            f"未配置 {ENV_HOST} / {ENV_USER} / {ENV_TEST_DATABASE}："
            "这个实验要真 MySQL，不提供内存退路（内存实现证明不了跨进程租约 CAS）。"
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


def _eval_app(broker: str, queue: str) -> Celery:
    """实验自己的 Celery app：独立队列 + 与真实 Worker 同一套可靠性与 prefetch 设置。"""

    app = Celery("codeinsight-q011-mp-harness", broker=broker, backend=None)
    app.conf.update(
        task_default_queue=queue,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
    )
    return app


def _redis_client(broker: str) -> redis.Redis:
    return redis.Redis.from_url(broker, decode_responses=True, socket_timeout=2)


def _purge(app: Celery, client: redis.Redis, queue: str) -> None:
    """清掉上一轮实验残留：队列、未确认消息、控制通道。

    只允许清一个独立的实验 db：默认 db 5，且必须在本机。flushdb 是粗动作，
    所以要挡住「指到开发库」这种误用。
    """

    connection = client.connection_pool.connection_kwargs
    if connection.get("host") not in {"127.0.0.1", "localhost"} or int(
        connection.get("db") or 0
    ) == 0:
        raise SystemExit("拒绝清理：broker 必须指向本机且不是 db 0（避免碰到开发队列）。")
    client.flushdb()
    app.control.purge()


def _task(index: int, scenario: str) -> AgentRunTask:
    return AgentRunTask(
        task_id=f"task-{scenario}-{index}",
        session_id=f"session-{scenario}-{index}",
        turn_id=f"turn-{scenario}-{index}",
        run_id=f"run-{scenario}-{index}",
        task_kind=TASK_AGENT_RUN,
        idempotency_key=f"turn-{scenario}-{index}",
        policy_version="q011-multiprocess-v1",
        deadline_epoch_ms=int(time.time() * 1000) + 10 * 60 * 1000,
    )


class _Workers:
    """一组真实 Celery Worker 进程。"""

    def __init__(
        self,
        *,
        count: int,
        run_dir: Path,
        base_env: dict[str, str],
        queue: str,
        broker: str,
        occupancy_ms: int,
        lease_ms: int,
    ) -> None:
        self._run_dir = run_dir
        self._ledgers: list[Path] = []
        self._processes: list[subprocess.Popen[bytes]] = []
        self._names: list[str] = []
        for index in range(count):
            name = f"q011mp-{index}"
            ledger = run_dir / f"ledger-{name}.jsonl"
            environment = dict(base_env)
            environment.update(
                {
                    "CODEINSIGHT_Q011_MP_BROKER": broker,
                    "CODEINSIGHT_Q011_MP_QUEUE": queue,
                    "CODEINSIGHT_Q011_MP_WORKER_ID": name,
                    "CODEINSIGHT_Q011_MP_LEDGER": str(ledger),
                    "CODEINSIGHT_Q011_MP_OCCUPANCY_MS": str(occupancy_ms),
                    "CODEINSIGHT_Q011_MP_LEASE_MS": str(lease_ms),
                    "PYTHONPATH": os.pathsep.join(
                        [str(BACKEND_ROOT / "src"), str(EVALS_DIR)]
                    ),
                }
            )
            log = (run_dir / f"worker-{name}.log").open("wb")
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "celery",
                    "-A",
                    WORKER_ENTRY,
                    "worker",
                    "--pool=solo",
                    "--concurrency=1",
                    "--loglevel=WARNING",
                    "-Q",
                    queue,
                    "-n",
                    f"{name}@%h",
                ],
                cwd=EVALS_DIR,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            self._ledgers.append(ledger)
            self._processes.append(process)
            self._names.append(name)
        self._logs = [entry for entry in self._run_dir.glob("worker-*.log")]

    @property
    def names(self) -> list[str]:
        return list(self._names)

    @property
    def pid_of(self) -> dict[str, int]:
        return dict(zip(self._names, [item.pid for item in self._processes], strict=True))

    def wait_ready(self, *, timeout_s: float = 120) -> list[str]:
        """等每个进程把 ready 写进账本。比 ping 更直接：账本就是本进程写的。"""

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ready = {
                line.get("worker_id")
                for line in self.read_ledger()
                if line.get("kind") == "ready"
            }
            if len(ready) >= len(self._names):
                return sorted(str(item) for item in ready)
            if any(item.poll() is not None for item in self._processes):
                raise SystemExit(
                    "有 Worker 进程在就绪前退出，看 "
                    + ", ".join(str(path) for path in self._logs)
                )
            time.sleep(0.1)
        raise SystemExit(f"{timeout_s} 秒内没有等到全部 Worker 就绪。")

    def kill(self, name: str) -> float:
        """杀掉一个 Worker 进程（硬杀，等价于进程崩溃），返回杀掉的时刻。"""

        index = self._names.index(name)
        killed_at = time.time() * 1000
        process = self._processes[index]
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)
        return killed_at

    def stop(self) -> None:
        for process in self._processes:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass

    def read_ledger(self) -> list[dict[str, Any]]:
        lines: list[dict[str, Any]] = []
        for path in self._ledgers:
            if not path.exists():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                if raw.strip():
                    lines.append(json.loads(raw))
        return lines


class _Sampler:
    """采样队列深度与未确认消息数。

    未确认数（Redis 的 unacked）是 prefetch 的直接证据：prefetch=1 时它不该长起来。
    """

    def __init__(self, client: redis.Redis, queue: str, *, interval_s: float = 0.02) -> None:
        self._client = client
        self._queue = queue
        self._interval = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.queue_depth: list[int] = []
        self.unacked: list[int] = []

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.queue_depth.append(int(self._client.llen(self._queue)))
                self.unacked.append(int(self._client.hlen("unacked")))
            except Exception:  # noqa: BLE001 - 采样不该拖垮实验
                pass
            time.sleep(self._interval)


def _aggregate(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """把账本压成「每个 Run 被投递几次、领取几次、执行几次」。"""

    deliveries: dict[str, int] = {}
    claims: dict[str, int] = {}
    executions: dict[str, int] = {}
    for line in ledger:
        run_id = str(line.get("run_id") or "")
        if not run_id:
            continue
        kind = line.get("kind")
        if kind == "delivery":
            deliveries[run_id] = deliveries.get(run_id, 0) + 1
        elif kind == "event" and line.get("event_type") == "worker_claimed":
            claims[run_id] = claims.get(run_id, 0) + 1
        elif kind == "execution":
            executions[run_id] = executions.get(run_id, 0) + 1
    return {
        "deliveries_per_run": deliveries,
        "claims_per_run": claims,
        "executions_per_run": executions,
        "duplicate_claims": sorted(
            run_id for run_id, count in claims.items() if count > 1
        ),
        "duplicate_executions": sorted(
            run_id for run_id, count in executions.items() if count > 1
        ),
    }


def _per_worker(ledger: list[dict[str, Any]]) -> dict[str, Any]:
    """逐进程资源与产量。

    CPU 与内存都是进程自己写在账本里的：父进程不去猜另一个平台的进程指标。
    """

    summary: dict[str, dict[str, Any]] = {}
    for line in ledger:
        worker_id = str(line.get("worker_id") or "")
        if not worker_id:
            continue
        entry = summary.setdefault(
            worker_id,
            {
                "pid": line.get("pid"),
                "claims": 0,
                "executions": 0,
                # 进程就绪之前的 CPU 是 Celery/Python 的启动与导入成本，不该摊到
                # 「跑一次 Run 要多少 CPU」上；两个数字都留下，口径写清楚。
                "startup_cpu_seconds": None,
                "cpu_seconds": 0.0,
                "rss_bytes": None,
            },
        )
        if line.get("kind") == "ready":
            entry["startup_cpu_seconds"] = float(line.get("cpu_seconds") or 0.0)
        if line.get("kind") == "event" and line.get("event_type") == "worker_claimed":
            entry["claims"] += 1
        if line.get("kind") == "execution":
            entry["executions"] += 1
        entry["cpu_seconds"] = max(
            float(entry["cpu_seconds"]), float(line.get("cpu_seconds") or 0.0)
        )
        if line.get("rss_bytes") is not None:
            entry["rss_bytes"] = max(
                int(entry["rss_bytes"] or 0), int(line["rss_bytes"])
            )
    for entry in summary.values():
        executions = max(1, int(entry["executions"]))
        startup = float(entry["startup_cpu_seconds"] or 0.0)
        entry["cpu_after_startup_seconds"] = round(
            max(0.0, float(entry["cpu_seconds"]) - startup), 4
        )
        entry["cpu_ms_per_execution_excluding_startup"] = round(
            max(0.0, float(entry["cpu_seconds"]) - startup) * 1000 / executions, 2
        )
        entry["cpu_ms_per_execution"] = round(
            float(entry["cpu_seconds"]) * 1000 / executions, 2
        )
    return summary


def _wait_until(predicate, *, timeout_s: float, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


def _run_compete(
    *,
    store: MySqlAgentRunStore,
    dispatcher: AgentRunDispatcher,
    transport: CeleryAgentRunTransport,
    workers: _Workers,
    client: redis.Redis,
    queue: str,
    burst: int,
) -> dict[str, Any]:
    scenario = "compete"
    sampler = _Sampler(client, queue)
    sampler.start()
    records = []
    dispatch_ms: list[float] = []
    started = time.perf_counter()
    try:
        for index in range(burst):
            task = _task(index, scenario)
            started_ms = time.perf_counter()
            record = dispatcher.dispatch(task)
            dispatch_ms.append((time.perf_counter() - started_ms) * 1000)
            records.append(record)
        # 每条消息再投一次：重复投递在真实系统里随时会发生（重试、重放、运维手抖）。
        for index in range(burst):
            transport.publish(_task(index, scenario))
        drained = _wait_until(
            lambda: all(
                store.get_run(record.run_id).status != "QUEUED"  # type: ignore[union-attr]
                and store.get_run(record.run_id).status  # type: ignore[union-attr]
                not in {"RUNNING", "WAITING_APPROVAL", "WAITING_VALIDATION"}
                for record in records
            ),
            timeout_s=max(60.0, burst * 2.0),
        )
        # 「重复投递不会跑两遍」这句话只有在消息全被消费完之后才站得住：
        # 队列里还压着没投出去的重复消息就停表，等于用没跑完的分母下结论。
        broker_drained = _wait_until(
            lambda: int(client.llen(queue)) == 0 and int(client.hlen("unacked")) == 0,
            timeout_s=60,
        )
    finally:
        makespan = time.perf_counter() - started
        sampler.stop()
    ledger = workers.read_ledger()
    aggregate = _aggregate(ledger)
    statuses = [store.get_run(record.run_id) for record in records]
    created_ms = {record.run_id: record.updated_at_epoch_ms for record in records}
    claimed_at = {
        str(line.get("run_id")): int(line.get("written_at_ms") or 0)
        for line in ledger
        if line.get("kind") == "event" and line.get("event_type") == "worker_claimed"
    }
    queue_waits = [
        float(claimed_at[run_id] - created_ms[run_id])
        for run_id in claimed_at
        if run_id in created_ms and claimed_at[run_id] >= created_ms[run_id]
    ]
    return {
        "scenario": scenario,
        "burst": burst,
        "workers": workers.names,
        "dispatch_ms": {
            "p50": _percentile(dispatch_ms, 0.50),
            "p95": _percentile(dispatch_ms, 0.95),
            "max": round(max(dispatch_ms), 2) if dispatch_ms else None,
        },
        "queue_wait_ms": {
            "samples": len(queue_waits),
            "p50": _percentile(queue_waits, 0.50),
            "p95": _percentile(queue_waits, 0.95),
            "p99": _percentile(queue_waits, 0.99),
        },
        "drained": drained,
        "broker_drained": broker_drained,
        "makespan_seconds": round(makespan, 3),
        "throughput_per_second": round(burst / makespan, 2) if makespan else None,
        "statuses": {
            status: sum(1 for item in statuses if item is not None and item.status == status)
            for status in sorted({item.status for item in statuses if item is not None})
        },
        "duplicate_claims": aggregate["duplicate_claims"],
        "duplicate_executions": aggregate["duplicate_executions"],
        "deliveries": sum(aggregate["deliveries_per_run"].values()),
        "claims": sum(aggregate["claims_per_run"].values()),
        "executions": sum(aggregate["executions_per_run"].values()),
        "per_worker": _per_worker(ledger),
        "queue_depth_max": max(sampler.queue_depth) if sampler.queue_depth else None,
        "unacked_max": max(sampler.unacked) if sampler.unacked else None,
        "unacked_p50": _percentile([float(item) for item in sampler.unacked], 0.50),
    }


def _run_kill(
    *,
    store: MySqlAgentRunStore,
    dispatcher: AgentRunDispatcher,
    transport: CeleryAgentRunTransport,
    workers: _Workers,
    burst: int,
    lease_ms: int,
) -> dict[str, Any]:
    scenario = "kill"
    tasks = [_task(index, scenario) for index in range(burst)]
    for task in tasks:
        dispatcher.dispatch(task)
    # 等到至少一个 Run 真的被某个进程领走，才知道该杀谁。
    claimed = _wait_until(
        lambda: any(
            line.get("kind") == "event" and line.get("event_type") == "worker_claimed"
            for line in workers.read_ledger()
        ),
        timeout_s=60,
    )
    if not claimed:
        raise SystemExit("60 秒内没有任何进程领到 Run，实验无法继续。")
    ledger_before = workers.read_ledger()
    victim = str(
        next(
            line["worker_id"]
            for line in ledger_before
            if line.get("kind") == "event" and line.get("event_type") == "worker_claimed"
        )
    )
    # 被杀那一刻它**仍然持有**的 Run：它声称过的那些里，Run 还停在 RUNNING 的。
    # 已经跑完的不用接管（接管测试里不该混进正常完成的任务），被别的进程领走的
    # 也不属于它。solo pool 一次只跑一条，所以这里通常只有一条——但结论要靠
    # 判定写对，而不是靠「通常」。
    victim_claims = [
        str(line.get("run_id"))
        for line in ledger_before
        if line.get("kind") == "event"
        and line.get("event_type") == "worker_claimed"
        and line.get("worker_id") == victim
    ]
    held = sorted(
        run_id
        for run_id in victim_claims
        if (record := store.get_run(run_id)) is not None and record.status == "RUNNING"
    )
    if not held:
        raise SystemExit(
            f"被杀进程 {victim} 在死亡瞬间没有持有任何 RUNNING 的 Run，"
            "这一次 kill 没有测到接管路径。"
        )
    killed_at_ms = workers.kill(victim)
    # 第一次重投：租约还在有效期内。跑不动是期望行为——重复投递不该重放正在跑的 Run。
    # 只统计「被杀进程持有过的那些 Run」在 kill 之后有没有被执行：别的进程正在跑的
    # 其它 Run 也会写 execution 事件，一起算进来就会把结论弄脏。
    for task in tasks:
        transport.publish(task)
    time.sleep(1.0)
    ledger_after_first = workers.read_ledger()
    executions_inside_lease = sum(
        1
        for line in ledger_after_first
        if line.get("kind") == "execution"
        and str(line.get("run_id")) in set(held)
        and line.get("written_at_ms", 0) >= killed_at_ms
    )
    # 第二次重投：等租约过期（按 Worker 自己的租约长度算），再看接管。
    lease_expired_at_ms = max(
        int(line.get("written_at_ms") or 0)
        for line in ledger_before
        if line.get("kind") == "event" and line.get("event_type") == "worker_claimed"
    ) + lease_ms
    remaining_ms = lease_expired_at_ms - time.time() * 1000
    if remaining_ms > 0:
        time.sleep(remaining_ms / 1000)
    expired_at = time.time() * 1000
    for task in tasks:
        transport.publish(task)
    took_over = _wait_until(
        lambda: all(
            (record := store.get_run(item.run_id)) is not None
            and record.status == COMPLETED
            for item in tasks
        ),
        timeout_s=max(60.0, burst * 5.0),
    )
    ledger = workers.read_ledger()
    aggregate = _aggregate(ledger)
    takeover_at_ms = min(
        [
            int(line.get("written_at_ms") or 0)
            for line in ledger
            if line.get("kind") == "execution" and int(line.get("written_at_ms") or 0) > expired_at
        ]
        or [0]
    )
    statuses = [store.get_run(item.run_id) for item in tasks]
    return {
        "scenario": scenario,
        "burst": burst,
        "lease_ms": lease_ms,
        "workers": workers.names,
        "killed_worker": victim,
        "victim_held_runs": held,
        "executions_inside_valid_lease": executions_inside_lease,
        "takeover": took_over,
        "takeover_latency_ms": (
            round(takeover_at_ms - expired_at, 2) if takeover_at_ms else None
        ),
        "statuses": {
            status: sum(1 for item in statuses if item is not None and item.status == status)
            for status in sorted({item.status for item in statuses if item is not None})
        },
        "duplicate_claims": aggregate["duplicate_claims"],
        # 这个场景里「被领两次」是预期行为，不是缺陷：被杀进程领过一次，租约过期后
        # 接管进程合法地再领一次。判据是下面那个「执行了几次」——它必须等于 Run 数。
        "duplicate_claims_note": (
            "kill 场景：被接管的那一条会被合法地领两次（死亡进程 + 接管进程）；"
            "是否重复执行看 duplicate_executions。"
        ),
        "duplicate_executions": aggregate["duplicate_executions"],
        "deliveries": sum(aggregate["deliveries_per_run"].values()),
        "claims": sum(aggregate["claims_per_run"].values()),
        "executions": sum(aggregate["executions_per_run"].values()),
        "per_worker": _per_worker(ledger),
    }


def _write_result(outcome: dict[str, Any], *, label: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = OUTPUT_ROOT / f"multiprocess-{outcome['scenario']}-{label}-{stamp}.json"
    path.write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q-011 多进程 Worker 竞争实验")
    parser.add_argument("--scenario", choices=["compete", "kill"], default="compete")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--burst", type=int, default=120)
    parser.add_argument("--occupancy-ms", type=int, default=25)
    parser.add_argument("--lease-ms", type=int, default=30_000)
    parser.add_argument("--broker", default=DEFAULT_BROKER)
    parser.add_argument("--queue", default=DEFAULT_QUEUE)
    parser.add_argument("--label", default="local")
    arguments = parser.parse_args(argv)

    config = _mysql_config()
    engine = create_db_engine(config)
    create_all_tables(engine)
    truncate_all_tables(engine)
    store = MySqlAgentRunStore(create_session_factory(engine))
    app = _eval_app(arguments.broker, arguments.queue)
    client = _redis_client(arguments.broker)
    try:
        client.ping()
    except Exception as error:  # noqa: BLE001 - 环境没起来就直接说清楚
        raise SystemExit(f"连不上 broker {arguments.broker}：{error}") from error
    _purge(app, client, arguments.queue)

    transport = CeleryAgentRunTransport(app)
    dispatcher = AgentRunDispatcher(store=store, transport=transport)
    run_dir = OUTPUT_ROOT / f"multiprocess-{arguments.scenario}-{arguments.label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base_env = dict(os.environ)
    # 子进程按测试库连接：实验会清空整库，不能让它落在应用库上。
    base_env["CODEINSIGHT_MYSQL_DATABASE"] = config.database
    if arguments.scenario == "kill":
        # 杀进程场景要能在几十秒内看到租约过期。
        arguments.lease_ms = min(arguments.lease_ms, 5_000)
        arguments.occupancy_ms = max(arguments.occupancy_ms, 4_000)
    print(
        f"[q011-mp] {arguments.scenario}: workers={arguments.workers} burst={arguments.burst} "
        f"occupancy={arguments.occupancy_ms}ms lease={arguments.lease_ms}ms "
        f"broker={arguments.broker} queue={arguments.queue}"
    )
    workers = _Workers(
        count=arguments.workers,
        run_dir=run_dir,
        base_env=base_env,
        queue=arguments.queue,
        broker=arguments.broker,
        occupancy_ms=arguments.occupancy_ms,
        lease_ms=arguments.lease_ms,
    )
    try:
        ready = workers.wait_ready()
        print(f"[q011-mp] 就绪进程：{ready}")
        if arguments.scenario == "kill":
            outcome = _run_kill(
                store=store,
                dispatcher=dispatcher,
                transport=transport,
                workers=workers,
                burst=arguments.burst,
                lease_ms=arguments.lease_ms,
            )
        else:
            outcome = _run_compete(
                store=store,
                dispatcher=dispatcher,
                transport=transport,
                workers=workers,
                client=client,
                queue=arguments.queue,
                burst=arguments.burst,
            )
    finally:
        workers.stop()
        app.control.purge()
    outcome["label"] = arguments.label
    outcome["occupancy_ms"] = arguments.occupancy_ms
    outcome["broker"] = arguments.broker
    outcome["queue"] = arguments.queue
    outcome["database"] = config.database
    outcome["python"] = sys.version.split()[0]
    outcome["generated_at"] = datetime.now(UTC).isoformat()
    outcome["ledger_dir"] = str(run_dir)
    outcome["incomplete"] = sorted(
        run_id
        for run_id, count in _aggregate(workers.read_ledger())["claims_per_run"].items()
        if count == 0
    )
    path = _write_result(outcome, label=arguments.label)
    print(f"[q011-mp] 结果 -> {path}")
    print(json.dumps(outcome, ensure_ascii=False, indent=2))
    engine.dispose()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
