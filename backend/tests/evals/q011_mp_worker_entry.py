"""Q-011 收尾实验用的 Worker 进程入口：真 Celery Worker，假模型。

多进程实验要测的是「两个独立进程同时抢同一批 Run」这条边，这条边不需要模型参与。
所以这里另起一个 Celery app（队列名与 broker 库都由环境变量指定，默认只碰本机实验用
的 6380/5），任务名与真实 Worker 相同（codeinsight.run_agent），但任务体里的执行器是
桩：不调模型、不碰仓库，只按固定占用时间结束。

为什么不用 codeinsight.agent.worker_tasks：那边装配的是真实服务（真模型、真工具、
真校验），跑一轮要花钱且不可复现。真实链路的端到端验收（含真模型）另有一份证据，
两条证据回答的是不同问题，不要互相替代。

每一个进程都往自己的账本文件追加 JSONL：投递到达、领取、执行。父进程（实验脚本）
只读这些账本，所以「跑了几次、谁跑的、多久」是进程自己写下的，不是父进程猜的。
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from celery import Celery
from celery.signals import worker_ready

from codeinsight.agent.agent_run_worker import AgentRunWorker
from codeinsight.application.agent_run_executor import (
    AgentRunContext,
    AgentRunExecutor,
)
from codeinsight.domain.agent_run import AgentRunRecord, AgentRunTask
from codeinsight.domain.chat import CHAT_COMPLETED
from codeinsight.infrastructure.celery_agent_dispatcher import AGENT_RUN_TASK_NAME
from codeinsight.infrastructure.chat_runtime import ChatExecution
from codeinsight.infrastructure.db.engine import (
    MySqlConfig,
    create_db_engine,
    create_session_factory,
)
from codeinsight.infrastructure.db.stores import MySqlAgentRunStore

BROKER = os.environ.get("CODEINSIGHT_Q011_MP_BROKER", "redis://127.0.0.1:6380/5")
QUEUE = os.environ.get("CODEINSIGHT_Q011_MP_QUEUE", "q011-mp-eval")
WORKER_ID = os.environ.get("CODEINSIGHT_Q011_MP_WORKER_ID", "")
LEDGER = os.environ.get("CODEINSIGHT_Q011_MP_LEDGER", "")
OCCUPANCY_MS = int(os.environ.get("CODEINSIGHT_Q011_MP_OCCUPANCY_MS", "25"))
LEASE_MS = int(os.environ.get("CODEINSIGHT_Q011_MP_LEASE_MS", "60000"))

celery_app = Celery("codeinsight-q011-mp", broker=BROKER, backend=None)
celery_app.conf.update(
    task_default_queue=QUEUE,
    # 与真实 Worker 同一套边界：prefetch=1、acks_late、丢进程要重新投递。
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)


def _rss_bytes() -> int | None:
    """当前进程的常驻内存。非 Windows 返回 None——取不到就不猜。"""

    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        handle = kernel32.GetCurrentProcess()
        # 句柄是指针宽度：不声明 argtypes 的话 ctypes 会按 32 位 int 传，
        # 64 位进程里句柄被截断，调用只会失败并返回 0（实测拿到过全 null）。
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_Counters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    except Exception:  # noqa: BLE001 - 取不到资源数字不该让实验失败
        return None


def _ledger(**fields: object) -> None:
    """追加一条账本。一个进程一个文件，因此不需要跨进程加锁。"""

    if not LEDGER:
        return
    line = dict(fields)
    line.setdefault("worker_id", _worker_id())
    line.setdefault("pid", os.getpid())
    # 每个进程自己报资源：父进程不跟着平台差异去采样别人的进程。
    line.setdefault("cpu_seconds", round(time.process_time(), 4))
    line.setdefault("rss_bytes", _rss_bytes())
    line.setdefault("written_at_ms", int(time.time() * 1000))
    with Path(LEDGER).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def _worker_id() -> str:
    return WORKER_ID or f"q011-mp-worker-{os.getpid()}"


class _StubLoader:
    """上下文重建的桩。

    真实验收里这一段是 AgentRunContextLoader（真读 Session 事实），本实验不重复
    验证它——这里只要让领取路径能走完，并把 attempt 如实带出来。
    """

    def load(self, task: AgentRunTask, record: AgentRunRecord) -> AgentRunContext:
        return AgentRunContext(
            run_id=task.run_id,
            turn_id=task.turn_id,
            session_id=task.session_id,
            task_id=task.task_id,
            repo_id="q011-mp-eval",
            repo_root="",
            repo_fingerprint="q011-mp-eval",
            user_message="q011 多进程实验探针",
            task_type="explain",
            confidence=1.0,
            rule="q011-mp-eval",
            options=record.options,
            attempt=record.attempt,
        )


def _runner(context: AgentRunContext) -> ChatExecution:
    """假执行：占用固定时间，然后报告完成。不调模型，也不碰仓库。"""

    started = time.perf_counter()
    time.sleep(OCCUPANCY_MS / 1000)
    _ledger(
        kind="execution",
        run_id=context.run_id,
        attempt=context.attempt,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return ChatExecution(
        status=CHAT_COMPLETED,
        assistant_message="q011 多进程实验探针",
        result={"kind": "q011_multiprocess_probe"},
    )


_worker: AgentRunWorker | None = None


def _service() -> AgentRunWorker:
    """本进程唯一的 Worker：引擎与 Store 建一次，重复领取不再付连接代价。"""

    global _worker
    if _worker is not None:
        return _worker
    config = MySqlConfig.from_env()
    if config is None:
        raise RuntimeError("未配置 CODEINSIGHT_MYSQL_*：多进程实验要真事实层。")
    engine = create_db_engine(config)
    _worker = AgentRunWorker(
        store=MySqlAgentRunStore(create_session_factory(engine)),
        loader=_StubLoader(),
        executor=AgentRunExecutor(emit=_emit),
        emit=_emit,
        worker_id=_worker_id(),
        lease_ms=LEASE_MS,
    )
    return _worker


def _emit(run_id: str, event_type: str, payload: dict[str, str]) -> None:
    _ledger(kind="event", event_type=event_type, run_id=run_id, payload=payload)


@worker_ready.connect
def _on_ready(**_kwargs: object) -> None:
    """账本里写一条就绪：父进程靠它判断「进程真的在监听队列了」。"""

    _ledger(kind="ready")


@celery_app.task(name=AGENT_RUN_TASK_NAME, acks_late=True)
def run_agent(**payload: object) -> dict[str, object]:
    """与真实任务同名同载荷，只把执行体换成桩。"""

    task = AgentRunTask.from_payload(payload)
    _ledger(kind="delivery", run_id=task.run_id, task_id=task.task_id)
    outcome = _service().handle(task, runner=_runner)
    return {
        "run_id": outcome.run_id,
        "claimed": outcome.claimed,
        "status": outcome.status,
        "error_class": outcome.error_class,
    }
