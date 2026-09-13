"""事件序号在并发写入下必须仍然连续：父进程与 Worker 同时写同一个 run 的场景。

为什么单独测这一层：`emit` 是「读下一个序号 + 追加」两步。2026-09-13 实测两次撞出
`EventSequenceError`（HTTP 500、Run 落成 MANUAL_REQUIRED）。这里既锁「冲突会重试」，
也锁「真的连续冲突时仍然显式失败」——不能把两条纪律合成一条。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from codeinsight.domain.trace import RUN_STARTED, RunEvent
from codeinsight.infrastructure.chat_runtime import EMIT_SEQUENCE_ATTEMPTS, ChatRuntime
from codeinsight.infrastructure.event_log import (
    EventSequenceError,
    InMemoryEventLog,
    LiveEventLog,
)


class _StealOnceLog:
    """第一次追加时假装「另一条线程抢先占了序号」，之后照常接受。"""

    def __init__(self) -> None:
        self._events: list[RunEvent] = []
        self.steals = 0

    def next_sequence(self, run_id: str) -> int:
        return len(self._events) + 1

    def append(self, event: RunEvent) -> None:
        if self.steals == 0:
            self.steals = 1
            # 模拟「别人已经写了一条」：内存里先占掉这个序号。
            self._events.append(
                RunEvent(
                    event_id=f"{event.run_id}:{event.sequence}",
                    run_id=event.run_id,
                    sequence=event.sequence,
                    event_type=RUN_STARTED,
                    occurred_at_epoch_ms=event.occurred_at_epoch_ms,
                    payload={},
                )
            )
            raise EventSequenceError("序号已被占用")
        if event.sequence != len(self._events) + 1:
            raise EventSequenceError("序号不连续")
        self._events.append(event)

    def read_events(self, run_id: str, *, after_sequence: int = 0):
        return tuple(
            event for event in self._events if event.sequence > after_sequence
        )


class _AlwaysConflictingLog(_StealOnceLog):
    """永远冲突：重试预算必须用完然后抛出，不能装作写成功了。"""

    def append(self, event: RunEvent) -> None:
        raise EventSequenceError("序号已被占用")


def test_emit_retries_when_another_writer_took_the_sequence() -> None:
    store = _StealOnceLog()
    runtime = ChatRuntime(event_log=LiveEventLog(store))  # type: ignore[arg-type]

    event = runtime.emit("run-1", RUN_STARTED, {"status": "started"})

    assert event.sequence == 2
    assert store.steals == 1
    assert len(store.read_events("run-1")) == 2


def test_emit_still_fails_after_the_retry_budget() -> None:
    runtime = ChatRuntime(event_log=LiveEventLog(_AlwaysConflictingLog()))  # type: ignore[arg-type]

    with pytest.raises(EventSequenceError):
        runtime.emit("run-1", RUN_STARTED, {"status": "started"})

    # 预算是有界的：不会无限重试。
    assert EMIT_SEQUENCE_ATTEMPTS > 1


def test_concurrent_emits_stay_contiguous() -> None:
    """两条线程各写 40 条：序号必须正好是 1..80，没有洞也没有重复。"""

    store = InMemoryEventLog()
    runtime = ChatRuntime(event_log=LiveEventLog(store))
    per_thread = 40

    def writer(prefix: str) -> None:
        for index in range(per_thread):
            runtime.emit("run-1", RUN_STARTED, {"writer": f"{prefix}-{index}"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(writer, name) for name in ("a", "b")]
        for future in futures:
            future.result()

    events = store.read_events("run-1")

    assert len(events) == per_thread * 2
    assert [event.sequence for event in events] == list(range(1, per_thread * 2 + 1))

