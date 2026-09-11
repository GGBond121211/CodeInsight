"""Q-011 Task 8：实时事件广播的单元测试。

这里不碰 HTTP，也不碰 Store：验证的是「订阅时先补历史再收新事件」「按 run 隔离」
「慢连接丢最旧的而不是无限积压」这几条广播语义。
"""

from __future__ import annotations

import pytest

from codeinsight.domain.trace import RunEvent
from codeinsight.infrastructure.live_event_broker import LiveEventBroker


def _event(sequence: int, *, run_id: str = "run-1") -> RunEvent:
    return RunEvent(
        event_id=f"{run_id}:{sequence}",
        run_id=run_id,
        sequence=sequence,
        event_type="tool_result_committed",
        occurred_at_epoch_ms=1_700_000_000_000 + sequence,
        payload={"tool_name": "search_repository"},
    )


def test_subscribe_replays_history_then_delivers_live_events() -> None:
    broker = LiveEventBroker()
    stored = (_event(1), _event(2))

    subscription = broker.subscribe("run-1", after_sequence=0, replay=lambda: stored)
    broker.publish(_event(3))

    assert [event.sequence for event in subscription.backlog] == [1, 2]
    kind, payload = subscription.subscriber.get_nowait()
    assert kind == "event"
    assert payload.sequence == 3


def test_replay_is_a_snapshot_of_the_position_that_was_asked_for() -> None:
    """重连带 after_sequence：回放只能是它还没看过的那一段。"""

    broker = LiveEventBroker()
    asked: list[int] = []

    def replay() -> tuple[RunEvent, ...]:
        asked.append(1)
        return ()

    subscription = broker.subscribe("run-1", after_sequence=2, replay=replay)

    assert asked == [1]
    assert subscription.backlog == ()
    assert subscription.debug_backlog == ()


def test_subscribers_are_scoped_to_one_run() -> None:
    broker = LiveEventBroker()
    first = broker.subscribe("run-1")
    second = broker.subscribe("run-2")

    broker.publish(_event(1, run_id="run-1"))

    assert first.subscriber.get_nowait()[1].sequence == 1
    assert second.subscriber.empty()
    assert broker.subscriber_count("run-1") == 1
    assert broker.subscriber_count("run-2") == 1


def test_unsubscribe_stops_delivery_and_forgets_the_connection() -> None:
    broker = LiveEventBroker()
    subscription = broker.subscribe("run-1")

    broker.unsubscribe("run-1", subscription.subscriber)
    broker.publish(_event(1))

    assert subscription.subscriber.empty()
    assert broker.subscriber_count("run-1") == 0


def test_a_slow_connection_loses_the_oldest_push_not_the_newest() -> None:
    """推送丢了可以从 Store 补齐，内存涨死不能。"""

    broker = LiveEventBroker(queue_size=1)
    subscription = broker.subscribe("run-1")

    broker.publish(_event(1))
    broker.publish(_event(2))

    kind, payload = subscription.subscriber.get_nowait()
    assert kind == "event"
    assert payload.sequence == 2
    assert subscription.subscriber.empty()


def test_debug_reasoning_is_replayed_once_per_connection() -> None:
    broker = LiveEventBroker()
    broker.publish_debug_reasoning("run-1", "先看 readme", model="fake")

    first = broker.subscribe("run-1", after_sequence=0)
    resumed = broker.subscribe("run-1", after_sequence=3)

    assert [item["content"] for item in first.debug_backlog] == ["先看 readme"]
    # 续订不回放：审批恢复时再放一遍上一阶段的思考，只会让人以为模型又跑了一次。
    assert resumed.debug_backlog == ()


def test_blank_debug_reasoning_is_not_published() -> None:
    broker = LiveEventBroker()
    broker.publish_debug_reasoning("run-1", "   ", model="fake")

    subscription = broker.subscribe("run-1", after_sequence=0)

    assert subscription.debug_backlog == ()
    assert subscription.subscriber.empty()


def test_negative_after_sequence_is_rejected() -> None:
    with pytest.raises(ValueError):
        LiveEventBroker().subscribe("run-1", after_sequence=-1)


def test_queue_size_must_be_positive() -> None:
    with pytest.raises(ValueError):
        LiveEventBroker(queue_size=0)
