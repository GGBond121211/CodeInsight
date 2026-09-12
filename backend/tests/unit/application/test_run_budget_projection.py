"""Q-012 U6：把花费闸结论从事件读成 Run 行上的摘要。

这一层只回答一个问题：读到的是什么数。它决定 Run 行上的 token_budget /
tokens_used 是「闸值 + 受闸用量」，而不是把 Router 调用、缓存命中或重试
也算进去的另一个数字。
"""

from __future__ import annotations

from codeinsight.application.conversation_service import _budget_projection
from codeinsight.domain.trace import BUDGET_LIMIT, BUDGET_USED, MODEL_RESULT, RunEvent


def _event(sequence: int, event_type: str, payload: dict[str, str] | None = None) -> RunEvent:
    return RunEvent(
        event_id=f"ev-{sequence}",
        run_id="run-1",
        sequence=sequence,
        event_type=event_type,
        occurred_at_epoch_ms=1_000 + sequence,
        payload=payload or {},
    )


def test_a_run_without_a_gate_reports_nothing_rather_than_zero() -> None:
    """没有 budget_limit 就是没受过闸：返回 None，让调用方什么都不写。"""

    events = [_event(1, MODEL_RESULT, {"outcome": "success", "input_tokens": "900"})]
    assert _budget_projection(events) == (None, 0)


def test_usage_is_the_last_cumulative_report_not_the_sum_of_all_of_them() -> None:
    """budget_used 是累计值：把多条相加会把同一段 token 数很多遍。"""

    events = [
        _event(1, BUDGET_LIMIT, {"budget_limit": "300000", "soft_ratio": "0.7000"}),
        _event(2, BUDGET_USED, {"budget_limit": "300000", "budget_used": "1200"}),
        _event(3, BUDGET_USED, {"budget_limit": "300000", "budget_used": "9800"}),
    ]
    assert _budget_projection(events) == (300_000, 9_800)


def test_malformed_payloads_do_not_turn_into_numbers() -> None:
    events = [
        _event(1, BUDGET_LIMIT, {"budget_limit": "不是数字"}),
        _event(2, BUDGET_USED, {"budget_used": ""}),
    ]
    assert _budget_projection(events) == (None, 0)
