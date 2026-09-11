from __future__ import annotations

import pytest

from codeinsight.application.context_budget import (
    CONTEXT_LIFECYCLE_POLICY_VERSION,
    ContextLifecyclePolicy,
    scale_tokens,
)


def test_policy_locks_the_registered_working_numbers() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.context_window_tokens == 128_000
    assert policy.compaction_threshold_ratio == 0.95
    assert policy.retain_recent_ratio == 0.16
    assert policy.summary_max_output_tokens == 8_192
    assert policy.compaction_retries == 1
    assert policy.overflow_retries == 1
    assert policy.version == CONTEXT_LIFECYCLE_POLICY_VERSION


def test_ratio_math_lands_exactly_on_the_registered_boundary() -> None:
    # 浮点乘法在这里会飘；定点换算必须精确落在 121600 和 20480。
    assert scale_tokens(128_000, 0.95) == 121_600
    assert scale_tokens(128_000, 0.16) == 20_480


def test_just_below_the_threshold_does_not_trigger_compaction() -> None:
    policy = ContextLifecyclePolicy()
    below = scale_tokens(policy.context_window_tokens, 0.9499)
    assert below == 121_587
    assert policy.should_compact(below) is False


def test_the_threshold_itself_triggers_compaction() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.compaction_trigger_tokens == 121_600
    assert policy.should_compact(121_599) is False
    assert policy.should_compact(121_600) is True
    assert policy.should_compact(policy.context_window_tokens) is True


def test_recent_retention_budget_is_sixteen_percent_of_the_window() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.retained_recent_tokens == 20_480
    assert policy.retained_recent_tokens < policy.compaction_trigger_tokens


def test_summary_budget_is_fixed_and_not_the_answer_output_cap() -> None:
    # 摘要预算跟着策略走，不跟着窗口大小，也不跟着某条路线的单次输出上限。
    narrow = ContextLifecyclePolicy(context_window_tokens=20_000)
    assert narrow.summary_max_output_tokens == 8_192
    assert ContextLifecyclePolicy().summary_max_output_tokens == 8_192


def test_overflow_guard_covers_input_plus_reserved_output() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.exceeds_window(100_000, 40_960) is True
    assert policy.exceeds_window(87_040, 40_960) is False
    # 恰好等于窗口不算溢出，超出 1 token 就算。
    assert policy.exceeds_window(87_041, 40_960) is True
    assert policy.exceeds_window(121_600, 6_400) is False


def test_output_reservation_cannot_fill_the_whole_window() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.input_allowance(40_960) == 87_040
    with pytest.raises(ValueError):
        policy.input_allowance(128_000)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("context_window_tokens", 0),
        ("context_window_tokens", -1),
        ("compaction_threshold_ratio", 0.0),
        ("compaction_threshold_ratio", 1.5),
        ("retain_recent_ratio", 0.0),
        ("retain_recent_ratio", 1.0),
        ("summary_max_output_tokens", 0),
        ("compaction_retries", -1),
        ("overflow_retries", -1),
        ("version", "   "),
    ),
)
def test_invalid_policy_field_is_refused(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        ContextLifecyclePolicy(**{field: value})


@pytest.mark.parametrize("value", (-1,))
def test_negative_usage_is_refused(value: int) -> None:
    policy = ContextLifecyclePolicy()
    with pytest.raises(ValueError):
        policy.should_compact(value)
    with pytest.raises(ValueError):
        policy.exceeds_window(value, 0)
    with pytest.raises(ValueError):
        policy.input_allowance(value)
