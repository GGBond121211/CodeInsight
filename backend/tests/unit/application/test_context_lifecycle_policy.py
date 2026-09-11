from __future__ import annotations

import pytest

from codeinsight.application.context_budget import (
    CONTEXT_LIFECYCLE_POLICY_VERSION,
    ContextLifecyclePolicy,
    scale_tokens,
)


def test_policy_locks_the_registered_working_numbers() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.context_window_tokens == 1_000_000
    assert policy.compaction_threshold_ratio == 0.95
    assert policy.retain_recent_ratio == 0.16
    assert policy.summary_max_tokens == 8_192
    assert policy.non_history_reserve_tokens == 4_096
    assert policy.compaction_retries == 1
    assert policy.overflow_retries == 1
    assert policy.version == CONTEXT_LIFECYCLE_POLICY_VERSION


def test_ratio_math_lands_exactly_on_the_registered_boundary() -> None:
    # 十进制定点而不是浮点乘法：换算到精确值 911088 与 160000。
    assert scale_tokens(959_040, 0.95) == 911_088
    assert scale_tokens(1_000_000, 0.16) == 160_000


def test_just_below_the_threshold_does_not_trigger_compaction() -> None:
    policy = ContextLifecyclePolicy()
    below = scale_tokens(policy.input_allowance(40_960), 0.9499)
    assert below == 910_992
    assert policy.should_compact(below, reserved_output_tokens=40_960) is False


def test_the_threshold_itself_triggers_compaction() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.compaction_trigger_tokens(40_960) == 911_088
    assert policy.should_compact(911_087, reserved_output_tokens=40_960) is False
    assert policy.should_compact(911_088, reserved_output_tokens=40_960) is True


def test_threshold_is_a_fraction_of_the_usable_input_budget() -> None:
    """95% 的分母是可用输入预算，不是原始窗口。

    Q-010 第一版用原始窗口当分母：在 128K 窗口、40960 输出预留下算出 121600，
    比拒收线 87040 还高，压缩永远不可能被触发。1M 窗口下这个错误不会显形
    （950000 < 959040），所以两种窗口都要钉：历史失败模式必须仍然可复现，
    当前默认的触发点必须落在拒收线以内。
    """
    policy = ContextLifecyclePolicy()
    allowance = policy.input_allowance(40_960)
    assert allowance == 959_040
    assert scale_tokens(policy.context_window_tokens, 0.95) == 950_000
    assert scale_tokens(policy.context_window_tokens, 0.95) < allowance
    assert policy.compaction_trigger_tokens(40_960) < allowance
    # 历史失败模式：128K 窗口下用原始窗口当分母会算出 121600，比拒收线还高。
    narrow = ContextLifecyclePolicy(context_window_tokens=128_000)
    assert scale_tokens(narrow.context_window_tokens, 0.95) == 121_600
    assert scale_tokens(narrow.context_window_tokens, 0.95) > narrow.input_allowance(40_960)
    assert narrow.compaction_trigger_tokens(40_960) == 82_688
    assert narrow.compaction_trigger_tokens(40_960) < narrow.input_allowance(40_960)


def test_trigger_reachability_invariant_holds_across_legal_output_caps() -> None:
    """合法的输出上限区间内，触发点都必须可达且压缩必须有收益。"""
    policy = ContextLifecyclePolicy()
    for reserved in (256, 4_096, 40_960, 100_000):
        policy.verify_trigger_reachable(reserved)
        assert policy.compaction_trigger_tokens(reserved) < policy.input_allowance(reserved)
        assert policy.session_history_budget(reserved) > policy.retained_recent_tokens


def test_unreachable_trigger_is_refused() -> None:
    """阈值取 1.0 时触发点等于拒收线，必须被拒绝而不是静默接受。"""
    policy = ContextLifecyclePolicy(compaction_threshold_ratio=1.0)
    with pytest.raises(ValueError):
        policy.verify_trigger_reachable(40_960)


def test_session_history_budget_reserves_the_non_history_overhead() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.session_history_budget(40_960) == 911_088 - 4_096


def test_history_budget_must_exceed_the_retained_window() -> None:
    """输出预留大到把历史预算压到保留窗口以下时，必须报错而不是空转压缩。"""
    policy = ContextLifecyclePolicy(context_window_tokens=20_000)
    with pytest.raises(ValueError):
        policy.session_history_budget(15_000)


def test_recent_retention_budget_is_sixteen_percent_of_the_window() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.retained_recent_tokens == 160_000
    assert policy.retained_recent_tokens < policy.session_history_budget(40_960)


def test_summary_budget_is_fixed_and_not_the_answer_output_cap() -> None:
    # 摘要预算跟着策略走，不跟着窗口大小，也不跟着某条路线的单次输出上限。
    narrow = ContextLifecyclePolicy(context_window_tokens=20_000)
    assert narrow.summary_max_tokens == 8_192
    assert ContextLifecyclePolicy().summary_max_tokens == 8_192


def test_overflow_guard_covers_input_plus_reserved_output() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.exceeds_window(1_000_000, 40_960) is True
    assert policy.exceeds_window(959_040, 40_960) is False
    # 恰好等于窗口不算溢出，超出 1 token 就算。
    assert policy.exceeds_window(959_041, 40_960) is True
    assert policy.exceeds_window(993_600, 6_400) is False


def test_output_reservation_cannot_fill_the_whole_window() -> None:
    policy = ContextLifecyclePolicy()
    assert policy.input_allowance(40_960) == 959_040
    with pytest.raises(ValueError):
        policy.input_allowance(1_000_000)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("context_window_tokens", 0),
        ("context_window_tokens", -1),
        ("compaction_threshold_ratio", 0.0),
        ("compaction_threshold_ratio", 1.5),
        ("retain_recent_ratio", 0.0),
        ("retain_recent_ratio", 1.0),
        ("summary_max_tokens", 0),
        ("non_history_reserve_tokens", -1),
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
        policy.should_compact(value, reserved_output_tokens=40_960)
    with pytest.raises(ValueError):
        policy.exceeds_window(value, 0)
    with pytest.raises(ValueError):
        policy.input_allowance(value)
