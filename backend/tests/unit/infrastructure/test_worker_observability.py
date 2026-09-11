"""Q-011 Task 10：Worker 运行指标必须是低基数的。

它守的是一条边界：Prometheus label 只放 task_kind / status 这类低基数维度，
run_id、turn_id、session_id、task_id、request_id、attempt_id 只能进事件和 Span
attribute。做错了不会报错，只会让时间序列随请求量线性增长——所以要有测试。
"""

from __future__ import annotations

import pytest

from codeinsight.domain.trace import MetricLabelCardinalityError, assert_metric_labels
from codeinsight.infrastructure.otel import Telemetry

FORBIDDEN_IN_LABELS = (
    "run_id",
    "turn_id",
    "session_id",
    "task_id",
    "request_id",
    "attempt_id",
)


def test_one_worker_attempt_is_visible_as_low_cardinality_metrics() -> None:
    telemetry = Telemetry()

    telemetry.record_queue_wait(task_kind="agent_run", wait_ms=250)
    telemetry.record_agent_run(
        task_kind="agent_run",
        status="COMPLETED",
        execution_ms=1200,
        model_calls=3,
        tool_calls=2,
        validation_wait_ms=400,
    )
    telemetry.record_worker_recovery(task_kind="resume_after_approval")
    rendered = telemetry.metrics().decode("utf-8")

    assert "codeinsight_agent_run_attempts_total" in rendered
    assert "codeinsight_agent_run_duration_seconds_count" in rendered
    assert "codeinsight_agent_run_queue_wait_seconds_count" in rendered
    assert "codeinsight_agent_run_validation_wait_seconds_count" in rendered
    assert (
        'codeinsight_agent_run_model_calls_total{task_kind="agent_run"} 3.0' in rendered
    )
    assert 'codeinsight_agent_run_tool_calls_total{task_kind="agent_run"} 2.0' in rendered
    assert "codeinsight_worker_recoveries_total" in rendered


def test_identifiers_never_become_metric_labels() -> None:
    telemetry = Telemetry()
    telemetry.record_queue_wait(task_kind="agent_run", wait_ms=10)
    telemetry.record_agent_run(
        task_kind="resume_after_validation", status="REVIEW_REQUIRED", execution_ms=5
    )

    rendered = telemetry.metrics().decode("utf-8")

    for identifier in FORBIDDEN_IN_LABELS:
        assert identifier not in rendered
    assert 'task_kind="agent_run"' in rendered
    assert 'status="REVIEW_REQUIRED"' in rendered


def test_metric_label_guard_rejects_identifiers() -> None:
    with pytest.raises(MetricLabelCardinalityError):
        assert_metric_labels(["task_kind", "run_id"])


def test_impossible_values_are_dropped_not_invented() -> None:
    telemetry = Telemetry()

    # 负的排队时间不是「很快」，是算错了：不记，也不报一个假的观测值。
    telemetry.record_queue_wait(task_kind="agent_run", wait_ms=-1)
    with pytest.raises(ValueError):
        telemetry.record_agent_run(task_kind="agent_run", status="FAILED", execution_ms=-1)

    assert "codeinsight_agent_run_queue_wait_seconds_count" not in telemetry.metrics().decode(
        "utf-8"
    )
