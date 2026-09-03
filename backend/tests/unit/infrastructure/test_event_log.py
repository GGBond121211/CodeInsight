"""事件日志、审计流与高基数字段隔离的回归测试。

保护三条纪律：
    1. 事件序号连续（否则回放与 SSE 续传不可信）
    2. 审计流与事件日志分开（否则清理旧事件会误删审计记录）
    3. 高基数字段不做 metric label（否则监控系统会被时间序列打挂）
"""

from __future__ import annotations

import pytest

from codeinsight.domain.trace import (
    APPROVAL_GRANTED,
    AUDIT_EVENT_TYPES,
    FORCE_RECONCILE_OUTCOMES,
    HIGH_CARDINALITY_FIELDS,
    LOW_CARDINALITY_FIELDS,
    NEVER_RECORDED_FIELDS,
    PATCH_APPLIED,
    RUN_STARTED,
    SUPPORTED_EVENT_TYPES,
    TOOL_CALLED,
    VERDICT_PARTIAL,
    AuditRecord,
    ForbiddenFieldError,
    IdempotencyKey,
    MetricLabelCardinalityError,
    ReconcileResult,
    RunEvent,
    TraceContext,
    assert_metric_labels,
    assert_recordable,
    is_auditable,
    requires_reconcile,
)
from codeinsight.infrastructure.event_log import (
    EventSequenceError,
    InMemoryAuditLog,
    InMemoryEventLog,
)

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
SPAN_ID = "00f067aa0ba902b7"


def _event(sequence: int, event_type: str = TOOL_CALLED, **overrides: object) -> RunEvent:
    defaults: dict[str, object] = {
        "event_id": f"ev-{sequence}",
        "run_id": "run-1",
        "sequence": sequence,
        "event_type": event_type,
        "occurred_at_epoch_ms": 1_000 + sequence,
    }
    defaults.update(overrides)
    return RunEvent(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 序号连续性
# ---------------------------------------------------------------------------


def test_events_append_in_order() -> None:
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    log.append(_event(2))
    log.append(_event(3))
    assert log.count("run-1") == 3
    assert log.next_sequence("run-1") == 4


def test_gap_in_sequence_is_refused() -> None:
    """序号有洞会让回放漏掉中间步骤，必须显式失败。"""
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    with pytest.raises(EventSequenceError, match="连续"):
        log.append(_event(3))


def test_duplicate_sequence_is_refused() -> None:
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    with pytest.raises(EventSequenceError):
        log.append(_event(1))


def test_first_event_must_be_sequence_one() -> None:
    log = InMemoryEventLog()
    with pytest.raises(EventSequenceError):
        log.append(_event(5))


def test_sequence_below_one_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="从 1 开始"):
        _event(0)


def test_next_sequence_of_unknown_run_is_one() -> None:
    assert InMemoryEventLog().next_sequence("nope") == 1


def test_runs_have_independent_sequences() -> None:
    """两个 run 各自从 1 开始，互不干扰。"""
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    log.append(_event(1, RUN_STARTED, run_id="run-2", event_id="ev-b1"))
    assert log.count("run-1") == 1
    assert log.count("run-2") == 1


# ---------------------------------------------------------------------------
# SSE 断线续传（增补 R-1）
# ---------------------------------------------------------------------------


def test_read_after_sequence_resumes_from_position() -> None:
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    log.append(_event(2))
    log.append(_event(3))
    resumed = log.read_events("run-1", after_sequence=1)
    assert len(resumed) == 2
    assert resumed[0].sequence == 2


def test_read_after_last_sequence_returns_empty() -> None:
    log = InMemoryEventLog()
    log.append(_event(1, RUN_STARTED))
    assert log.read_events("run-1", after_sequence=1) == ()


def test_read_unknown_run_returns_empty() -> None:
    assert InMemoryEventLog().read_events("nope") == ()


def test_negative_after_sequence_is_refused() -> None:
    with pytest.raises(ValueError):
        InMemoryEventLog().read_events("run-1", after_sequence=-1)


def test_sse_event_id_encodes_run_and_sequence() -> None:
    """客户端把这个值回传为 Last-Event-ID，服务端据此续传。"""
    assert _event(7).sse_event_id == "run-1:7"


# ---------------------------------------------------------------------------
# 不记录的字段（增补 R-6）
# ---------------------------------------------------------------------------


def test_raw_query_cannot_enter_payload() -> None:
    with pytest.raises(ForbiddenFieldError, match="不允许"):
        _event(1, RUN_STARTED, payload={"raw_query": "用户原话"})


def test_hidden_reasoning_cannot_enter_payload() -> None:
    with pytest.raises(ForbiddenFieldError):
        _event(1, RUN_STARTED, payload={"model_reasoning": "..."})


def test_absolute_path_cannot_enter_payload() -> None:
    with pytest.raises(ForbiddenFieldError):
        _event(1, RUN_STARTED, payload={"repo_absolute_path": "C:/repos/shop"})


def test_safe_payload_is_accepted() -> None:
    event = _event(1, RUN_STARTED, payload={"task_type": "change", "tool_name": "read_file"})
    assert event.payload["task_type"] == "change"


def test_assert_recordable_accepts_a_plain_list() -> None:
    assert_recordable(["task_type", "outcome"])


def test_assert_recordable_rejects_non_iterable() -> None:
    with pytest.raises(TypeError):
        assert_recordable(42)


# ---------------------------------------------------------------------------
# metric label 基数
# ---------------------------------------------------------------------------


def test_low_cardinality_labels_are_accepted() -> None:
    assert_metric_labels(sorted(LOW_CARDINALITY_FIELDS))


@pytest.mark.parametrize("name", sorted(HIGH_CARDINALITY_FIELDS))
def test_high_cardinality_field_cannot_be_a_metric_label(name: str) -> None:
    """每个新 run_id 都会创建一条新时间序列——这是打挂 Prometheus 的标准方式。"""
    with pytest.raises(MetricLabelCardinalityError):
        assert_metric_labels([name])


def test_never_recorded_field_cannot_be_a_metric_label() -> None:
    with pytest.raises(MetricLabelCardinalityError):
        assert_metric_labels(["api_key"])


def test_the_three_field_tiers_do_not_overlap() -> None:
    """三层必须互斥，否则同一个字段会有两种矛盾的处理方式。"""
    assert not (LOW_CARDINALITY_FIELDS & HIGH_CARDINALITY_FIELDS)
    assert not (LOW_CARDINALITY_FIELDS & NEVER_RECORDED_FIELDS)
    assert not (HIGH_CARDINALITY_FIELDS & NEVER_RECORDED_FIELDS)


# ---------------------------------------------------------------------------
# 审计流独立（增补 R-5）
# ---------------------------------------------------------------------------


def test_audit_event_types_are_all_known_events() -> None:
    assert AUDIT_EVENT_TYPES <= SUPPORTED_EVENT_TYPES


@pytest.mark.parametrize("event_type", sorted(AUDIT_EVENT_TYPES))
def test_audit_events_are_flagged(event_type: str) -> None:
    assert is_auditable(event_type)


def test_ordinary_event_is_not_auditable() -> None:
    assert not is_auditable(TOOL_CALLED)
    assert not _event(1, TOOL_CALLED).requires_audit


def test_approval_event_requires_audit() -> None:
    assert _event(1, APPROVAL_GRANTED).requires_audit


def _audit(**overrides: object) -> AuditRecord:
    defaults: dict[str, object] = {
        "audit_id": "aud-1",
        "run_id": "run-1",
        "event_type": APPROVAL_GRANTED,
        "actor": "local",
        "occurred_at_epoch_ms": 1_000,
        "subject": "hash-a",
        "outcome": "granted",
    }
    defaults.update(overrides)
    return AuditRecord(**defaults)  # type: ignore[arg-type]


def test_audit_log_stores_and_reads_records() -> None:
    log = InMemoryAuditLog()
    log.record(_audit())
    log.record(_audit(audit_id="aud-2", event_type=PATCH_APPLIED, outcome="applied"))
    records = log.read_records("run-1")
    assert len(records) == 2
    assert log.total_records() == 2


def test_audit_log_refuses_non_audit_event_type() -> None:
    """普通事件不该进审计流——混进来会让审计流的保留策略被滥用。"""
    with pytest.raises(ValueError, match="不属于审计事件"):
        _audit(event_type=TOOL_CALLED)


def test_audit_record_requires_an_actor() -> None:
    """没有操作者的审计记录回答不了「谁批准的」。"""
    with pytest.raises(ValueError, match="操作者"):
        _audit(actor="  ")


def test_audit_record_refuses_unknown_outcome() -> None:
    with pytest.raises(ValueError, match="审计结果"):
        _audit(outcome="maybe")


def test_audit_record_details_are_also_filtered() -> None:
    with pytest.raises(ForbiddenFieldError):
        _audit(details={"approval_token": "tok-1"})


def test_duplicate_audit_id_is_refused() -> None:
    log = InMemoryAuditLog()
    log.record(_audit())
    with pytest.raises(ValueError, match="已存在"):
        log.record(_audit())


def test_audit_log_has_no_delete_method() -> None:
    """刻意不提供清理接口——提供了早晚会有人调用。"""
    log = InMemoryAuditLog()
    assert not hasattr(log, "delete")
    assert not hasattr(log, "purge")
    assert not hasattr(log, "clear")


# ---------------------------------------------------------------------------
# Trace Context
# ---------------------------------------------------------------------------


def test_traceparent_is_w3c_shaped() -> None:
    trace = TraceContext(trace_id=TRACE_ID, span_id=SPAN_ID)
    assert trace.traceparent == f"00-{TRACE_ID}-{SPAN_ID}-01"


@pytest.mark.parametrize(
    ("trace_id", "span_id"),
    [
        ("short", SPAN_ID),
        (TRACE_ID, "short"),
        (TRACE_ID.upper(), SPAN_ID),  # 必须小写
        ("0" * 32, SPAN_ID),  # 全 0 无效
    ],
)
def test_invalid_trace_context_is_refused(trace_id: str, span_id: str) -> None:
    with pytest.raises(ValueError):
        TraceContext(trace_id=trace_id, span_id=span_id)


def test_event_can_carry_trace_context() -> None:
    trace = TraceContext(trace_id=TRACE_ID, span_id=SPAN_ID)
    assert _event(1, RUN_STARTED, trace=trace).trace is trace


# ---------------------------------------------------------------------------
# 幂等键三粒度
# ---------------------------------------------------------------------------


def test_action_key_includes_fingerprint() -> None:
    """键里必须带 diff_hash，否则应用补丁 A 和 B 会撞成同一个键。"""
    key = IdempotencyKey.for_action(goal_id="goal-1", action="apply_patch", fingerprint="hash-a")
    assert key.scope == "action"
    assert "hash-a" in key.key


def test_action_keys_differ_for_different_patches() -> None:
    first = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-a")
    second = IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="hash-b")
    assert first.key != second.key


def test_action_key_without_fingerprint_is_refused() -> None:
    with pytest.raises(ValueError, match="指纹"):
        IdempotencyKey.for_action(goal_id="g", action="apply_patch", fingerprint="")


def test_unknown_idempotency_scope_is_refused() -> None:
    with pytest.raises(ValueError, match="粒度"):
        IdempotencyKey(scope="whatever", key="k")


# ---------------------------------------------------------------------------
# 对账（增补 R-4）
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome", sorted(FORCE_RECONCILE_OUTCOMES))
def test_unknown_and_timeout_force_reconcile(outcome: str) -> None:
    """这两个结果不代表失败，只代表「不知道成没成」。当成失败重试会应用两次。"""
    assert requires_reconcile(outcome)


def test_plain_error_does_not_force_reconcile() -> None:
    assert not requires_reconcile("INVALID_ARGUMENT")


def test_partial_verdict_requires_manual_review() -> None:
    """「改了一半」自动修复只会更糟，一律转人工。"""
    result = ReconcileResult(
        run_id="run-1",
        action="apply_patch",
        verdict=VERDICT_PARTIAL,
        observed_fingerprint="tree-x",
        evidence="workspace 里 2 个文件已改、1 个未改",
    )
    assert result.requires_manual_review


def test_reconcile_verdict_must_be_three_state() -> None:
    with pytest.raises(ValueError, match="三态"):
        ReconcileResult(
            run_id="run-1",
            action="apply_patch",
            verdict="true",
            observed_fingerprint="tree-x",
            evidence="读到的事实",
        )


def test_reconcile_requires_evidence() -> None:
    """结论必须基于读到的仓库事实，不能凭猜。"""
    with pytest.raises(ValueError, match="依据"):
        ReconcileResult(
            run_id="run-1",
            action="apply_patch",
            verdict="applied",
            observed_fingerprint="tree-x",
            evidence="  ",
        )
