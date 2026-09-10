"""事件日志与审计的领域模型。

为什么要事件日志，而不是只存「当前状态」：

    只存当前状态时，用户问「它刚才干了什么」只能回答「它现在在 CHECKING」。
    事件日志让每一步可回放——排障、复现、审计三件事都依赖它。

三条纪律（来自 docs/PLAN_2.0.md Step 2 与增补 R-5 / R-6）：

    1. **不落隐藏推理**。只记录决策与结果，不记录模型的 reasoning 原文。
       写进日志的东西迟早会被人看到，隐藏推理不该在其中。

    2. **审计流独立于 Trace，永不采样**（R-5）。Trace 为了控成本会丢弃一部分
       Span；审计记录一旦被采样丢弃，「谁批准了什么」就成了空白。两者分开存。

    3. **高基数字段三层隔离**（R-6）。同一个 run_id 放错位置会让监控系统爆掉：
       - 绝不做 metric label（每个新值都会创建一条新时间序列）
       - 可以做 Span attribute（Trace 系统按需索引，不做聚合）
       - 结构化日志字段里也可以
       原始 query 文本与仓库绝对路径则**永不落任何一层**。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# 事件类型
# ---------------------------------------------------------------------------
RUN_STARTED = "run_started"
MODEL_CALLED = "model_called"
MODEL_RESULT = "model_result"
STEP_STARTED = "step_started"
TOOL_CALLED = "tool_called"
TOOL_RESULT = "tool_result"
TOOL_CALL_REQUESTED = "tool_call_requested"
TOOL_CALL_VALIDATED = "tool_call_validated"
TOOL_DISPATCHED = "tool_dispatched"
TOOL_RESULT_COMMITTED = "tool_result_committed"
TOOL_ABORTED = "tool_aborted"
EVIDENCE_SELECTED = "evidence_selected"
PLAN_PROPOSED = "plan_proposed"
PATCH_GENERATED = "patch_generated"
APPROVAL_REQUESTED = "approval_requested"
APPROVAL_GRANTED = "approval_granted"
APPROVAL_DENIED = "approval_denied"
CHECKPOINT_CREATED = "checkpoint_created"
PATCH_APPLIED = "patch_applied"
VALIDATION_STARTED = "validation_started"
VALIDATION_FINISHED = "validation_finished"
VALIDATION_PREFLIGHT = "validation_preflight"
PATCH_REJECTED = "patch_rejected"
REPAIR_ATTEMPTED = "repair_attempted"
ROLLBACK_PERFORMED = "rollback_performed"
STATE_TRANSITIONED = "state_transitioned"
CANCEL_REQUESTED = "cancel_requested"
RECONCILE_PERFORMED = "reconcile_performed"
RUN_FINISHED = "run_finished"
TURN_ACCEPTED = "turn_accepted"
SESSION_LOADED = "session_loaded"
INTENT_CLASSIFIED = "intent_classified"
CONTEXT_ASSEMBLED = "context_assembled"
CONTEXT_COMPACTED = "context_compacted"
RETRIEVAL_STARTED = "retrieval_started"
RETRIEVAL_FINISHED = "retrieval_finished"
MODEL_GENERATING = "model_generating"
ANSWER_READY = "answer_ready"
# Q-009：证据评估与修复的公开事件。只记录状态、计数和动作，
# 不记录隐藏推理、完整 Prompt 或工具 stdout。
EVIDENCE_ASSESSED = "evidence_assessed"
EVIDENCE_REPAIR_STARTED = "evidence_repair_started"
EVIDENCE_REPAIR_FINISHED = "evidence_repair_finished"

SUPPORTED_EVENT_TYPES: frozenset[str] = frozenset(
    {
        RUN_STARTED,
        MODEL_CALLED,
        MODEL_RESULT,
        STEP_STARTED,
        TOOL_CALLED,
        TOOL_RESULT,
        TOOL_CALL_REQUESTED,
        TOOL_CALL_VALIDATED,
        TOOL_DISPATCHED,
        TOOL_RESULT_COMMITTED,
        TOOL_ABORTED,
        EVIDENCE_SELECTED,
        PLAN_PROPOSED,
        PATCH_GENERATED,
        APPROVAL_REQUESTED,
        APPROVAL_GRANTED,
        APPROVAL_DENIED,
        CHECKPOINT_CREATED,
        PATCH_APPLIED,
        VALIDATION_STARTED,
        VALIDATION_FINISHED,
        VALIDATION_PREFLIGHT,
        PATCH_REJECTED,
        REPAIR_ATTEMPTED,
        ROLLBACK_PERFORMED,
        STATE_TRANSITIONED,
        CANCEL_REQUESTED,
        RECONCILE_PERFORMED,
        RUN_FINISHED,
        TURN_ACCEPTED,
        SESSION_LOADED,
        INTENT_CLASSIFIED,
        CONTEXT_ASSEMBLED,
        CONTEXT_COMPACTED,
        RETRIEVAL_STARTED,
        RETRIEVAL_FINISHED,
        MODEL_GENERATING,
        ANSWER_READY,
        EVIDENCE_ASSESSED,
        EVIDENCE_REPAIR_STARTED,
        EVIDENCE_REPAIR_FINISHED,
    }
)

# 必须同时写入独立审计流的事件（增补 R-5）。
# 判据是「事后有人会问『到底谁允许的』」——凡符合的都在这里。
AUDIT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        APPROVAL_REQUESTED,
        APPROVAL_GRANTED,
        APPROVAL_DENIED,
        CHECKPOINT_CREATED,
        PATCH_APPLIED,
        ROLLBACK_PERFORMED,
        RECONCILE_PERFORMED,
    }
)

# ---------------------------------------------------------------------------
# 高基数字段隔离（增补 R-6）
# ---------------------------------------------------------------------------

# 可以做 Prometheus metric label：取值集合有限且稳定。
LOW_CARDINALITY_FIELDS: frozenset[str] = frozenset(
    {
        "event_type",
        "run_status",
        "task_type",
        "mode",
        "tool_name",
        "error_class",
        "route_profile",
        "outcome",
    }
)

# 只能做 Span attribute 或结构化日志字段，**绝不做 metric label**。
HIGH_CARDINALITY_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "session_id",
        "goal_id",
        "trace_id",
        "span_id",
        "evidence_id",
        "patch_id",
        "checkpoint_id",
        "approval_token_id",
        "idempotency_key",
    }
)

# 三层都不许出现。这些字段本身可能含用户意图、本机信息或密钥。
NEVER_RECORDED_FIELDS: frozenset[str] = frozenset(
    {
        "raw_query",
        "raw_question",
        "repo_absolute_path",
        "workspace_absolute_path",
        "model_reasoning",
        "hidden_reasoning",
        "api_key",
        "approval_token",
        "env_dump",
        "full_test_output",
        "file_content",
    }
)


class ForbiddenFieldError(Exception):
    """试图记录一个不允许出现在事件里的字段。"""


class MetricLabelCardinalityError(Exception):
    """试图把高基数字段用作 metric label。"""


def _field_names_of(candidates: object, label: str) -> list[str]:
    """把 dict / list / tuple / set 统一成字段名列表。"""
    if isinstance(candidates, dict):
        names: list[str] = []
        for key in candidates:
            names.append(str(key))
        return names
    if isinstance(candidates, (list, tuple, set, frozenset)):
        names = []
        for item in candidates:
            names.append(str(item))
        return names
    raise TypeError(f"{label} 必须是可迭代的字段名集合")


def assert_recordable(field_names: object) -> None:
    """校验一组字段名是否允许写进事件。

    参数刻意宽松为 ``object``，因为调用方往往直接传 ``dict`` 本身。
    """
    forbidden: list[str] = []
    for name in _field_names_of(field_names, "field_names"):
        if name in NEVER_RECORDED_FIELDS:
            forbidden.append(name)
    if forbidden:
        raise ForbiddenFieldError(
            f"这些字段不允许写进事件日志：{sorted(forbidden)}。"
            "原始 query、绝对路径、密钥与模型隐藏推理三层都不记录（增补 R-6）。"
        )


def assert_metric_labels(label_names: object) -> None:
    """校验一组 metric label 是否安全。

    高基数字段做 label 会让 Prometheus 的时间序列数量随请求量线性增长，
    这是把监控系统打挂的最常见方式。
    """
    offenders: list[str] = []
    for name in _field_names_of(label_names, "label_names"):
        if name in HIGH_CARDINALITY_FIELDS:
            offenders.append(name)
        elif name in NEVER_RECORDED_FIELDS:
            offenders.append(name)
    if offenders:
        raise MetricLabelCardinalityError(
            f"这些字段不能做 metric label：{sorted(offenders)}。"
            "高基数字段只允许出现在 Span attribute 或结构化日志中（增补 R-6）。"
        )


def is_auditable(event_type: str) -> bool:
    """该事件是否必须同时写入永不采样的审计流。"""
    if event_type in AUDIT_EVENT_TYPES:
        return True
    return False


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


def _assert_hex(value: str, label: str) -> None:
    for character in value:
        if character not in "0123456789abcdef":
            raise ValueError(f"{label} 必须是小写十六进制，发现非法字符：{character!r}")


@dataclass(frozen=True)
class TraceContext:
    """W3C Trace Context 的最小承载。

    ``trace_id`` 让一次请求在 Gateway、检索、Sandbox、检查之间能串起来。
    没有它，四个组件的日志就是四堆互不相关的文本。
    """

    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.trace_id) != 32:
            raise ValueError("W3C trace_id 为 32 位十六进制字符")
        if len(self.span_id) != 16:
            raise ValueError("W3C span_id 为 16 位十六进制字符")
        _assert_hex(self.trace_id, "trace_id")
        _assert_hex(self.span_id, "span_id")
        if self.trace_id == "0" * 32:
            raise ValueError("trace_id 不能全为 0")
        if self.parent_span_id is not None:
            if len(self.parent_span_id) != 16:
                raise ValueError("parent_span_id 为 16 位十六进制字符")
            _assert_hex(self.parent_span_id, "parent_span_id")

    @property
    def traceparent(self) -> str:
        """序列化为 W3C ``traceparent`` 头。采样位固定为 01。"""
        return f"00-{self.trace_id}-{self.span_id}-01"


@dataclass(frozen=True)
class RunEvent:
    """事件日志中的一条记录。

    ``sequence`` 在同一个 run 内严格递增且连续——这是回放的前提，也是
    SSE 断线重连时 ``Last-Event-ID`` 的依据（增补 R-1）。
    """

    event_id: str
    run_id: str
    sequence: int
    event_type: str
    occurred_at_epoch_ms: int
    payload: dict[str, str] = field(default_factory=dict)
    trace: TraceContext | None = None

    def __post_init__(self) -> None:
        if not self.event_id.strip():
            raise ValueError("event_id 不能为空")
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if self.sequence < 1:
            raise ValueError("事件序号从 1 开始")
        if self.event_type not in SUPPORTED_EVENT_TYPES:
            raise ValueError(
                f"未登记的事件类型：{self.event_type}。"
                "新增事件类型须同时决定它是否进审计流。"
            )
        if self.occurred_at_epoch_ms <= 0:
            raise ValueError("occurred_at_epoch_ms 必须为正")
        assert_recordable(self.payload)

    @property
    def requires_audit(self) -> bool:
        return is_auditable(self.event_type)

    @property
    def sse_event_id(self) -> str:
        """SSE 的 ``id:`` 字段。重连时客户端回传它，服务端据此续传。"""
        return f"{self.run_id}:{self.sequence}"


@dataclass(frozen=True)
class AuditRecord:
    """独立审计流中的一条记录，永不采样（增补 R-5）。

    与 ``RunEvent`` 刻意分开而不是加个 ``is_audit`` 标记：
    两者的保留期、采样策略与存储位置都不同，混在一张表里迟早会因为
    「清理旧事件」把审计记录一起删掉。
    """

    audit_id: str
    run_id: str
    event_type: str
    actor: str
    occurred_at_epoch_ms: int
    subject: str
    outcome: str
    details: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audit_id.strip():
            raise ValueError("audit_id 不能为空")
        if self.event_type not in AUDIT_EVENT_TYPES:
            raise ValueError(
                f"{self.event_type} 不属于审计事件。"
                f"审计流只接受 {sorted(AUDIT_EVENT_TYPES)}。"
            )
        if not self.actor.strip():
            raise ValueError("审计记录必须写明操作者——否则无法回答「谁批准的」")
        if not self.subject.strip():
            raise ValueError("审计记录必须写明操作对象")
        if self.outcome not in {"granted", "denied", "applied", "rolled_back", "reconciled"}:
            raise ValueError(f"不支持的审计结果：{self.outcome}")
        assert_recordable(self.details)


# ---------------------------------------------------------------------------
# 幂等键
# ---------------------------------------------------------------------------

IDEMPOTENCY_SCOPE_REQUEST = "request"
IDEMPOTENCY_SCOPE_ATTEMPT = "attempt"
IDEMPOTENCY_SCOPE_ACTION = "action"

SUPPORTED_IDEMPOTENCY_SCOPES: frozenset[str] = frozenset(
    {IDEMPOTENCY_SCOPE_REQUEST, IDEMPOTENCY_SCOPE_ATTEMPT, IDEMPOTENCY_SCOPE_ACTION}
)


@dataclass(frozen=True)
class IdempotencyKey:
    """幂等键，三种粒度不能混用。

    这是 Step 2 里最容易做错的一处。三者的语义完全不同：

    ``request``  同一个 HTTP 请求被重发（网络重试）。重发应返回同一结果。
    ``attempt``  同一次执行尝试中的一步（工具重试）。重试可以重跑。
    ``action``   一个业务动作（应用某个补丁）。**跨 run 也不允许重复执行**。

    混用的典型后果：把 apply_patch 挂在 ``attempt`` 粒度上，于是重试
    一次就应用了两遍补丁。业务动作必须用 ``action`` 粒度，键里带 diff_hash。
    """

    scope: str
    key: str

    def __post_init__(self) -> None:
        if self.scope not in SUPPORTED_IDEMPOTENCY_SCOPES:
            raise ValueError(
                f"不支持的幂等粒度：{self.scope}。"
                f"必须是 {sorted(SUPPORTED_IDEMPOTENCY_SCOPES)} 之一。"
            )
        if not self.key.strip():
            raise ValueError("幂等键不能为空")

    @classmethod
    def for_action(cls, *, goal_id: str, action: str, fingerprint: str) -> IdempotencyKey:
        """构造业务动作级幂等键。

        键里必须含 ``fingerprint``（如 diff_hash）——否则「应用补丁 A」和
        「应用补丁 B」会撞成同一个键，第二个补丁被静默跳过。
        """
        if not goal_id.strip():
            raise ValueError("业务动作幂等键必须绑定 goal_id")
        if not fingerprint.strip():
            raise ValueError("业务动作幂等键必须包含内容指纹")
        return cls(scope=IDEMPOTENCY_SCOPE_ACTION, key=f"{goal_id}:{action}:{fingerprint}")


# ---------------------------------------------------------------------------
# 对账（增补 R-4）
# ---------------------------------------------------------------------------

VERDICT_APPLIED = "applied"
VERDICT_NOT_APPLIED = "not_applied"
VERDICT_PARTIAL = "partial"

SUPPORTED_RECONCILE_VERDICTS: frozenset[str] = frozenset(
    {VERDICT_APPLIED, VERDICT_NOT_APPLIED, VERDICT_PARTIAL}
)

# 这些工具结果必须强制对账。原因：它们**不代表操作失败**，只代表
# 「我不知道成没成」。当成失败去重试，就会把补丁应用两次。
FORCE_RECONCILE_OUTCOMES: frozenset[str] = frozenset({"UNKNOWN", "TIMEOUT"})


@dataclass(frozen=True)
class ReconcileResult:
    """一次对账结论。三态而不是布尔——``partial`` 是真实会发生的情况。"""

    run_id: str
    action: str
    verdict: str
    observed_fingerprint: str
    evidence: str

    def __post_init__(self) -> None:
        if self.verdict not in SUPPORTED_RECONCILE_VERDICTS:
            raise ValueError(
                f"不支持的对账结论：{self.verdict}。"
                "必须是 applied / not_applied / partial 三态之一——"
                "布尔两态无法表达「改了一半」。"
            )
        if not self.evidence.strip():
            raise ValueError("对账结论必须给出依据（读到的仓库事实），不能凭猜")

    @property
    def requires_manual_review(self) -> bool:
        """``partial`` 一律转人工。自动修复一个说不清状态的仓库会更糟。"""
        return self.verdict == VERDICT_PARTIAL


def requires_reconcile(outcome: str) -> bool:
    """该工具结果是否必须走对账再决定下一步（增补 R-4）。"""
    if outcome in FORCE_RECONCILE_OUTCOMES:
        return True
    return False
