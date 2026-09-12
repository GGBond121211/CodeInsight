"""SQLAlchemy ORM 表定义。

设计上的几处关键判断，逐条说明理由：

**1. ``RunRow`` 用 ``version_id_col`` 做乐观锁**

    ``__mapper_args__ = {"version_id_col": state_version,
                         "version_id_generator": False}``

    ``version_id_col`` 让 SQLAlchemy 在每次 UPDATE 时自动加上
    ``WHERE state_version = <读到的值>``，并在 rowcount 为 0 时抛
    ``StaleDataError``。这正是 CAS。

    ``version_id_generator=False`` 是必须的：默认情况下 SQLAlchemy 会
    **自己**把版本号 +1，那样领域对象里的 ``state_version`` 就成了摆设，
    两套编号机制会打架。设为 False 后版本号由领域层给出，SQLAlchemy 只负责
    把它写进 WHERE 子句。**领域层拥有语义，ORM 只提供机制。**

**2. 所有表显式 InnoDB + utf8mb4**

    InnoDB 才有行锁与事务，MyISAM 没有——CAS 在 MyISAM 上不成立。
    utf8mb4 而不是 utf8：MySQL 的 "utf8" 是每字符最多 3 字节的残缺实现，
    存不了 emoji 与部分 CJK 扩展字符。本项目要处理中文提问，必须 utf8mb4。

**3. ``run_events`` 上有 ``UNIQUE(run_id, sequence)``**

    序号连续性不能只靠应用层检查——两个并发写入各自读到「下一个是 5」，
    应用层都放行，结果两条 sequence=5。唯一约束是最后一道防线，
    由数据库保证。

**4. 审计表与事件表分开**

    不是加一个 ``is_audit`` 列。两者保留期与采样策略不同，同一张表里
    早晚会有一段「清理 90 天前的事件」把审计记录一起删掉（增补 R-5）。

**5. ``tenant_id`` / ``user_id`` 进每张业务表并建索引**

    当前恒为单值、不做鉴权（增补 R-3）。现在加的成本是两个列，
    将来不加的成本是改全部表结构与缓存键。
"""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# 所有表统一的建表参数。写成常量而不是每张表重复，避免漏掉某一张。
TABLE_ARGS: dict[str, str] = {
    "mysql_engine": "InnoDB",
    "mysql_charset": "utf8mb4",
    "mysql_collate": "utf8mb4_0900_ai_ci",
}

# 标识符列宽。36 足够放 UUID；固定下来避免各表宽度不一致导致 JOIN 时隐式转换。
ID_LENGTH = 64
# 内容指纹为 sha256 十六进制，固定 64 字符。
FINGERPRINT_LENGTH = 64


class Base(DeclarativeBase):
    """全部 ORM 表的基类。"""


class SessionRow(Base):
    """一段对话。"""

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_tenant_user", "tenant_id", "user_id"),
        TABLE_ARGS,
    )

    session_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    user_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    repo_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    active_goal_id: Mapped[str | None] = mapped_column(String(ID_LENGTH), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 会话绑定的仓库根。跨进程 Worker 靠它知道该读哪个目录。
    repo_root: Mapped[str | None] = mapped_column(String(1024), nullable=True)


class MemoryRecordRow(Base):
    """Working、Session 与 Semantic Memory 的独立事实表。"""

    __tablename__ = "memory_records"
    __table_args__ = (
        Index(
            "ix_memory_scope_owner",
            "tenant_id",
            "user_id",
            "repo_id",
            "layer",
            "owner_id",
            "status",
        ),
        TABLE_ARGS,
    )

    record_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    user_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    repo_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    layer: Mapped[str] = mapped_column(String(16), nullable=False)
    owner_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_estimate: Mapped[int] = mapped_column(Integer, nullable=False)
    is_trusted: Mapped[bool] = mapped_column(Boolean, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    expires_at_epoch_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)


class GoalRow(Base):
    """一个用户目标。"""

    __tablename__ = "goals"
    __table_args__ = (
        Index("ix_goals_session_status", "session_id", "status"),
        Index("ix_goals_tenant_user", "tenant_id", "user_id"),
        TABLE_ARGS,
    )

    goal_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    user_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    repo_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    user_goal: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    # 目标文件范围。以换行分隔的相对路径列表——不建关联表，因为它永远
    # 只被整体读写，拆表只会多一次 JOIN。
    target_scope: Mapped[str] = mapped_column(Text, nullable=False, default="")
    validation_profile: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)


class RunRow(Base):
    """一次执行尝试。``state_version`` 是乐观锁列。"""

    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_goal", "goal_id"),
        Index("ix_runs_session_status", "session_id", "status"),
        TABLE_ARGS,
    )

    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    goal_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    user_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    state_version: Mapped[int] = mapped_column(Integer, nullable=False)
    step_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_steps: Mapped[int] = mapped_column(Integer, nullable=False)
    next_action: Mapped[str | None] = mapped_column(String(255), nullable=True)
    deadline_epoch_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    token_budget: Mapped[int | None] = mapped_column(Integer, nullable=True)
    tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    repair_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_repair_attempts: Mapped[int] = mapped_column(Integer, nullable=False)
    stuck_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    active_tool: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 必须写在 state_version 之后：此处引用的是类体命名空间里的列对象。
    # 见模块 docstring 第 1 条——领域层拥有版本语义，ORM 只提供 WHERE 机制。
    __mapper_args__ = {
        "version_id_col": state_version,
        "version_id_generator": False,
    }


class RunEventRow(Base):
    """事件日志。``UNIQUE(run_id, sequence)`` 是序号唯一性的最后防线。"""

    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_events_run_sequence"),
        Index("ix_run_events_run_sequence", "run_id", "sequence"),
        TABLE_ARGS,
    )

    event_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # payload 是已过 assert_recordable 过滤的字段，存为 JSON 文本。
    # 不用 MySQL 的 JSON 类型：这里从不做 JSON 路径查询，只整体读写，
    # 用 TEXT 更简单且跨数据库可移植。
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)
    span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)
    parent_span_id: Mapped[str | None] = mapped_column(String(16), nullable=True)


class AuditRecordRow(Base):
    """独立审计流。与 ``run_events`` 分表，永不采样、不参与事件清理。"""

    __tablename__ = "audit_records"
    __table_args__ = (
        Index("ix_audit_records_run", "run_id"),
        Index("ix_audit_records_time", "occurred_at_epoch_ms"),
        TABLE_ARGS,
    )

    audit_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    actor: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    occurred_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")


class GatewayCostRow(Base):
    """每次模型 attempt 的版本化成本事实，不随后续价格调整重算覆盖。"""

    __tablename__ = "gateway_cost_records"
    __table_args__ = (
        Index("ix_gateway_cost_request", "request_id"),
        Index("ix_gateway_cost_tenant_scene", "tenant_id", "scene"),
        TABLE_ARGS,
    )

    attempt_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    request_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    tenant_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    user_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    scene: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model_tier: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    cached_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    price_version: Mapped[str] = mapped_column(String(128), nullable=False)
    total_stars: Mapped[Decimal] = mapped_column(Numeric(20, 8), nullable=False)
    latency_milliseconds: Mapped[float] = mapped_column(Float, nullable=False)
    ttft_milliseconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_class: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ApprovalRow(Base):
    """审批令牌。

    ``consumed_at_epoch_ms`` 为 NULL 表示未消费。一次性消费靠
    ``UPDATE ... WHERE token = ? AND consumed_at_epoch_ms IS NULL``
    的 rowcount 判定，不靠先读后写。
    """

    __tablename__ = "approvals"
    __table_args__ = (
        Index("ix_approvals_run", "run_id"),
        TABLE_ARGS,
    )

    token: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    diff_hash: Mapped[str] = mapped_column(String(FINGERPRINT_LENGTH), nullable=False)
    base_fingerprint: Mapped[str] = mapped_column(String(FINGERPRINT_LENGTH), nullable=False)
    approval_scope: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    consumed_at_epoch_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)


class IdempotencyRow(Base):
    """幂等键登记表。

    主键是 ``(scope, key_value)`` 复合键——三种粒度的键空间必须隔离，
    否则 request 级的键会和 action 级的键互相误命中。
    """

    __tablename__ = "idempotency_keys"
    __table_args__ = (TABLE_ARGS,)

    scope: Mapped[str] = mapped_column(String(16), primary_key=True)
    key_value: Mapped[str] = mapped_column(String(255), primary_key=True)
    result_ref: Mapped[str] = mapped_column(String(255), nullable=False)


class WorkspaceRow(Base):
    """隔离 workspace 记录。原仓库路径与 workspace 路径分列保存。"""

    __tablename__ = "workspaces"
    __table_args__ = (
        Index("ix_workspaces_run", "run_id"),
        TABLE_ARGS,
    )

    workspace_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    source_repo_path: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)
    is_disposed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class AgentRunOutputRow(Base):
    """一次 Agent Run 尝试的公开产出。

    为什么单独一张表，而不是给 agent_runs 加几列：agent_runs 的契约是「只存标识、
    状态、租约与参数」（谁在领、第几次 attempt），加进正文之后，那张表的每一行都
    变成一个内容对象，恢复巡检、队列计数与租约 CAS 都要拖着它走。产出是另一类
    事实，用主键关联、按需读取。

    attempt 保留下来是为了回答「这是第几次尝试的产出」：审批续跑会覆盖它，覆盖
    之前的那一版属于上一手，不该被当成当前结论。
    """

    __tablename__ = "agent_run_outputs"
    __table_args__ = (TABLE_ARGS,)

    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False)
    assistant_message: Mapped[str] = mapped_column(Text, nullable=False)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    updated_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)


class AgentRunRow(Base):
    """后台 Agent Run 的调度事实，与 run_events 一起支撑重启后恢复。

    与 runs 的分工：runs 保存一次执行的状态版本与步数（带乐观锁），这张表保存
    调度层的事实——谁在领、第几次 attempt、租约什么时候过期。它们是两类问题，
    合成一张表会让乐观锁去管租约，两套版本号迟早打架。

    UNIQUE(turn_id) 是「同一条用户消息不会创建两个 Run」的最后防线：重发的
    请求只能命中已有记录，不能新开一条并再跑一次模型。
\n+    attempt 与 task_id 合起来就是计划里的 task_attempt_id：第几次尝试由这两个
    字段共同确定，单独再存一个 ID 只会多一处可能对不上的事实。

    这里没有 tenant_id / user_id：本表不含任何内容字段，租户归属由 session_id
    指向的 sessions 决定，重复一份维度只会带来两处不一致的可能。
    """

    __tablename__ = "agent_runs"
    __table_args__ = (
        UniqueConstraint("turn_id", name="uq_agent_runs_turn"),
        Index("ix_agent_runs_status", "status"),
        Index("ix_agent_runs_session", "session_id"),
        TABLE_ARGS,
    )

    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    turn_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    session_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    task_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    task_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=2)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_until_epoch_ms: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    deadline_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    error_class: Mapped[str | None] = mapped_column(String(64), nullable=True)
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # 本轮被要求的参数。Worker 只有 run_id，重建执行时必须能读到它们。
    validation_profile: Mapped[str] = mapped_column(
        String(64), nullable=False, default="python_compile"
    )
    result_limit: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    show_debug_reasoning: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    # 续跑任务（审批恢复 / 校验恢复）要用到的补丁与审批令牌。
    patch_id: Mapped[str | None] = mapped_column(String(ID_LENGTH), nullable=True)
    approval_token: Mapped[str | None] = mapped_column(String(255), nullable=True)


class CheckpointRow(Base):
    """workspace 的可回滚点。"""

    __tablename__ = "checkpoints"
    __table_args__ = (
        Index("ix_checkpoints_run_time", "run_id", "created_at_epoch_ms"),
        TABLE_ARGS,
    )

    checkpoint_id: Mapped[str] = mapped_column(String(ID_LENGTH), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(ID_LENGTH), nullable=False)
    workspace_path: Mapped[str] = mapped_column(Text, nullable=False)
    tree_fingerprint: Mapped[str] = mapped_column(String(FINGERPRINT_LENGTH), nullable=False)
    created_at_epoch_ms: Mapped[int] = mapped_column(BigInteger, nullable=False)
