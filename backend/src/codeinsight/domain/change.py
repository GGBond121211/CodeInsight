"""代码修改任务的领域状态：Session、Goal、Run 及其状态机。

为什么要分三层（这是 Step 2 最核心的设计判断）：

    Session  一段对话。用户可以隔几天回来，长期容器。
    Goal     一个用户目标。从提出到完成或放弃。
    Run      一次执行尝试。有限、可重放。

    不能合成两层的原因是**重试**。同一个 Goal 可能失败三次才成功；
    如果没有 Run 这一层，步数和预算就没地方记——最后会得到一个执行了
    40 步的 Goal，分不清那是一次长任务还是四次失败重试。

    计划原文：「每次 Run 是可重放的有限尝试；多轮 Session 是长期容器，
    不能把一次 Run 的 step counter 无限累加。」

本模块只定义不可变值对象与合法状态迁移，不依赖 FastAPI、SQLAlchemy 或任何
外部框架。持久化实现在 infrastructure 层，契约在 domain/ports.py。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# Run 状态
#
# 九种状态来自 docs/PLAN_2.0.md Step 2。终态之外的每一种都必须能解释
# 「现在在等什么」，否则用户会看到一个转圈但没人知道原因的任务。
# ---------------------------------------------------------------------------
RUNNING = "RUNNING"
WAITING_USER = "WAITING_USER"
WAITING_APPROVAL = "WAITING_APPROVAL"
CHECKING = "CHECKING"
COMPLETED = "COMPLETED"
FAILED = "FAILED"
REVIEW_REQUIRED = "REVIEW_REQUIRED"
ROLLED_BACK = "ROLLED_BACK"
STUCK = "STUCK"

SUPPORTED_RUN_STATUSES: frozenset[str] = frozenset(
    {
        RUNNING,
        WAITING_USER,
        WAITING_APPROVAL,
        CHECKING,
        COMPLETED,
        FAILED,
        REVIEW_REQUIRED,
        ROLLED_BACK,
        STUCK,
    }
)

# 终态。进入终态后不允许再迁移——这是防止「已回滚的任务又被恢复执行」的闸门。
TERMINAL_RUN_STATUSES: frozenset[str] = frozenset({COMPLETED, FAILED, ROLLED_BACK})

# 合法迁移表。显式列举而不是「除了这些都行」，因为漏掉一条非法迁移的代价
# 是状态机静默走进不该到的地方。
LEGAL_RUN_TRANSITIONS: dict[str, frozenset[str]] = {
    RUNNING: frozenset(
        {WAITING_USER, WAITING_APPROVAL, CHECKING, COMPLETED, FAILED, REVIEW_REQUIRED, STUCK}
    ),
    WAITING_USER: frozenset({RUNNING, FAILED}),
    # 审批被拒时，若已建过 checkpoint 就要回滚，否则直接失败
    WAITING_APPROVAL: frozenset({RUNNING, FAILED, ROLLED_BACK}),
    # 检查失败进入有限修复（回到 RUNNING）；修复次数用尽转人工或回滚
    CHECKING: frozenset({RUNNING, COMPLETED, REVIEW_REQUIRED, ROLLED_BACK, FAILED}),
    REVIEW_REQUIRED: frozenset({RUNNING, COMPLETED, ROLLED_BACK, FAILED}),
    STUCK: frozenset({RUNNING, ROLLED_BACK, FAILED}),
    COMPLETED: frozenset(),
    FAILED: frozenset(),
    ROLLED_BACK: frozenset(),
}

# ---------------------------------------------------------------------------
# Goal 状态与类型
# ---------------------------------------------------------------------------
GOAL_ACTIVE = "ACTIVE"
GOAL_COMPLETED = "COMPLETED"
GOAL_ABANDONED = "ABANDONED"

SUPPORTED_GOAL_STATUSES: frozenset[str] = frozenset({GOAL_ACTIVE, GOAL_COMPLETED, GOAL_ABANDONED})

SUPPORTED_TASK_TYPES: frozenset[str] = frozenset(
    {"explain", "locate", "change", "failing_test_repair", "unknown"}
)

# 模式决定是否允许写。默认只读——这是从 1.0 继承的地基。
MODE_READ_ONLY = "read_only"
MODE_ISOLATED_WRITE = "isolated_write"
SUPPORTED_MODES: frozenset[str] = frozenset({MODE_READ_ONLY, MODE_ISOLATED_WRITE})

# ---------------------------------------------------------------------------
# 取消语义分级（增补 R-1）
#
# 「用户说算了」不等于「远端已经停下了」。只读工具没有副作用，可以立刻中断；
# 写类工具中途打断会留下谁也说不清的中间状态，必须等它结束并对账。
# ---------------------------------------------------------------------------
INTERRUPTIBLE_TOOLS: frozenset[str] = frozenset(
    {
        "search_repository",
        "read_file",
        "get_repository_map",
        "lsp_definition",
        "scip_references",
        "get_evidence_context",
        "get_run_events",
        "get_diff",
        "generate_patch",
        "validate_patch",
    }
)

NON_INTERRUPTIBLE_TOOLS: frozenset[str] = frozenset(
    {
        "create_checkpoint",
        "apply_patch_isolated",
        "run_allowlisted_checks",
        "rollback_workspace",
    }
)


def is_interruptible(tool_name: str) -> bool:
    """判断某工具是否允许在取消时立即中断。

    未登记的工具**按不可中断处理**——保守假设它有副作用。这个默认值方向
    很重要：猜错方向的代价是留下半应用的补丁。
    """
    if tool_name in INTERRUPTIBLE_TOOLS:
        return True
    return False


class IllegalTransitionError(Exception):
    """尝试了状态机不允许的迁移。"""


class StateVersionConflictError(Exception):
    """stateVersion 不匹配，拒绝恢复。

    这是防止重复 resume 导致补丁被应用两次的核心异常。发生时正确做法是
    重新读取仓库事实，而不是重试。
    """


def assert_legal_transition(current: str, target: str) -> None:
    """校验一次状态迁移是否合法，非法时抛出并说明原因。"""
    if current not in SUPPORTED_RUN_STATUSES:
        raise IllegalTransitionError(f"未知的当前状态：{current}")
    if target not in SUPPORTED_RUN_STATUSES:
        raise IllegalTransitionError(f"未知的目标状态：{target}")
    if current in TERMINAL_RUN_STATUSES:
        raise IllegalTransitionError(
            f"{current} 是终态，不能再迁移到 {target}。"
            "已完成或已回滚的 Run 不允许被重新执行。"
        )
    allowed = LEGAL_RUN_TRANSITIONS[current]
    if target not in allowed:
        raise IllegalTransitionError(
            f"不允许从 {current} 迁移到 {target}；允许的目标为 {sorted(allowed)}。"
        )


# ---------------------------------------------------------------------------
# 值对象
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TenantScope:
    """租户与用户维度（增补 R-3）。

    **当前恒为单值，不做鉴权**。存在的意义是让隔离维度提前进入数据模型、
    缓存键与 Trace 索引，将来加租户不需要改表结构和失效逻辑。
    这是刻意的范围控制，不是遗漏。
    """

    tenant_id: str = "default"
    user_id: str = "local"

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id 不能为空")
        if not self.user_id.strip():
            raise ValueError("user_id 不能为空")


@dataclass(frozen=True)
class ConversationTurn:
    """一轮对话。只保存公开内容，不保存模型隐藏推理。"""

    sequence: int
    role: str
    content: str

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("对话轮次从 1 开始")
        if self.role not in {"user", "assistant"}:
            raise ValueError(f"不支持的 role：{self.role}")
        if not self.content.strip():
            raise ValueError("对话内容不能为空")


@dataclass(frozen=True)
class ConversationSession:
    """一段长期存在的对话容器。"""

    session_id: str
    scope: TenantScope
    repo_id: str
    # 会话绑定的仓库根目录。Worker 在另一个进程里只有 session_id，没有请求对象；
    # 没有这一项它就不知道该去读哪个目录，恢复也就无从谈起。
    # 绝对路径属于敏感字段：它只进事实表，不进事件 payload。
    repo_root: str = ""
    recent_turns: tuple[ConversationTurn, ...] = ()
    summary: str | None = None
    active_goal_id: str | None = None
    compacted_through_sequence: int = 0

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not self.repo_id.strip():
            raise ValueError("repo_id 不能为空")
        if self.compacted_through_sequence < 0:
            raise ValueError("compacted_through_sequence 不能为负")
        expected = self.compacted_through_sequence + 1
        for turn in self.recent_turns:
            if turn.sequence != expected:
                raise ValueError(
                    f"对话轮次必须连续；期望 {expected}，实际 {turn.sequence}"
                )
            expected += 1

    def with_turn(self, role: str, content: str) -> ConversationSession:
        """追加一轮对话，返回新实例。"""
        turn = ConversationTurn(
            self.compacted_through_sequence + len(self.recent_turns) + 1,
            role,
            content,
        )
        return replace(self, recent_turns=self.recent_turns + (turn,))

    def with_active_goal(self, goal_id: str | None) -> ConversationSession:
        return replace(self, active_goal_id=goal_id)


def merge_session_facts(
    existing: ConversationSession, incoming: ConversationSession
) -> ConversationSession:
    """把一次会话写入合并到已有事实上。

    规则只有一条：仓库根一旦绑定就不被空值抹掉。它是会话级的一次性事实
    （换仓库会被 _resolve_session_repository 拒绝），而进程内缓存里的会话副本
    可能是在绑定之前加载的——直接覆盖会让跨进程的 Worker 失去读哪个目录的依据。
    """

    if not incoming.repo_root and existing.repo_root:
        return replace(incoming, repo_root=existing.repo_root)
    return incoming


@dataclass(frozen=True)
class CodeGoal:
    """一个用户目标。跨多次 Run 存活。"""

    goal_id: str
    session_id: str
    scope: TenantScope
    repo_id: str
    task_type: str
    user_goal: str
    mode: str = MODE_READ_ONLY
    target_scope: tuple[str, ...] = ()
    validation_profile: str | None = None
    status: str = GOAL_ACTIVE

    def __post_init__(self) -> None:
        if not self.goal_id.strip():
            raise ValueError("goal_id 不能为空")
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not self.user_goal.strip():
            raise ValueError("user_goal 不能为空")
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ValueError(f"不支持的 task_type：{self.task_type}")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"不支持的 mode：{self.mode}")
        if self.status not in SUPPORTED_GOAL_STATUSES:
            raise ValueError(f"不支持的 goal status：{self.status}")
        if self.mode == MODE_ISOLATED_WRITE and not self.validation_profile:
            raise ValueError(
                "isolated_write 模式必须指定 validation_profile——"
                "允许写但没有固定检查，等于放开了没有验证的修改"
            )

    @property
    def allows_write(self) -> bool:
        return self.mode == MODE_ISOLATED_WRITE


@dataclass(frozen=True)
class RunSnapshot:
    """一次执行尝试的可持久化快照。

    ``state_version`` 是并发闸门：恢复时版本对不上就拒绝（见
    StateVersionConflictError）。每次状态变更都必须递增它。
    """

    run_id: str
    session_id: str
    goal_id: str
    scope: TenantScope
    status: str = RUNNING
    state_version: int = 1
    step_count: int = 0
    max_steps: int = 8
    next_action: str | None = None
    deadline_epoch_ms: int | None = None
    token_budget: int | None = None
    tokens_used: int = 0
    repair_attempts: int = 0
    max_repair_attempts: int = 2
    stuck_reason: str | None = None
    active_tool: str | None = None

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not self.goal_id.strip():
            raise ValueError("goal_id 不能为空")
        if self.status not in SUPPORTED_RUN_STATUSES:
            raise ValueError(f"不支持的 run status：{self.status}")
        if self.state_version < 1:
            raise ValueError("state_version 从 1 开始")
        if self.step_count < 0:
            raise ValueError("step_count 不能为负")
        if self.max_steps < 1:
            raise ValueError("max_steps 必须为正")
        if self.step_count > self.max_steps:
            raise ValueError(
                f"step_count {self.step_count} 超过上限 {self.max_steps}；"
                "步数上限是防失控的硬约束，不允许越过"
            )
        if self.repair_attempts < 0:
            raise ValueError("repair_attempts 不能为负")
        if self.repair_attempts > self.max_repair_attempts:
            raise ValueError(
                f"repair_attempts {self.repair_attempts} 超过上限 "
                f"{self.max_repair_attempts}；超限必须转 REVIEW_REQUIRED 或回滚"
            )
        if self.tokens_used < 0:
            raise ValueError("tokens_used 不能为负")
        if self.status == STUCK and not self.stuck_reason:
            raise ValueError("进入 STUCK 必须说明原因，否则无法排障")

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_RUN_STATUSES

    @property
    def steps_remaining(self) -> int:
        return self.max_steps - self.step_count

    @property
    def repairs_remaining(self) -> int:
        return self.max_repair_attempts - self.repair_attempts

    def transition_to(self, target: str, *, reason: str | None = None) -> RunSnapshot:
        """迁移到新状态并递增 state_version。非法迁移直接抛错。"""
        assert_legal_transition(self.status, target)
        stuck_reason = self.stuck_reason
        if target == STUCK:
            if not reason:
                raise ValueError("迁移到 STUCK 必须提供原因")
            stuck_reason = reason
        return replace(
            self,
            status=target,
            state_version=self.state_version + 1,
            stuck_reason=stuck_reason,
            active_tool=None,
        )

    def with_step(self, *, tool_name: str | None = None) -> RunSnapshot:
        """步数 +1。超过上限时转 STUCK 而不是抛错——超限是可预期的业务结果。"""
        if self.is_terminal:
            raise IllegalTransitionError(f"{self.status} 是终态，不能继续执行步骤")
        next_count = self.step_count + 1
        if next_count > self.max_steps:
            return self.transition_to(STUCK, reason=f"步数达到上限 {self.max_steps}")
        return replace(
            self,
            step_count=next_count,
            state_version=self.state_version + 1,
            active_tool=tool_name,
        )

    def with_repair_attempt(self) -> RunSnapshot:
        """记录一次有限修复。用尽后转 REVIEW_REQUIRED。"""
        next_attempts = self.repair_attempts + 1
        if next_attempts > self.max_repair_attempts:
            return self.transition_to(REVIEW_REQUIRED)
        return replace(
            self,
            repair_attempts=next_attempts,
            state_version=self.state_version + 1,
        )

    def can_cancel_immediately(self) -> bool:
        """当前是否可以立即取消（增补 R-1）。

        正在跑写类工具时返回 False——必须等它结束并对账，再按 checkpoint
        决定回滚。强行中断会留下半应用的补丁。
        """
        if self.active_tool is None:
            return True
        return is_interruptible(self.active_tool)


@dataclass(frozen=True)
class EvidenceRef:
    """一条证据引用。路径与行号由应用映射，模型只能给编号。"""

    evidence_id: str
    path: str
    start_line: int
    end_line: int
    source_fingerprint: str
    retrieval_method: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip():
            raise ValueError("evidence_id 不能为空")
        if not self.path.strip():
            raise ValueError("path 不能为空")
        if self.start_line < 1:
            raise ValueError("行号为 1-based，start_line 必须 >= 1")
        if self.end_line < self.start_line:
            raise ValueError("end_line 不能小于 start_line")
        if not self.source_fingerprint.strip():
            raise ValueError("source_fingerprint 不能为空——否则无法判断证据是否已过期")


@dataclass(frozen=True)
class ChangePlan:
    """模型提出的修改计划。只是提议，不代表被批准。"""

    plan_id: str
    goal_id: str
    summary: str
    target_files: tuple[str, ...]
    evidence_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.plan_id.strip():
            raise ValueError("plan_id 不能为空")
        if not self.summary.strip():
            raise ValueError("计划摘要不能为空")
        if not self.target_files:
            raise ValueError("修改计划必须声明目标文件，否则范围门禁无从校验")


@dataclass(frozen=True)
class PatchArtifact:
    """一份生成的补丁。``diff_hash`` 用于绑定 approval。"""

    patch_id: str
    goal_id: str
    diff_text: str
    diff_hash: str
    base_fingerprint: str
    touched_files: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.patch_id.strip():
            raise ValueError("patch_id 不能为空")
        if not self.diff_text.strip():
            raise ValueError("补丁内容不能为空")
        if not self.diff_hash.strip():
            raise ValueError("diff_hash 不能为空——approval 需要绑定它")
        if not self.base_fingerprint.strip():
            raise ValueError("base_fingerprint 不能为空——基线变化时补丁必须失效")
        if not self.touched_files:
            raise ValueError("补丁必须至少涉及一个文件")


@dataclass(frozen=True)
class ChangeApproval:
    """一次性、会过期、范围绑定的审批令牌。

    四个绑定字段缺一不可：``run_id`` 防跨任务复用，``diff_hash`` 防批准 A
    却应用 B，``base_fingerprint`` 防基线漂移后仍生效，``scope`` 防越权。
    """

    token: str
    run_id: str
    diff_hash: str
    base_fingerprint: str
    scope: tuple[str, ...]
    expires_at_epoch_ms: int
    consumed_at_epoch_ms: int | None = None

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise ValueError("approval token 不能为空")
        if not self.run_id.strip():
            raise ValueError("approval 必须绑定 run_id")
        if not self.diff_hash.strip():
            raise ValueError("approval 必须绑定 diff_hash")
        if not self.base_fingerprint.strip():
            raise ValueError("approval 必须绑定 base_fingerprint")
        if not self.scope:
            raise ValueError("approval 必须声明生效范围")
        if self.expires_at_epoch_ms <= 0:
            raise ValueError("approval 必须有过期时间")

    @property
    def is_consumed(self) -> bool:
        return self.consumed_at_epoch_ms is not None

    def is_valid_for(
        self,
        *,
        run_id: str,
        diff_hash: str,
        base_fingerprint: str,
        now_epoch_ms: int,
    ) -> bool:
        """逐条校验绑定关系。任何一项不符即失效。"""
        if self.is_consumed:
            return False
        if now_epoch_ms >= self.expires_at_epoch_ms:
            return False
        if self.run_id != run_id:
            return False
        if self.diff_hash != diff_hash:
            return False
        if self.base_fingerprint != base_fingerprint:
            return False
        return True


@dataclass(frozen=True)
class WorkspaceCheckpoint:
    """隔离 workspace 的一个可回滚点。"""

    checkpoint_id: str
    run_id: str
    workspace_path: str
    tree_fingerprint: str
    created_at_epoch_ms: int

    def __post_init__(self) -> None:
        if not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id 不能为空")
        if not self.tree_fingerprint.strip():
            raise ValueError("tree_fingerprint 不能为空——回滚校验依赖它")


@dataclass(frozen=True)
class WorkspaceRun:
    """一次隔离 workspace 的生命周期记录。原仓库始终只读。"""

    workspace_id: str
    run_id: str
    source_repo_path: str
    workspace_path: str
    checkpoints: tuple[WorkspaceCheckpoint, ...] = ()

    def __post_init__(self) -> None:
        if not self.workspace_id.strip():
            raise ValueError("workspace_id 不能为空")
        if self.workspace_path == self.source_repo_path:
            raise ValueError(
                "workspace 路径不能等于源仓库路径——"
                "原仓库默认只读是从 1.0 继承的硬边界"
            )

    @property
    def latest_checkpoint(self) -> WorkspaceCheckpoint | None:
        if not self.checkpoints:
            return None
        return self.checkpoints[-1]


@dataclass(frozen=True)
class TestFailureDigest:
    """脱敏后的检查失败摘要，供有限修复使用。

    刻意不保存完整输出：测试日志可能含路径、环境变量与密钥。
    """

    digest_id: str
    run_id: str
    failed_test_ids: tuple[str, ...]
    error_class: str
    message_excerpt: str

    def __post_init__(self) -> None:
        if not self.failed_test_ids:
            raise ValueError("失败摘要必须列出至少一个失败项")
        if len(self.message_excerpt) > 2000:
            raise ValueError("摘要过长——应保存脱敏摘要而不是完整日志")


@dataclass(frozen=True)
class ValidationRun:
    """一次固定检查的执行结果。命令来自 allowlist，不由模型拼接。

    ``skipped`` 只用于显式的本地开发模式：它和检查通过/失败不同，表示
    这次没有执行命令，避免把“未校验”伪装成“通过”或“失败”。
    """

    validation_id: str
    run_id: str
    profile: str
    commands: tuple[str, ...]
    passed: bool
    digest: TestFailureDigest | None = None
    skipped: bool = False

    def __post_init__(self) -> None:
        if not self.commands and not self.skipped:
            raise ValueError("检查必须至少执行一条命令")
        if not self.passed and not self.skipped and self.digest is None:
            raise ValueError("检查失败必须附带 TestFailureDigest，否则无法有限修复")


@dataclass(frozen=True)
class RepositoryMap:
    """仓库导航地图。只用于导航与预算选择。

    **最终回答必须回到 Evidence 的真实路径和行号**——地图本身不是证据。
    """

    index_version: str
    repo_id: str
    symbols: tuple[str, ...] = ()
    imports: tuple[tuple[str, str], ...] = ()
    file_summaries: tuple[tuple[str, str], ...] = ()
    token_budget: int = 0
    repo_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.index_version.strip():
            raise ValueError("index_version 不能为空——地图必须能对应到具体索引")
        if self.token_budget < 0:
            raise ValueError("token_budget 不能为负")


@dataclass(frozen=True)
class GoalRouting:
    """新消息进来时的路由结论。

    默认继续当前 Goal，**不能默认新建**——否则「批准」会变成一个孤立的
    新任务，而原来那个暂停的任务永远悬着。
    """

    CONTINUE = "continue"
    AMEND = "amend"
    NEW = "new"

    decision: str
    goal_id: str | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.decision not in {self.CONTINUE, self.AMEND, self.NEW}:
            raise ValueError(f"不支持的路由决定：{self.decision}")
        if self.decision in {self.CONTINUE, self.AMEND} and not self.goal_id:
            raise ValueError(f"{self.decision} 必须指向一个已有 goal_id")
        if self.decision == self.NEW and self.goal_id:
            raise ValueError("新建 Goal 时不应携带既有 goal_id")


@dataclass(frozen=True)
class RunOutcome:
    """一次 Run 结束时对外公开的结果。不含模型隐藏推理。"""

    run_id: str
    status: str
    summary: str
    citations: tuple[EvidenceRef, ...] = ()
    diff_hash: str | None = None
    next_step: str | None = None
    events_recorded: int = 0
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in SUPPORTED_RUN_STATUSES:
            raise ValueError(f"不支持的 run status：{self.status}")
        if not self.summary.strip():
            raise ValueError("结果摘要不能为空")
