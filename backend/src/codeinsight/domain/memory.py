"""三层记忆与上下文预算的领域模型。

为什么要分三层（来自 docs/PLAN_2.0.md Step 2）：

    Working Memory   当前 Run 的工作区：目标、已选证据、已试路径、剩余步数。
                     随 Run 结束而丢弃。
    Session Memory   跨 Run 的对话状态：用户偏好、已确认结论、被否决的方案。
                     随 Session 存活。
    Semantic Memory  仓库长期事实：RepositoryMap、符号索引、模块摘要。
                     随仓库版本存活，与用户无关。

    不分层的后果是上下文无限增长。把整段对话历史全塞进 Prompt，第 20 轮时
    要么超窗，要么把真正相关的证据挤出去——**而后者更危险，因为它不报错**。

一条硬规则：**Semantic Memory 里的内容不是证据**。RepositoryMap 只用于
导航与预算选择，最终回答必须回到 Evidence 的真实路径与行号。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

LAYER_WORKING = "working"
LAYER_SESSION = "session"
LAYER_SEMANTIC = "semantic"

SUPPORTED_MEMORY_LAYERS: frozenset[str] = frozenset(
    {LAYER_WORKING, LAYER_SESSION, LAYER_SEMANTIC}
)

# 各层的生命周期归属。写死在这里是为了让「这条记忆什么时候该消失」
# 有唯一答案，而不是散落在各处的清理代码里。
MEMORY_LIFETIME_OWNER: dict[str, str] = {
    LAYER_WORKING: "run",
    LAYER_SESSION: "session",
    LAYER_SEMANTIC: "repo_index_version",
}

# 上下文分区。顺序即优先级：预算不足时从后往前裁。
SECTION_SYSTEM = "system"
SECTION_GOAL = "goal"
SECTION_EVIDENCE = "evidence"
SECTION_TOOL_RESULTS = "tool_results"
SECTION_SESSION_SUMMARY = "session_summary"
SECTION_REPOSITORY_MAP = "repository_map"

CONTEXT_SECTION_PRIORITY: tuple[str, ...] = (
    SECTION_SYSTEM,
    SECTION_GOAL,
    SECTION_EVIDENCE,
    SECTION_TOOL_RESULTS,
    SECTION_SESSION_SUMMARY,
    SECTION_REPOSITORY_MAP,
)

# 永不裁剪的分区。裁掉 system 会丢掉安全规则，裁掉 goal 会让模型忘记在做什么。
NEVER_TRIMMED_SECTIONS: frozenset[str] = frozenset({SECTION_SYSTEM, SECTION_GOAL})


class ContextBudgetExceededError(Exception):
    """即使裁到只剩不可裁剪分区仍然超预算。

    这时正确做法是缩小任务范围或减少证据条数，而不是悄悄截断 system 分区。
    """


@dataclass(frozen=True)
class MemoryRecord:
    """一条记忆。``layer`` 决定它何时被清理。"""

    record_id: str
    layer: str
    owner_id: str
    content: str
    token_estimate: int
    is_trusted: bool = True

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id 不能为空")
        if self.layer not in SUPPORTED_MEMORY_LAYERS:
            raise ValueError(f"不支持的记忆层：{self.layer}")
        if not self.owner_id.strip():
            raise ValueError("owner_id 不能为空——否则不知道这条记忆归谁、何时清理")
        if not self.content.strip():
            raise ValueError("记忆内容不能为空")
        if self.token_estimate < 0:
            raise ValueError("token_estimate 不能为负")

    @property
    def lifetime_owner(self) -> str:
        return MEMORY_LIFETIME_OWNER[self.layer]


@dataclass(frozen=True)
class WorkingMemory:
    """当前 Run 的工作区。随 Run 结束丢弃。

    ``rejected_paths`` 是防打转的关键：记下已经试过且失败的方向，
    否则 Agent 会在同一个死胡同里反复消耗步数。
    """

    run_id: str
    user_goal: str
    selected_evidence_ids: tuple[str, ...] = ()
    rejected_paths: tuple[str, ...] = ()
    steps_remaining: int = 0

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("run_id 不能为空")
        if not self.user_goal.strip():
            raise ValueError("user_goal 不能为空")
        if self.steps_remaining < 0:
            raise ValueError("steps_remaining 不能为负")

    def with_rejected_path(self, description: str) -> WorkingMemory:
        """记录一条已排除的路径。重复的不再追加。"""
        if description in self.rejected_paths:
            return self
        return replace(self, rejected_paths=self.rejected_paths + (description,))

    def with_evidence(self, evidence_id: str) -> WorkingMemory:
        if evidence_id in self.selected_evidence_ids:
            return self
        return replace(self, selected_evidence_ids=self.selected_evidence_ids + (evidence_id,))


@dataclass(frozen=True)
class SessionMemory:
    """跨 Run 的对话状态。

    ``rejected_approaches`` 存在的理由和 WorkingMemory 的 rejected_paths 一样，
    但作用在更长的时间尺度：用户明确否决过的方案，下一次 Run 不该再提。
    """

    session_id: str
    confirmed_conclusions: tuple[str, ...] = ()
    rejected_approaches: tuple[str, ...] = ()
    user_preferences: dict[str, str] = field(default_factory=dict)
    summary: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")


@dataclass(frozen=True)
class SemanticMemory:
    """仓库长期事实。按索引版本失效，与具体用户无关。

    ``index_version`` 变了就必须整体失效——用旧地图导航新代码，会把模型
    引向已经不存在的符号。
    """

    repo_id: str
    index_version: str
    module_summaries: tuple[tuple[str, str], ...] = ()
    symbol_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.repo_id.strip():
            raise ValueError("repo_id 不能为空")
        if not self.index_version.strip():
            raise ValueError("index_version 不能为空——记忆失效判断依赖它")

    def is_stale_against(self, current_index_version: str) -> bool:
        return self.index_version != current_index_version


@dataclass(frozen=True)
class ContextSection:
    """Prompt 中的一个分区。

    ``is_untrusted`` 对应增补 R-2：被分析仓库的全部内容都是不可信数据，
    必须带标记进入 Evidence 分区，且不得改变工具选择与权限范围。
    """

    name: str
    content: str
    token_estimate: int
    is_untrusted: bool = False

    def __post_init__(self) -> None:
        if self.name not in CONTEXT_SECTION_PRIORITY:
            raise ValueError(
                f"未登记的上下文分区：{self.name}。"
                f"必须是 {list(CONTEXT_SECTION_PRIORITY)} 之一，"
                "否则裁剪顺序无从确定。"
            )
        if self.token_estimate < 0:
            raise ValueError("token_estimate 不能为负")
        if self.name == SECTION_EVIDENCE and not self.is_untrusted:
            raise ValueError(
                "Evidence 分区必须标记为不可信。"
                "被分析仓库的源码、注释、README、文件名全部是不可信数据（增补 R-2）。"
            )
        if self.name == SECTION_SYSTEM and self.is_untrusted:
            raise ValueError("system 分区不能来自不可信来源")


@dataclass(frozen=True)
class ContextBudget:
    """一次模型调用的上下文预算。

    裁剪按 ``CONTEXT_SECTION_PRIORITY`` 从后往前，system 与 goal 永不裁剪。
    从后往前的理由：RepositoryMap 只是导航辅助，丢了还能靠检索补；
    Evidence 丢了回答就没有依据。
    """

    max_tokens: int
    reserved_output_tokens: int = 0

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens 必须为正")
        if self.reserved_output_tokens < 0:
            raise ValueError("reserved_output_tokens 不能为负")
        if self.reserved_output_tokens >= self.max_tokens:
            raise ValueError("为输出预留的额度不能占满整个预算")

    @property
    def input_allowance(self) -> int:
        return self.max_tokens - self.reserved_output_tokens

    def fit(self, sections: tuple[ContextSection, ...]) -> tuple[ContextSection, ...]:
        """按优先级裁剪到预算内，返回保留下来的分区。

        裁到只剩不可裁剪分区仍超预算时抛 ContextBudgetExceededError——
        这时静默截断 system 分区会丢掉安全规则，必须让调用方知道。
        """
        by_name: dict[str, ContextSection] = {}
        for section in sections:
            by_name[section.name] = section

        kept: list[ContextSection] = []
        for name in CONTEXT_SECTION_PRIORITY:
            section = by_name.get(name)
            if section is not None:
                kept.append(section)

        total = 0
        for section in kept:
            total += section.token_estimate

        while total > self.input_allowance:
            victim_index = -1
            for index in range(len(kept) - 1, -1, -1):
                if kept[index].name not in NEVER_TRIMMED_SECTIONS:
                    victim_index = index
                    break
            if victim_index < 0:
                raise ContextBudgetExceededError(
                    f"裁剪到只剩 {sorted(NEVER_TRIMMED_SECTIONS)} 后仍需 {total} token，"
                    f"超过输入额度 {self.input_allowance}。"
                    "请缩小任务范围或减少证据条数，不要截断 system 分区。"
                )
            total -= kept[victim_index].token_estimate
            del kept[victim_index]

        result: list[ContextSection] = []
        for section in kept:
            result.append(section)
        return tuple(result)


@dataclass(frozen=True)
class CompactionResult:
    """一次上下文压缩的结果。

    刻意保留 ``dropped_section_names``：用户问「为什么这次没引用那个文件」时，
    答案往往是「它在压缩时被裁掉了」。不记录就无从解释。
    """

    kept_section_names: tuple[str, ...]
    dropped_section_names: tuple[str, ...]
    tokens_before: int
    tokens_after: int

    def __post_init__(self) -> None:
        if self.tokens_after > self.tokens_before:
            raise ValueError("压缩后的 token 数不应大于压缩前")
        for name in NEVER_TRIMMED_SECTIONS:
            if name in self.dropped_section_names:
                raise ValueError(f"{name} 分区不允许被裁剪")
