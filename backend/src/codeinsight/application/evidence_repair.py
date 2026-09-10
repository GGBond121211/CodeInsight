"""Evidence Retrieval Repair 的有界控制器（Q-009）。

这一层夹在 ToolResult 和下一次模型调用之间：它决定「还能不能再检索一次」，
以及这一次只做哪一种修复动作。模型可以提出方向，但最终放行由这里判定。

它不是答案修订（LangGraph 的 Critic-Reviser），也不是补丁修复（Patch Repair）。
三者预算分开计数，不能用 Patch Repair 的次数代替 Evidence Repair 的次数。

不进入 Repair 的情况：安全/权限错误、路径越界、MCP 协议错误、写工具尝试、
预算耗尽、以及同一动作重复。
"""

from __future__ import annotations

from dataclasses import dataclass

from codeinsight.application.evidence_assessor import (
    CONTRADICTORY,
    CONTRADICTORY_EVIDENCE,
    INSUFFICIENT,
    INVALID,
    LOW_COVERAGE,
    MISSING_REQUIRED_EVIDENCE,
    NO_RESULTS,
    SUFFICIENT,
    EvidenceAssessment,
)

# 四种修复动作；每轮只允许一种，便于把结果归因到具体动作。
ACTION_QUERY_BROADEN = "A_query_broaden"
ACTION_WIDEN_SCOPE = "B_widen_scope"
ACTION_RELATION_TOOLS = "C_relation_tools"
ACTION_ADJACENT_CONTEXT = "D_adjacent_context"

REPAIR_ACTIONS: tuple[str, ...] = (
    ACTION_QUERY_BROADEN,
    ACTION_WIDEN_SCOPE,
    ACTION_RELATION_TOOLS,
    ACTION_ADJACENT_CONTEXT,
)

# 每种触发原因下允许尝试的动作，按优先级排列。
_ACTION_PRIORITY: dict[str, tuple[str, ...]] = {
    NO_RESULTS: (
        ACTION_QUERY_BROADEN,
        ACTION_RELATION_TOOLS,
        ACTION_WIDEN_SCOPE,
    ),
    MISSING_REQUIRED_EVIDENCE: (
        ACTION_QUERY_BROADEN,
        ACTION_RELATION_TOOLS,
    ),
    LOW_COVERAGE: (
        ACTION_WIDEN_SCOPE,
        ACTION_ADJACENT_CONTEXT,
    ),
    CONTRADICTORY_EVIDENCE: (
        ACTION_ADJACENT_CONTEXT,
        ACTION_QUERY_BROADEN,
    ),
}


@dataclass(frozen=True)
class RepairBudget:
    """Repair 自己的预算；与 Tool Loop 的步数/调用数/超时分别计数。"""

    max_repair_rounds: int = 1
    max_candidates_per_round: int = 20
    max_read_file_lines: int = 200

    def __post_init__(self) -> None:
        if self.max_repair_rounds < 0:
            raise ValueError("max_repair_rounds 不能为负")
        if self.max_candidates_per_round < 1:
            raise ValueError("max_candidates_per_round 必须是正整数")
        if self.max_read_file_lines < 1:
            raise ValueError("max_read_file_lines 必须是正整数")


@dataclass(frozen=True)
class RepairRequest:
    """一次被批准执行的修复动作。"""

    action: str
    round: int
    reason: str
    guidance: str
    anchors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.action not in REPAIR_ACTIONS:
            raise ValueError(f"未登记的 Repair action：{self.action}")
        if self.round < 1:
            raise ValueError("Repair round 从 1 开始")


@dataclass(frozen=True)
class RepairDecision:
    allowed: bool
    reason: str
    request: RepairRequest | None = None


class EvidenceRepairController:
    """决定是否放行一次证据修复，并保证动作不重复、预算不超支。"""

    def __init__(self, *, budget: RepairBudget | None = None) -> None:
        self.budget = budget or RepairBudget()
        self._used_actions: list[str] = []

    @property
    def rounds_used(self) -> int:
        return len(self._used_actions)

    @property
    def used_actions(self) -> tuple[str, ...]:
        return tuple(self._used_actions)

    def decide(
        self,
        *,
        assessment: EvidenceAssessment,
        repair_round: int,
    ) -> RepairDecision:
        """``repair_round`` 是即将开始的轮次序号，从 1 开始。"""
        if assessment.status == INVALID:
            return RepairDecision(False, "证据位置越界或畸形，属于安全边界，不进入 Repair")
        if assessment.status == SUFFICIENT:
            return RepairDecision(False, "证据已足够，不再发起检索")
        if assessment.status not in {INSUFFICIENT, CONTRADICTORY}:
            return RepairDecision(False, f"未支持的评估状态：{assessment.status}")
        if self.rounds_used >= self.budget.max_repair_rounds:
            return RepairDecision(
                False,
                f"Repair 轮数已用尽（上限 {self.budget.max_repair_rounds}）",
            )
        if repair_round != self.rounds_used + 1:
            return RepairDecision(False, "Repair round 序号与已用轮数不一致")

        for action in _candidate_actions(assessment.reasons):
            if action in self._used_actions:
                continue
            return RepairDecision(
                True,
                assessment.reasons[0] if assessment.reasons else "insufficient",
                RepairRequest(
                    action=action,
                    round=repair_round,
                    reason=assessment.reasons[0] if assessment.reasons else "insufficient",
                    guidance=_guidance(
                        action,
                        assessment=assessment,
                        budget=self.budget,
                    ),
                    anchors=assessment.missing_anchors,
                ),
            )
        return RepairDecision(False, "没有未使用过的 Repair 动作可用，停止重复检索")

    def commit(self, request: RepairRequest) -> None:
        """登记一次已放行的动作；重复登记同一动作会被拒绝。"""
        if request.action in self._used_actions:
            raise ValueError(f"Repair action 已经用过：{request.action}")
        self._used_actions.append(request.action)

    def as_dict(self) -> dict[str, object]:
        return {
            "rounds_used": self.rounds_used,
            "used_actions": list(self._used_actions),
            "max_repair_rounds": self.budget.max_repair_rounds,
        }


def _candidate_actions(reasons: tuple[str, ...]) -> tuple[str, ...]:
    """按评估原因汇总可用动作，保持首次出现的顺序。"""
    ordered: list[str] = []
    for reason in reasons:
        for action in _ACTION_PRIORITY.get(reason, ()):
            if action not in ordered:
                ordered.append(action)
    return tuple(ordered)


def _guidance(
    action: str,
    *,
    assessment: EvidenceAssessment,
    budget: RepairBudget,
) -> str:
    if action == ACTION_QUERY_BROADEN:
        anchors = list(assessment.missing_anchors)
        detail = f"把 {anchors} 作为额外关键词" if anchors else "补充更具体的代码关键词"
        return (
            f"请保留原始问题，再用 search_repository 重新检索一次：{detail}。"
            "不要只使用改写后的问句，改写错误会替换掉原本正确的召回。"
            f"本轮候选取前 {budget.max_candidates_per_round} 条。"
        )
    if action == ACTION_WIDEN_SCOPE:
        return (
            "请扩大已验证的检索范围：提高 search_repository 的 limit，"
            "或对同一问题补充关联子问题，再合并去重。"
            "扩大范围仍受 max_evidence 和上下文预算限制。"
        )
    if action == ACTION_RELATION_TOOLS:
        return (
            "请改用关系工具定位：先用 get_repository_map 找符号所在文件，"
            "必要时用 lsp_definition 或 scip_references 定位定义与引用。"
            "拿到位置后必须回到 read_file 形成证据——导航线索本身不算证据。"
        )
    if action == ACTION_ADJACENT_CONTEXT:
        return (
            "请用 read_file 读取目标文件的相关行区间，确认当前内容。"
            f"单次读取不超过 {budget.max_read_file_lines} 行，且不得越过仓库范围。"
        )
    return "请补充检索证据后再作答。"
