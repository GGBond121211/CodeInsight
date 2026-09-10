"""代码理解的只读 MCP Tool Loop（Q-008），并接入 Evidence Repair（Q-009）。

迁移目标：让模型根据 ToolResult 自己选择下一步，而不是由 LangGraph 节点固定
explain → draft → review → revise。自主不等于没有约束，因此这里同时施加：

- 只读工具目录：修改类工具根本不进 discovery 结果；
- Evidence Ledger：模型只能引用应用分配的编号；
- 确定性评估：证据够不够由规则判定，不由模型自报；
- 有界 Repair：轮数、动作都不重复，预算耗尽即停。

代码修改路径保持独立，不经过这里。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event

from codeinsight.agent.tool_loop import (
    LoopIntervention,
    MCPToolClient,
    ToolCall,
    ToolLoop,
    ToolLoopConfig,
    ToolModel,
    ToolResult,
)
from codeinsight.application.evidence_assessor import (
    INVALID,
    SUFFICIENT,
    EvidenceAssessment,
    EvidenceAssessor,
)
from codeinsight.application.evidence_ledger import (
    NAVIGATION_TOOLS,
    EvidenceLedger,
    EvidenceRecord,
    LedgerIngestReport,
)
from codeinsight.application.evidence_repair import (
    EvidenceRepairController,
    RepairBudget,
)
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    AnswerCitation,
    ModelAnswer,
)
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.trace import (
    EVIDENCE_ASSESSED,
    EVIDENCE_REPAIR_FINISHED,
    EVIDENCE_REPAIR_STARTED,
    RunEvent,
)
from codeinsight.infrastructure.openai_chat import parse_model_answer
from codeinsight.prompts.code_understanding import (
    PROMPT_VERSION,
    build_code_understanding_prompt,
    evidence_index_message,
)

# 代码理解可以看到的只读工具。修改类工具不在这个集合里。
CODE_UNDERSTANDING_TOOLS: tuple[str, ...] = (
    "get_repository_map",
    "search_repository",
    "get_evidence_context",
    "read_file",
    "lsp_definition",
    "scip_references",
)

# 明确禁止出现在代码理解目录里的工具；出现即视为目录配置错误。
FORBIDDEN_TOOLS: tuple[str, ...] = (
    "generate_patch",
    "validate_patch",
    "apply_patch_isolated",
    "run_allowlisted_checks",
    "rollback_workspace",
    "get_diff",
)

# 终止状态。ABORTED 不在 Q-008 计划列举的集合里，但取消是真实状态，
# 单独保留比伪装成 FAILED 更诚实。
ANSWERED_STATUS = "ANSWERED"
PARTIAL_ANSWERED_STATUS = "PARTIALLY_ANSWERED"
INSUFFICIENT_STATUS = "INSUFFICIENT_EVIDENCE"
STUCK_STATUS = "STUCK"
TIMEOUT_STATUS = "TIMEOUT"
FAILED_STATUS = "FAILED"
ABORTED_STATUS = "ABORTED"

CODE_UNDERSTANDING_STATUSES: frozenset[str] = frozenset(
    {
        ANSWERED_STATUS,
        PARTIAL_ANSWERED_STATUS,
        INSUFFICIENT_STATUS,
        STUCK_STATUS,
        TIMEOUT_STATUS,
        FAILED_STATUS,
        ABORTED_STATUS,
    }
)


@dataclass(frozen=True)
class CodeUnderstandingConfig:
    """代码理解路线的显式预算；默认值是工程基线，不是质量最优值。"""

    loop: ToolLoopConfig = field(
        default_factory=lambda: ToolLoopConfig(
            max_steps=8,
            max_tool_calls=64,
            deadline_seconds=60.0,
            repeated_call_limit=2,
            repeated_error_limit=2,
        )
    )
    repair: RepairBudget = field(default_factory=RepairBudget)
    min_evidence: int = 2
    min_distinct_paths: int = 1
    max_evidence: int = 40
    # 模型声称 answered 但证据不足时，允许纠正它的次数。
    max_finalize_attempts: int = 1
    # 已告知证据足够之后，连续多少批「无新增证据」的检索算无效搜索。
    idle_batches_before_stop: int = 2

    def __post_init__(self) -> None:
        if self.min_evidence < 1:
            raise ValueError("min_evidence 必须是正整数")
        if self.max_evidence < 1:
            raise ValueError("max_evidence 必须是正整数")
        if self.max_finalize_attempts < 0:
            raise ValueError("max_finalize_attempts 不能为负")
        if self.idle_batches_before_stop < 1:
            raise ValueError("idle_batches_before_stop 必须是正整数")


@dataclass(frozen=True)
class CodeUnderstandingResult:
    status: str
    answer: str
    outcome: str
    citations: tuple[AnswerCitation, ...]
    evidence: tuple[EvidenceRecord, ...]
    assessment: EvidenceAssessment
    unknown_evidence_ids: tuple[str, ...]
    repair_rounds: int
    repair_actions: tuple[str, ...]
    navigation_seen: bool
    tool_calls: int
    steps: int
    input_tokens: int
    output_tokens: int
    loop_status: str
    termination_reason: str
    prompt_version: str = PROMPT_VERSION
    events: tuple[RunEvent, ...] = ()

    @property
    def ok(self) -> bool:
        return self.status in {ANSWERED_STATUS, PARTIAL_ANSWERED_STATUS, INSUFFICIENT_STATUS}

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "outcome": self.outcome,
            "answer": self.answer,
            "citations": [
                {
                    "evidence_id": item.evidence_id,
                    "relative_path": item.relative_path,
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                }
                for item in self.citations
            ],
            "evidence": [record.as_dict() for record in self.evidence],
            "assessment": self.assessment.as_dict(),
            "repair": {
                "rounds": self.repair_rounds,
                "actions": list(self.repair_actions),
            },
            "unknown_evidence_ids": list(self.unknown_evidence_ids),
            "navigation_seen": self.navigation_seen,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "loop_status": self.loop_status,
            "termination_reason": self.termination_reason,
            "prompt_version": self.prompt_version,
        }


def filter_readonly_tools(
    discovered: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    """只保留登记过的只读工具；写工具和未登记工具都被剔除。"""
    allowed: list[Mapping[str, object]] = []
    for item in discovered:
        name = item.get("name")
        if not isinstance(name, str) or name not in CODE_UNDERSTANDING_TOOLS:
            continue
        # readOnly 明确为 False 的条目一律不暴露，避免目录被改坏时静默放行。
        if item.get("readOnly") is False:
            continue
        if name in FORBIDDEN_TOOLS:
            continue
        allowed.append(item)
    return tuple(allowed)


class _ReadOnlyMCPView:
    """把 MCP Client 收窄成只读视图，循环拿不到写工具。"""

    def __init__(self, client: MCPToolClient) -> None:
        self._client = client

    def list_tools(self) -> tuple[Mapping[str, object], ...]:
        return filter_readonly_tools(self._client.list_tools())

    def call_tool(self, call: ToolCall) -> ToolResult:
        if call.name not in CODE_UNDERSTANDING_TOOLS:
            return ToolResult.failure(
                call.id,
                call.name,
                "PERMISSION",
                "代码理解路线只允许只读工具",
            )
        return self._client.call_tool(call)


class EvidenceGate:
    """夹在 ToolResult 与下一次模型调用之间的证据闸门。

    它同时承担 Q-008 的 Evidence Ledger 映射和 Q-009 的 Repair 放行判定。
    """

    def __init__(
        self,
        *,
        question: str,
        config: CodeUnderstandingConfig,
        repo_root: str | None = None,
        emit=None,
    ) -> None:
        self.question = question
        self.config = config
        self.repo_root = repo_root
        # 只发出状态/计数/动作，不发隐藏推理或工具原文。
        self._emit = emit
        self._pending_repair = None
        self.ledger = EvidenceLedger(max_evidence=config.max_evidence)
        self.assessor = EvidenceAssessor(
            min_evidence=config.min_evidence,
            min_distinct_paths=config.min_distinct_paths,
        )
        self.repair = EvidenceRepairController(budget=config.repair)
        self.assessment = EvidenceAssessment(status="insufficient")
        self.navigation_seen = False
        self.invalid_signals: list[str] = []
        self.decision_notes: list[str] = []
        self.idle_batches = 0
        self.sufficient_announced = False
        self.finalize_attempts_used = 0
        self.accepted: ModelAnswer | None = None
        self.unknown_evidence_ids: tuple[str, ...] = ()
        # 模型声称 answered 但纠正次数用尽时置位：保留答案文字，不承认已验证状态。
        self.downgraded = False

    @property
    def repair_round(self) -> int:
        return self.repair.rounds_used + 1

    def after_tools(
        self,
        *,
        step: int,
        calls: Sequence[ToolCall],
        results: Sequence[ToolResult],
    ) -> LoopIntervention | None:
        reports = self._ingest(calls, results)
        new_entries = _index_entries(reports)
        new_count = len(new_entries)
        duplicate_count = sum(report.duplicate_count for report in reports)
        if self._pending_repair is not None:
            self._publish(
                EVIDENCE_REPAIR_FINISHED,
                {
                    "round": str(self._pending_repair.round),
                    "repair_action": self._pending_repair.action,
                    "new_evidence_count": str(new_count),
                    "duplicate_evidence_count": str(duplicate_count),
                },
            )
            self._pending_repair = None
        self.assessment = self.assessor.assess(
            question=self.question,
            ledger=self.ledger,
            retrieval_round=self.repair.rounds_used,
            navigation_seen=self.navigation_seen,
            invalid_signals=tuple(dict.fromkeys(self.invalid_signals)),
        )
        self._publish(
            EVIDENCE_ASSESSED,
            {
                "step": str(step),
                "retrieval_round": str(self.repair.rounds_used),
                "status": self.assessment.status,
                "reasons": ",".join(self.assessment.reasons),
                "evidence_count": str(len(self.ledger)),
                "new_evidence_count": str(new_count),
            },
        )
        if not new_entries:
            self.idle_batches += 1
        else:
            self.idle_batches = 0
        if self.assessment.sufficient:
            self.sufficient_announced = True

        if self.assessment.status == INVALID:
            return LoopIntervention(
                terminate_reason=self.assessment.guidance or "证据位置非法",
                status="FAILED",
            )
        if self.sufficient_announced and not new_entries:
            if self.idle_batches >= self.config.idle_batches_before_stop:
                return LoopIntervention(
                    terminate_reason="证据已足够，但模型继续检索且没有新增证据",
                    status="STUCK",
                )

        parts: list[str] = []
        if new_entries:
            parts.append(evidence_index_message(new_entries))

        if self.assessment.status == SUFFICIENT:
            if new_entries:
                parts.append("[status] 证据已足够，请直接给出最终回答。")
        else:
            decision = self.repair.decide(
                assessment=self.assessment, repair_round=self.repair_round
            )
            if decision.allowed and decision.request is not None:
                self.repair.commit(decision.request)
                self.decision_notes.append(
                    f"round={decision.request.round}:{decision.request.action}"
                )
                self._pending_repair = decision.request
                self._publish(
                    EVIDENCE_REPAIR_STARTED,
                    {
                        "round": str(decision.request.round),
                        "repair_action": decision.request.action,
                        "reason": decision.request.reason,
                    },
                )
                parts.append(
                    f"[repair round {decision.request.round}] {decision.request.guidance}"
                )
        if not parts:
            return None
        return LoopIntervention(message="\n\n".join(parts))

    def _publish(self, event_type: str, payload: Mapping[str, str]) -> None:
        if self._emit is None:
            return
        self._emit(event_type, payload)

    def after_final_answer(
        self, *, step: int, content: str | None
    ) -> LoopIntervention | None:
        if content is None or not content.strip():
            return self._reject("模型返回了空内容")
        try:
            answer = parse_model_answer(
                content, model="", input_tokens=None, output_tokens=None
            )
        except ModelResponseError as error:
            return self._reject(f"最终回答不符合契约：{error}")
        unknown = self.ledger.unknown_ids(answer.evidence_ids)
        if unknown:
            self.unknown_evidence_ids = unknown
            return self._reject(
                f"引用了未分配的证据编号 {list(unknown)}；只能引用应用回填过的编号"
            )
        if answer.outcome == ANSWERED and not self.assessment.sufficient:
            if self.finalize_attempts_used < self.config.max_finalize_attempts:
                self.finalize_attempts_used += 1
                return LoopIntervention(
                    message=(
                        "[status] 证据尚未达到充分标准，不要声称 answered；"
                        "请继续检索，或改用 insufficient_evidence 并说明不足。"
                    )
                )
            # 纠正次数用尽：不丢弃模型文字，但把公开状态降级为「部分回答」，
            # 不让未经验证的结论以 answered 的身份交付。
            self.downgraded = True
        self.accepted = answer
        return None

    def _reject(self, reason: str) -> LoopIntervention:
        if self.finalize_attempts_used < self.config.max_finalize_attempts:
            self.finalize_attempts_used += 1
            return LoopIntervention(message=f"[status] {reason}")
        return LoopIntervention(terminate_reason=reason, status="STUCK")

    def _ingest(
        self, calls: Sequence[ToolCall], results: Sequence[ToolResult]
    ) -> tuple[LedgerIngestReport, ...]:
        reports: list[LedgerIngestReport] = []
        for call, result in zip(calls, results, strict=False):
            if call.name in NAVIGATION_TOOLS and result.ok:
                self.navigation_seen = True
            report = self.ledger.ingest(
                call=call, result=result, retrieval_round=self.repair.rounds_used
            )
            reports.append(report)
            for reason in report.skipped_reasons:
                if "越界" in reason or "行号" in reason:
                    self.invalid_signals.append(f"{call.name}:{reason}")
        return tuple(reports)

    def citations(self) -> tuple[AnswerCitation, ...]:
        if self.accepted is None:
            return ()
        citations: list[AnswerCitation] = []
        for evidence_id in self.accepted.evidence_ids:
            record = self.ledger.by_id(evidence_id)
            if record is None:
                continue
            citations.append(
                AnswerCitation(
                    evidence_id=record.evidence_id,
                    relative_path=record.path,
                    start_line=record.start_line,
                    end_line=record.end_line,
                )
            )
        return tuple(citations)


def _index_entries(
    reports: Sequence[LedgerIngestReport],
) -> tuple[tuple[str, str, int, int], ...]:
    entries: list[tuple[str, str, int, int]] = []
    for report in reports:
        for record in report.added:
            entries.append(
                (record.evidence_id, record.path, record.start_line, record.end_line)
            )
    return tuple(entries)


class CodeUnderstandingToolLoop:
    """运行一次只读的代码理解 Tool Loop。"""

    def __init__(
        self,
        model: ToolModel,
        mcp_client: MCPToolClient,
        *,
        config: CodeUnderstandingConfig | None = None,
        run_id: str = "local-run",
        event_log=None,
        repo_root: str | None = None,
        emit=None,
    ) -> None:
        self._model = model
        self._mcp = mcp_client
        self._config = config or CodeUnderstandingConfig()
        self._run_id = run_id
        self._event_log = event_log
        self._repo_root = repo_root
        self._emit = emit
        self._gate: EvidenceGate | None = None

    @property
    def gate(self) -> EvidenceGate | None:
        """最近一次运行的证据闸门；用于检查与复盘。"""
        return self._gate

    def run(
        self, question: str, *, cancel_event: Event | None = None
    ) -> CodeUnderstandingResult:
        system_prompt, user_prompt = build_code_understanding_prompt(question)
        gate = EvidenceGate(
            question=question,
            config=self._config,
            repo_root=self._repo_root,
            emit=self._emit,
        )
        self._gate = gate
        loop = ToolLoop(
            self._model,
            _ReadOnlyMCPView(self._mcp),
            config=self._config.loop,
            run_id=self._run_id,
            event_log=self._event_log,
            outcome_controller=gate,
        )
        loop_result = loop.run(system_prompt, user_prompt, cancel_event=cancel_event)
        return self._assemble(loop_result, gate)

    def _assemble(self, loop_result, gate: EvidenceGate) -> CodeUnderstandingResult:
        status, reason = _map_status(loop_result.status, loop_result.reason, gate)
        answer = gate.accepted
        return CodeUnderstandingResult(
            status=status,
            answer=answer.answer if answer is not None else "",
            outcome=answer.outcome if answer is not None else INSUFFICIENT_EVIDENCE,
            citations=gate.citations(),
            evidence=gate.ledger.records,
            assessment=gate.assessment,
            unknown_evidence_ids=gate.unknown_evidence_ids,
            repair_rounds=gate.repair.rounds_used,
            repair_actions=gate.repair.used_actions,
            navigation_seen=gate.navigation_seen,
            tool_calls=len(loop_result.tool_calls),
            steps=loop_result.steps,
            input_tokens=loop_result.input_tokens,
            output_tokens=loop_result.output_tokens,
            loop_status=loop_result.status,
            termination_reason=reason,
            events=loop_result.lifecycle_events,
        )


def _map_status(loop_status: str, loop_reason: str | None, gate: EvidenceGate) -> tuple[str, str]:
    """把循环状态映射成公开终止状态。"""
    if loop_status == "COMPLETED":
        answer = gate.accepted
        if answer is None:
            return FAILED_STATUS, "循环已完成但没有可接受的最终回答"
        if gate.downgraded:
            return (
                PARTIAL_ANSWERED_STATUS,
                "模型声称已回答，但确定性评估认为证据不充分",
            )
        if answer.outcome == ANSWERED and gate.assessment.sufficient:
            return ANSWERED_STATUS, ""
        return INSUFFICIENT_STATUS, gate.assessment.guidance or "证据不足"
    if loop_status == "ABORTED":
        return ABORTED_STATUS, loop_reason or "运行已取消"
    if loop_status == "FAILED":
        return FAILED_STATUS, loop_reason or "工具循环失败"
    if loop_status == "STUCK":
        reason = loop_reason or "工具循环达到上限"
        if "deadline" in reason:
            return TIMEOUT_STATUS, reason
        if gate.ledger.records:
            # 有证据但没能收尾：保留已有证据，明确标记为不完整。
            return STUCK_STATUS, reason
        return INSUFFICIENT_STATUS, reason
    return FAILED_STATUS, f"未支持的循环状态：{loop_status}"
