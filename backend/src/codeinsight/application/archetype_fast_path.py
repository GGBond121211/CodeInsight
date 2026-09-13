"""问题原型快路径（Q-012 U2）。

命中原型后按固定配方取证，然后只调一次模型写措辞。它和 Tool Loop 的关系是
「更省的同一件事」，不是「另一条答案通道」：

  - 同一份只读工具视图（写工具在这里同样不可见）；
  - 同一个 Evidence Ledger，模型只能引用应用分配的 E 编号；
  - 同一套 {outcome, answer, citations} 契约；
  - 证据不够就返回 None，调用方回退到 Tool Loop——快路径不许「硬答」。

默认关闭：CODEINSIGHT_ARCHETYPE_FAST_PATH 不是 on/true/1 时不启用。错命中的
代价是「看起来正确但答非所问」，所以在 C2 的分阶段指标出来之前不默认打开。
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence

from codeinsight.agent.code_understanding_tool_loop import (
    ANSWERED_STATUS,
    INSUFFICIENT_STATUS,
    CodeUnderstandingResult,
    _ReadOnlyMCPView,
)
from codeinsight.agent.tool_loop import (
    MCPToolClient,
    ToolCall,
    ToolModel,
    ToolResult,
)
from codeinsight.application.evidence_assessor import EvidenceAssessor
from codeinsight.application.evidence_ledger import EvidenceLedger, EvidenceRecord
from codeinsight.application.structured_evidence import (
    build_structured_evidence,
    first_location,
)
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    AnswerCitation,
)
from codeinsight.domain.archetype import (
    LINE_PLACEHOLDER,
    PATH_PLACEHOLDER,
    QUESTION_PLACEHOLDER,
    QuestionArchetype,
)
from codeinsight.domain.errors import ModelResponseError
from codeinsight.domain.trace import ARCHETYPE_MATCHED, RunEvent
from codeinsight.infrastructure.openai_chat import parse_model_answer
from codeinsight.prompts.code_understanding import (
    build_code_understanding_prompt,
    evidence_context_message,
)

FAST_PATH_ENV = "CODEINSIGHT_ARCHETYPE_FAST_PATH"
FAST_PATH_MODEL_LABEL = "archetype-fast-path"


def fast_path_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """默认关闭；只有显式写成 on/true/1/yes 才启用。"""

    source = os.environ if environ is None else environ
    raw = (source.get(FAST_PATH_ENV) or "").strip().lower()
    return raw in {"1", "true", "on", "yes"}


def _resolve_arguments(
    step_arguments: Mapping[str, object],
    *,
    question: str,
    locations: Sequence[tuple[str, int | None, int | None]],
) -> dict[str, object] | None:
    """把占位符换成真实值；缺少上一步位置时返回 None（不猜）。"""

    fallback = locations[0] if locations else None
    resolved: dict[str, object] = {}
    for key, value in step_arguments.items():
        if not isinstance(value, str) or not value.startswith("{"):
            resolved[key] = value
            continue
        if value == QUESTION_PLACEHOLDER:
            resolved[key] = question
            continue
        if fallback is None:
            return None
        path, line, column = fallback
        if value == PATH_PLACEHOLDER:
            resolved[key] = path
        elif value == LINE_PLACEHOLDER:
            if line is None:
                return None
            resolved[key] = line
        else:
            resolved[key] = column if column is not None else 0
    return resolved


def run_archetype_fast_path(
    *,
    archetype: QuestionArchetype,
    question: str,
    mcp_client: MCPToolClient,
    model: ToolModel,
    run_id: str = "local-run",
    event_log=None,
    max_evidence: int = 40,
    min_evidence: int = 2,
    min_distinct_paths: int = 1,
    emit: Callable[[str, Mapping[str, str]], None] | None = None,
) -> CodeUnderstandingResult | None:
    """按配方取证并生成一次答案；证据不足返回 None 交给 Tool Loop。"""

    if not archetype.recipe:
        raise ValueError("配方不能为空")
    ledger = EvidenceLedger(max_evidence=max_evidence)
    view = _ReadOnlyMCPView(mcp_client)
    locations: list[tuple[str, int | None, int | None]] = []
    new_records: list[EvidenceRecord] = []
    tool_results: list[ToolResult] = []
    navigation_seen = False

    for index, step in enumerate(archetype.recipe):
        arguments = _resolve_arguments(
            step.arguments, question=question, locations=locations
        )
        if arguments is None:
            _report(emit, event_log, run_id, archetype, "missing_location", False)
            return None
        call = ToolCall(f"recipe-{index}", step.tool, arguments)
        result = view.call_tool(call)
        if not result.ok:
            reason = f"tool_error:{result.error_code}"
            _report(emit, event_log, run_id, archetype, reason, False)
            return None
        report = ledger.ingest(call=call, result=result, retrieval_round=0)
        tool_results.append(result)
        for record in report.added:
            new_records.append(record)
        if not report.added and not report.duplicates:
            # 导航工具不产出 Evidence；记下「看到过导航」以免判定过严。
            navigation_seen = True
        location = first_location(result)
        if location is not None:
            locations.append((location.path, location.line, location.column))

    assessment = EvidenceAssessor(
        min_evidence=min_evidence, min_distinct_paths=min_distinct_paths
    ).assess(question=question, ledger=ledger, navigation_seen=navigation_seen)
    if not assessment.sufficient:
        _report(emit, event_log, run_id, archetype, "insufficient_evidence", False)
        return None

    entries = tuple(
        (
            record.evidence_id,
            record.path,
            record.start_line,
            record.end_line,
            record.excerpt,
        )
        for record in new_records
    )
    system_prompt, user_prompt = build_code_understanding_prompt(question)
    messages: list[dict[str, object]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    if entries:
        messages.append({"role": "user", "content": evidence_context_message(entries)})
    try:
        response = model.complete_with_tools(tuple(messages), ())
    except Exception as error:  # noqa: BLE001 - 快路径不承担「模型挂了」的兜底
        # 带上异常类型：只写 model_failed 分不清是网络、限流还是响应格式，运维时要重跑一次才知道。
        _report(
            emit, event_log, run_id, archetype, f"model_failed:{type(error).__name__}", False
        )
        return None
    if not response.content or response.tool_calls:
        # 快路径只做一次措辞调用；它还想调工具就说明配方没喂饱它。
        _report(emit, event_log, run_id, archetype, "wanted_tools", False)
        return None
    try:
        answer = parse_model_answer(
            response.content,
            model=FAST_PATH_MODEL_LABEL,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
        )
    except ModelResponseError:
        # 答案不是合法的 {outcome, answer, citations} JSON：与「证据不够」同一种处理，
        # 不硬答、不把解析异常抛给调用方——回退 Tool Loop 是这条路线唯一的兜底。
        _report(emit, event_log, run_id, archetype, "invalid_answer", False)
        return None
    records = {record.evidence_id: record for record in ledger.records}
    unknown = [item for item in answer.evidence_ids if item not in records]
    if unknown:
        # 引用了解算不出的编号：与 Tool Loop 的同一条硬门槛。
        _report(emit, event_log, run_id, archetype, "unknown_evidence_id", False)
        return None
    citations = tuple(
        AnswerCitation(
            evidence_id=item,
            relative_path=records[item].path,
            start_line=records[item].start_line,
            end_line=records[item].end_line,
        )
        for item in answer.evidence_ids
    )
    _report(emit, event_log, run_id, archetype, "hit", True)
    rows, truncated = build_structured_evidence(
        tool_results=tool_results,
        evidence=ledger.records,
    )
    sufficient = answer.outcome == ANSWERED and assessment.sufficient
    return CodeUnderstandingResult(
        status=ANSWERED_STATUS if sufficient else INSUFFICIENT_STATUS,
        answer=answer.answer,
        outcome=answer.outcome if sufficient else INSUFFICIENT_EVIDENCE,
        citations=citations,
        evidence=ledger.records,
        assessment=assessment,
        unknown_evidence_ids=(),
        repair_rounds=0,
        repair_actions=(),
        navigation_seen=navigation_seen,
        tool_calls=len(archetype.recipe),
        steps=len(archetype.recipe),
        input_tokens=response.input_tokens or 0,
        output_tokens=response.output_tokens or 0,
        loop_status="ARCHETYPE_FAST_PATH",
        termination_reason="" if sufficient else "快路径证据未获确定性评估通过",
        structured_evidence=rows,
        structured_evidence_truncated=truncated,
        archetype=archetype.name,
        fast_path=True,
    )


def _report(
    emit: Callable[[str, Mapping[str, str]], None] | None,
    event_log,
    run_id: str,
    archetype: QuestionArchetype,
    reason: str,
    matched: bool,
) -> None:
    """只发状态与原型的名字；没有 emit 也没有日志时静默。"""

    payload = {
        "archetype": archetype.name,
        "archetype_version": archetype.version,
        "reason": reason,
        "fast_path": "true" if matched else "false",
    }
    if emit is not None:
        emit(ARCHETYPE_MATCHED, payload)
        return
    if event_log is None:
        return
    sequence = event_log.next_sequence(run_id)
    event_log.append(
        RunEvent(
            event_id=f"{run_id}:{sequence}",
            run_id=run_id,
            sequence=sequence,
            event_type=ARCHETYPE_MATCHED,
            occurred_at_epoch_ms=int(time.time() * 1000),
            payload=payload,
        )
    )
