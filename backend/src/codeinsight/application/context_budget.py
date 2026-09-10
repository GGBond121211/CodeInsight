"""上下文 token 估算、预算裁剪和压缩结果。"""

from __future__ import annotations

from dataclasses import dataclass

from codeinsight.domain.memory import (
    CONTEXT_SECTION_PRIORITY,
    SECTION_OUTPUT_SCHEMA,
    SECTION_REPOSITORY_MAP,
    SECTION_SESSION_SUMMARY,
    SECTION_TEST_FAILURE_DIGEST,
    SECTION_TOOL_RESULTS,
    SECTION_TOOL_SCHEMA,
    CompactionResult,
    ContextBudget,
    ContextSection,
)
from codeinsight.domain.tokens import estimate_tokens as domain_estimate_tokens


def estimate_tokens(text: str) -> int:
    """用确定性的保守近似估算 token，不引入 tokenizer 依赖。

    实体实现已下沉到 domain.tokens，供 ingestion 的切块 token 上限共用。
    这里保留同名入口，避免既有调用方和登记表 S3-P03 的来源指针失效。
    """
    return domain_estimate_tokens(text)


@dataclass(frozen=True)
class BudgetEstimate:
    """交给后续 Gateway 做 reserve 的预算估算。"""

    max_tokens: int
    input_allowance: int
    reserved_output_tokens: int
    input_tokens: int
    total_tokens: int
    remaining_input_tokens: int

    @property
    def fits(self) -> bool:
        return self.input_tokens <= self.input_allowance


@dataclass(frozen=True)
class ContextCompactionPolicy:
    """上下文窗口接近上限时的确定性压缩策略。

    先压缩低优先级的辅助信息，再按领域优先级整体丢弃；安全规则、用户目标、
    代码任务、证据和当前 diff 从不进入这里。默认保留 64 token 的摘要窗口，
    因而不会为了「勉强塞进窗口」而把一整个历史分区压成不可读的碎片。
    """

    minimum_retained_tokens: int = 64
    candidate_sections: tuple[str, ...] = (
        SECTION_OUTPUT_SCHEMA,
        SECTION_REPOSITORY_MAP,
        SECTION_SESSION_SUMMARY,
        SECTION_TOOL_SCHEMA,
        SECTION_TOOL_RESULTS,
        SECTION_TEST_FAILURE_DIGEST,
    )
    marker: str = "\n[…中间内容已由确定性上下文压缩省略…]\n"

    def __post_init__(self) -> None:
        if self.minimum_retained_tokens < 8:
            raise ValueError("minimum_retained_tokens 不能小于 8")
        unknown = set(self.candidate_sections) - set(CONTEXT_SECTION_PRIORITY)
        if unknown:
            raise ValueError(f"压缩策略包含未登记分区：{sorted(unknown)}")
        if not self.marker.strip():
            raise ValueError("压缩标记不能为空")


@dataclass(frozen=True)
class FittedContext:
    """预算裁剪后的分区与可审查压缩记录。"""

    sections: tuple[ContextSection, ...]
    estimate: BudgetEstimate
    compaction: CompactionResult

    @property
    def text(self) -> str:
        blocks: list[str] = []
        for section in self.sections:
            blocks.append(f"[{section.name}]\n{section.content}")
        return "\n\n".join(blocks)


def make_section(name: str, content: str, *, is_untrusted: bool = False) -> ContextSection | None:
    """把非空文本变成带估算值的上下文分区。"""
    if not content.strip():
        return None
    return ContextSection(
        name=name,
        content=content,
        token_estimate=estimate_tokens(content),
        is_untrusted=is_untrusted,
    )


def estimate_budget(
    sections: tuple[ContextSection, ...],
    *,
    max_tokens: int,
    reserved_output_tokens: int,
) -> BudgetEstimate:
    """估算当前上下文，允许调用方在模型调用前决定是否需要裁剪。"""
    budget = ContextBudget(
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    input_tokens = 0
    for section in sections:
        input_tokens += section.token_estimate
    return BudgetEstimate(
        max_tokens=max_tokens,
        input_allowance=budget.input_allowance,
        reserved_output_tokens=reserved_output_tokens,
        input_tokens=input_tokens,
        total_tokens=input_tokens + reserved_output_tokens,
        remaining_input_tokens=budget.input_allowance - input_tokens,
    )


def fit_context(
    sections: tuple[ContextSection, ...],
    *,
    max_tokens: int,
    reserved_output_tokens: int,
    compaction_policy: ContextCompactionPolicy | None = None,
) -> FittedContext:
    """先压缩辅助历史，再按领域层定义的优先级裁剪。

    这是模型调用前的本地确定性步骤，不调用模型生成摘要，因此不会引入第二次
    成本或新的事实来源。被压缩和被整体丢弃的分区都会进入 ``CompactionResult``。
    """
    before = estimate_budget(
        sections,
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    policy = compaction_policy or ContextCompactionPolicy()
    compacted_sections, compacted_names = _compact_optional_sections(
        sections,
        input_allowance=before.input_allowance,
        policy=policy,
    )
    budget = ContextBudget(
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    kept = budget.fit(compacted_sections)
    kept_names: list[str] = []
    for section in kept:
        kept_names.append(section.name)
    dropped_names: list[str] = []
    for section in sections:
        if section.name not in kept_names:
            dropped_names.append(section.name)
    after = estimate_budget(
        kept,
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    compaction = CompactionResult(
        kept_section_names=tuple(kept_names),
        dropped_section_names=tuple(dropped_names),
        tokens_before=before.input_tokens,
        tokens_after=after.input_tokens,
        compacted_section_names=tuple(compacted_names),
        strategy=(
            "compact_then_drop_low_priority"
            if compacted_names
            else "drop_low_priority"
        ),
    )
    return FittedContext(sections=kept, estimate=after, compaction=compaction)


def _compact_optional_sections(
    sections: tuple[ContextSection, ...],
    *,
    input_allowance: int,
    policy: ContextCompactionPolicy,
) -> tuple[tuple[ContextSection, ...], tuple[str, ...]]:
    total = sum(section.token_estimate for section in sections)
    if total <= input_allowance:
        return sections, ()

    by_name = {section.name: section for section in sections}
    compacted_names: list[str] = []
    updated = dict(by_name)
    for name in policy.candidate_sections:
        if total <= input_allowance:
            break
        section = by_name.get(name)
        if section is None or section.token_estimate <= policy.minimum_retained_tokens:
            continue
        overflow = total - input_allowance
        target = max(policy.minimum_retained_tokens, section.token_estimate - overflow)
        if target >= section.token_estimate:
            continue
        compacted = _compact_section(section, target_tokens=target, marker=policy.marker)
        if compacted.token_estimate >= section.token_estimate:
            continue
        updated[name] = compacted
        total -= section.token_estimate - compacted.token_estimate
        compacted_names.append(name)

    result = tuple(updated.get(section.name, section) for section in sections)
    return result, tuple(compacted_names)


def _compact_section(
    section: ContextSection,
    *,
    target_tokens: int,
    marker: str,
) -> ContextSection:
    """保留头尾，避免工具错误摘要或 repository map 只剩单侧信息。"""
    if section.token_estimate <= target_tokens:
        return section
    marker_bytes = len(marker.encode("utf-8"))
    available_bytes = max(8, target_tokens * 4 - marker_bytes)
    head_bytes = available_bytes // 2
    tail_bytes = available_bytes - head_bytes
    raw = section.content.encode("utf-8")
    head = raw[:head_bytes].decode("utf-8", errors="ignore")
    tail = raw[-tail_bytes:].decode("utf-8", errors="ignore")
    content = f"{head}{marker}{tail}"
    # estimate_tokens 是保守近似；如果 UTF-8 边界让结果略超目标，再缩短一次。
    while estimate_tokens(content) > target_tokens and len(head) + len(tail) > 2:
        if len(head) >= len(tail):
            head = head[:-1]
        else:
            tail = tail[1:]
        content = f"{head}{marker}{tail}"
    return ContextSection(
        name=section.name,
        content=content,
        token_estimate=estimate_tokens(content),
        is_untrusted=section.is_untrusted,
    )
