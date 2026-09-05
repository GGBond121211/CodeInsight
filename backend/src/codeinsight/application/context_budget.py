"""上下文 token 估算、预算裁剪和压缩结果。"""

from __future__ import annotations

from dataclasses import dataclass

from codeinsight.domain.memory import (
    CompactionResult,
    ContextBudget,
    ContextSection,
)


def estimate_tokens(text: str) -> int:
    """用确定性的保守近似估算 token，不引入 tokenizer 依赖。"""
    if not text:
        return 0
    encoded_length = len(text.encode("utf-8"))
    return max(1, (encoded_length + 3) // 4)


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
) -> FittedContext:
    """按领域层定义的优先级裁剪，并记录丢弃了哪些分区。"""
    before = estimate_budget(
        sections,
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    budget = ContextBudget(
        max_tokens=max_tokens,
        reserved_output_tokens=reserved_output_tokens,
    )
    kept = budget.fit(sections)
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
    )
    return FittedContext(sections=kept, estimate=after, compaction=compaction)
