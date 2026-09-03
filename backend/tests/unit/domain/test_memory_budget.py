"""三层记忆与上下文预算的回归测试。

最重要的一条：**预算不足时宁可报错，也不能截断 system 分区**。
截断 system 会丢掉安全规则，而且不报错——这是最坏的失败方式。
"""

from __future__ import annotations

import pytest

from codeinsight.domain.memory import (
    CONTEXT_SECTION_PRIORITY,
    LAYER_SEMANTIC,
    LAYER_SESSION,
    LAYER_WORKING,
    MEMORY_LIFETIME_OWNER,
    NEVER_TRIMMED_SECTIONS,
    SECTION_EVIDENCE,
    SECTION_GOAL,
    SECTION_REPOSITORY_MAP,
    SECTION_SESSION_SUMMARY,
    SECTION_SYSTEM,
    SECTION_TOOL_RESULTS,
    SUPPORTED_MEMORY_LAYERS,
    CompactionResult,
    ContextBudget,
    ContextBudgetExceededError,
    ContextSection,
    MemoryRecord,
    SemanticMemory,
    SessionMemory,
    WorkingMemory,
)


def _section(name: str, tokens: int) -> ContextSection:
    return ContextSection(
        name=name,
        content=f"{name} 的内容",
        token_estimate=tokens,
        is_untrusted=(name == SECTION_EVIDENCE),
    )


# ---------------------------------------------------------------------------
# 三层记忆
# ---------------------------------------------------------------------------


def test_every_layer_has_a_lifetime_owner() -> None:
    """漏掉一层会让「这条记忆什么时候该消失」没有答案。"""
    assert set(MEMORY_LIFETIME_OWNER) == set(SUPPORTED_MEMORY_LAYERS)


@pytest.mark.parametrize(
    ("layer", "owner"),
    [
        (LAYER_WORKING, "run"),
        (LAYER_SESSION, "session"),
        (LAYER_SEMANTIC, "repo_index_version"),
    ],
)
def test_layer_lifetime_owners(layer: str, owner: str) -> None:
    record = MemoryRecord(
        record_id="m-1",
        layer=layer,
        owner_id="owner-1",
        content="内容",
        token_estimate=10,
    )
    assert record.lifetime_owner == owner


def test_unknown_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="记忆层"):
        MemoryRecord(
            record_id="m-1",
            layer="long_term",
            owner_id="o",
            content="c",
            token_estimate=1,
        )


def test_memory_requires_owner_id() -> None:
    """没有 owner 就不知道这条记忆归谁、何时清理。"""
    with pytest.raises(ValueError, match="owner_id"):
        MemoryRecord(
            record_id="m-1",
            layer=LAYER_WORKING,
            owner_id="  ",
            content="c",
            token_estimate=1,
        )


# ---------------------------------------------------------------------------
# Working Memory 防打转
# ---------------------------------------------------------------------------


def test_rejected_paths_prevent_repeating_dead_ends() -> None:
    """记下试过且失败的方向，否则 Agent 会在死胡同里反复消耗步数。"""
    memory = WorkingMemory(run_id="run-1", user_goal="改重试逻辑", steps_remaining=8)
    memory = memory.with_rejected_path("在 config.py 里找配置项——不存在")
    memory = memory.with_rejected_path("在 config.py 里找配置项——不存在")
    assert len(memory.rejected_paths) == 1


def test_evidence_ids_are_deduplicated() -> None:
    memory = WorkingMemory(run_id="run-1", user_goal="目标")
    memory = memory.with_evidence("E1").with_evidence("E1").with_evidence("E2")
    assert memory.selected_evidence_ids == ("E1", "E2")


def test_working_memory_is_immutable() -> None:
    original = WorkingMemory(run_id="run-1", user_goal="目标")
    updated = original.with_evidence("E1")
    assert original.selected_evidence_ids == ()
    assert updated.selected_evidence_ids == ("E1",)


def test_session_memory_defaults_are_empty() -> None:
    memory = SessionMemory(session_id="sess-1")
    assert memory.confirmed_conclusions == ()
    assert memory.rejected_approaches == ()


# ---------------------------------------------------------------------------
# Semantic Memory 按索引版本失效
# ---------------------------------------------------------------------------


def test_semantic_memory_is_stale_when_index_version_changes() -> None:
    """用旧地图导航新代码，会把模型引向已经不存在的符号。"""
    memory = SemanticMemory(repo_id="repo-1", index_version="v1")
    assert memory.is_stale_against("v2") is True
    assert memory.is_stale_against("v1") is False


def test_semantic_memory_requires_index_version() -> None:
    with pytest.raises(ValueError, match="index_version"):
        SemanticMemory(repo_id="repo-1", index_version="  ")


# ---------------------------------------------------------------------------
# 上下文分区
# ---------------------------------------------------------------------------


def test_unknown_section_is_refused() -> None:
    """未登记的分区没有裁剪优先级，无从决定何时裁它。"""
    with pytest.raises(ValueError, match="上下文分区"):
        ContextSection(name="extra_notes", content="c", token_estimate=1)


def test_evidence_section_must_be_marked_untrusted() -> None:
    """被分析仓库的源码、注释、README、文件名全部是不可信数据（增补 R-2）。"""
    with pytest.raises(ValueError, match="不可信"):
        ContextSection(
            name=SECTION_EVIDENCE,
            content="仓库里的代码",
            token_estimate=10,
            is_untrusted=False,
        )


def test_system_section_cannot_be_untrusted() -> None:
    with pytest.raises(ValueError, match="不可信来源"):
        ContextSection(
            name=SECTION_SYSTEM,
            content="规则",
            token_estimate=10,
            is_untrusted=True,
        )


def test_never_trimmed_sections_are_in_the_priority_list() -> None:
    for name in NEVER_TRIMMED_SECTIONS:
        assert name in CONTEXT_SECTION_PRIORITY


def test_priority_order_puts_system_and_goal_first() -> None:
    """裁剪从后往前，所以不可裁剪的必须在最前面。"""
    assert CONTEXT_SECTION_PRIORITY[0] == SECTION_SYSTEM
    assert CONTEXT_SECTION_PRIORITY[1] == SECTION_GOAL


# ---------------------------------------------------------------------------
# 预算裁剪
# ---------------------------------------------------------------------------


def test_everything_fits_when_budget_is_ample() -> None:
    budget = ContextBudget(max_tokens=1_000, reserved_output_tokens=200)
    sections = (
        _section(SECTION_SYSTEM, 50),
        _section(SECTION_GOAL, 50),
        _section(SECTION_EVIDENCE, 100),
        _section(SECTION_REPOSITORY_MAP, 100),
    )
    kept = budget.fit(sections)
    assert len(kept) == 4


def test_output_reservation_is_excluded_from_input_allowance() -> None:
    budget = ContextBudget(max_tokens=1_000, reserved_output_tokens=400)
    assert budget.input_allowance == 600


def test_lowest_priority_section_is_dropped_first() -> None:
    """RepositoryMap 只是导航辅助，丢了还能靠检索补；Evidence 丢了回答就没依据。"""
    budget = ContextBudget(max_tokens=200)
    sections = (
        _section(SECTION_SYSTEM, 50),
        _section(SECTION_GOAL, 50),
        _section(SECTION_EVIDENCE, 80),
        _section(SECTION_REPOSITORY_MAP, 100),
    )
    kept = budget.fit(sections)
    names: list[str] = []
    for section in kept:
        names.append(section.name)
    assert SECTION_REPOSITORY_MAP not in names
    assert SECTION_EVIDENCE in names


def test_sections_are_dropped_from_the_back_in_order() -> None:
    budget = ContextBudget(max_tokens=120)
    sections = (
        _section(SECTION_SYSTEM, 40),
        _section(SECTION_GOAL, 40),
        _section(SECTION_EVIDENCE, 30),
        _section(SECTION_TOOL_RESULTS, 30),
        _section(SECTION_SESSION_SUMMARY, 30),
        _section(SECTION_REPOSITORY_MAP, 30),
    )
    kept = budget.fit(sections)
    names: list[str] = []
    for section in kept:
        names.append(section.name)
    assert names == [SECTION_SYSTEM, SECTION_GOAL, SECTION_EVIDENCE]


def test_output_is_always_in_priority_order() -> None:
    """输入顺序打乱也要按优先级输出，否则裁剪结果不确定。"""
    budget = ContextBudget(max_tokens=1_000)
    sections = (
        _section(SECTION_REPOSITORY_MAP, 10),
        _section(SECTION_GOAL, 10),
        _section(SECTION_SYSTEM, 10),
    )
    kept = budget.fit(sections)
    names: list[str] = []
    for section in kept:
        names.append(section.name)
    assert names == [SECTION_SYSTEM, SECTION_GOAL, SECTION_REPOSITORY_MAP]


def test_over_budget_after_full_trim_raises() -> None:
    """裁到只剩 system + goal 仍超预算时必须报错，不能悄悄截断 system。"""
    budget = ContextBudget(max_tokens=50)
    sections = (
        _section(SECTION_SYSTEM, 40),
        _section(SECTION_GOAL, 40),
    )
    with pytest.raises(ContextBudgetExceededError, match="缩小任务范围"):
        budget.fit(sections)


def test_missing_sections_are_simply_absent() -> None:
    budget = ContextBudget(max_tokens=1_000)
    kept = budget.fit((_section(SECTION_SYSTEM, 10),))
    assert len(kept) == 1


def test_reservation_cannot_fill_the_whole_budget() -> None:
    with pytest.raises(ValueError, match="占满"):
        ContextBudget(max_tokens=100, reserved_output_tokens=100)


def test_non_positive_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        ContextBudget(max_tokens=0)


# ---------------------------------------------------------------------------
# 压缩结果
# ---------------------------------------------------------------------------


def test_compaction_records_what_was_dropped() -> None:
    """用户问「为什么没引用那个文件」时，答案往往是「压缩时被裁掉了」。"""
    result = CompactionResult(
        kept_section_names=(SECTION_SYSTEM, SECTION_GOAL, SECTION_EVIDENCE),
        dropped_section_names=(SECTION_REPOSITORY_MAP,),
        tokens_before=500,
        tokens_after=300,
    )
    assert SECTION_REPOSITORY_MAP in result.dropped_section_names


def test_compaction_cannot_drop_protected_sections() -> None:
    with pytest.raises(ValueError, match="不允许被裁剪"):
        CompactionResult(
            kept_section_names=(SECTION_GOAL,),
            dropped_section_names=(SECTION_SYSTEM,),
            tokens_before=500,
            tokens_after=300,
        )


def test_compaction_cannot_grow_tokens() -> None:
    with pytest.raises(ValueError, match="不应大于"):
        CompactionResult(
            kept_section_names=(SECTION_SYSTEM,),
            dropped_section_names=(),
            tokens_before=100,
            tokens_after=200,
        )
