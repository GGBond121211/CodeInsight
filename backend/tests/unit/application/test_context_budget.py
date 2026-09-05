from __future__ import annotations

import pytest

from codeinsight.application.context_assembler import ContextAssembler, ContextRequest
from codeinsight.application.context_budget import estimate_budget, make_section
from codeinsight.domain.memory import (
    SECTION_CURRENT_DIFF,
    SECTION_EVIDENCE,
    SECTION_GOAL,
    SECTION_SYSTEM,
    ContextBudgetExceededError,
)


def test_assembler_does_not_create_an_empty_evidence_section() -> None:
    assembly = ContextAssembler().assemble(
        ContextRequest(
            system_safety="不能越权",
            user_goal="解释函数",
            user_code_task="找到入口",
        )
    )
    assert SECTION_EVIDENCE not in [section.name for section in assembly.sections]


def test_repository_content_is_marked_untrusted() -> None:
    assembly = ContextAssembler().assemble(
        ContextRequest(
            system_safety="安全规则",
            user_goal="解释",
            user_code_task="任务",
            evidence="E1 src/a.py:1-3",
            current_diff="diff --git a/src/a.py",
        )
    )
    sections = {section.name: section for section in assembly.sections}
    assert sections[SECTION_EVIDENCE].is_untrusted is True
    assert sections[SECTION_CURRENT_DIFF].is_untrusted is True


def test_budget_keeps_goal_code_task_evidence_and_diff() -> None:
    sections = tuple(
        section
        for section in (
            make_section(SECTION_SYSTEM, "safe " * 20, is_untrusted=False),
            make_section(SECTION_GOAL, "goal " * 20),
            make_section("user_code_task", "task " * 20),
            make_section(SECTION_EVIDENCE, "E1 path.py:1-2 " * 20, is_untrusted=True),
            make_section(SECTION_CURRENT_DIFF, "diff " * 20, is_untrusted=True),
            make_section("repository_map", "map " * 400),
        )
        if section is not None
    )
    result = estimate_budget(sections, max_tokens=100, reserved_output_tokens=10)
    assert result.fits is False
    fitted = ContextAssembler().assemble(
        ContextRequest(
            system_safety="safe",
            user_goal="goal",
            user_code_task="task",
            evidence="E1 path.py:1-2",
            current_diff="diff",
            repository_map="map " * 400,
            max_tokens=100,
            reserved_output_tokens=10,
        )
    )
    names = [section.name for section in fitted.fitted.sections]
    assert {
        SECTION_SYSTEM,
        SECTION_GOAL,
        "user_code_task",
        SECTION_EVIDENCE,
        SECTION_CURRENT_DIFF,
    } <= set(names)
    assert fitted.fitted.compaction.tokens_after <= fitted.fitted.compaction.tokens_before


def test_overlarge_protected_context_fails_loudly() -> None:
    with pytest.raises(ContextBudgetExceededError):
        ContextAssembler().assemble(
            ContextRequest(
                system_safety="safe " * 100,
                user_goal="goal",
                user_code_task="task",
                evidence="E1 path.py:1-2",
                current_diff="diff",
                max_tokens=20,
                reserved_output_tokens=0,
            )
        )


def test_excess_tool_schema_and_history_are_compacted_first() -> None:
    assembly = ContextAssembler().assemble(
        ContextRequest(
            system_safety="safe",
            user_goal="goal",
            user_code_task="task",
            evidence="E1 src/a.py:1-2",
            current_diff="diff --git a/src/a.py",
            tool_schema="tool " * 400,
            session_summary="history " * 400,
            repository_map="map " * 400,
            output_schema="output " * 400,
            max_tokens=100,
            reserved_output_tokens=10,
        )
    )
    names = {section.name for section in assembly.fitted.sections}
    assert {
        SECTION_SYSTEM,
        SECTION_GOAL,
        "user_code_task",
        SECTION_EVIDENCE,
        SECTION_CURRENT_DIFF,
    } <= names
    assert "tool_schema" not in names
    assert "session_summary" not in names
    assert assembly.fitted.compaction.dropped_section_names
