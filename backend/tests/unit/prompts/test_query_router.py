"""Tests for the Query Router prompt contract."""

from codeinsight.prompts.query_router import PROMPT_VERSION, build_router_prompt


def test_router_prompt_preserves_user_text_and_forbids_evidence_generation() -> None:
    system, user = build_router_prompt("checkout 先校验什么？")

    assert PROMPT_VERSION == "query-router-v4"
    assert "file paths" in system
    assert "evidence IDs" in system
    assert "single bounded request normally has exactly one subquestion" in system
    assert "Use linear by default" in system
    assert "Semantic retrieval is always added" in system
    assert "checkout 先校验什么？" in user
