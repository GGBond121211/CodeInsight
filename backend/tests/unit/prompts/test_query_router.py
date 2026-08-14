"""Query Router Prompt 契约测试。"""

from codeinsight.prompts.query_router import PROMPT_VERSION, build_router_prompt


def test_router_prompt_preserves_user_text_and_forbids_evidence_generation() -> None:
    system, user = build_router_prompt("checkout 先校验什么？")

    assert PROMPT_VERSION == "query-router-v4"
    assert "文件路径" in system
    assert "evidence ID" in system
    assert "一个边界明确的请求通常只有一个" in system
    assert "默认使用 linear" in system
    assert "Semantic 检索总是" in system
    assert "checkout 先校验什么？" in user
