import pytest

from codeinsight.application.conversation_router import classify_chat_task


@pytest.mark.parametrize("message", ["你好", "谢谢，你能做什么？", "hello"])
def test_smalltalk_uses_general_chat_route(message: str) -> None:
    result = classify_chat_task(message)

    assert result.task_type == "general_chat"
    assert result.rule == "smalltalk_or_capability"


@pytest.mark.parametrize(
    "message", ["我喜欢打篮球", "今天天气不错", "我最喜欢科比", "谢谢你，我喜欢打篮球"]
)
def test_out_of_scope_natural_language_uses_scope_redirect(message: str) -> None:
    result = classify_chat_task(message)

    assert result.task_type == "scope_redirect"
    assert result.rule == "out_of_scope_natural_language"


def test_code_anchor_wins_over_greeting() -> None:
    result = classify_chat_task("你好，顺便解释 workflow.py")

    assert result.task_type == "explain"
    assert result.rule == "code_anchor"


def test_explicit_write_intent_keeps_change_route() -> None:
    result = classify_chat_task("把这个 bug 修复一下")

    assert result.task_type == "change"
    assert result.rule == "explicit_write_intent"


def test_write_without_code_anchor_never_enters_change_route() -> None:
    result = classify_chat_task("帮我修改一下说法")

    assert result.task_type == "clarify"
    assert result.rule == "write_without_code_anchor"


def test_ambiguous_reference_requests_clarification_without_active_goal() -> None:
    result = classify_chat_task("这个东西怎么样")

    assert result.task_type == "clarify"
    assert result.rule == "ambiguous_reference"


def test_active_code_goal_resolves_ambiguous_reference_as_explain() -> None:
    result = classify_chat_task("这个地方是不是有问题", has_active_code_goal=True)

    assert result.task_type == "explain"
    assert result.rule == "active_code_goal_reference"
