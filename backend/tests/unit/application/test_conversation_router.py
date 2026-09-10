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


@pytest.mark.parametrize("message", ["把这个 bug 修复一下", "请把未知 region 的处理修好"])
def test_explicit_write_intent_keeps_change_route(message: str) -> None:
    result = classify_chat_task(message)

    assert result.task_type == "change"
    assert result.rule == "explicit_write_intent"


@pytest.mark.parametrize(
    "message",
    [
        "请看 src/shop/pricing.py 的 unit_price 实现",
        "解释 workflow.py 里面的实现",
    ],
)
def test_implementation_as_noun_keeps_code_explanation_route(message: str) -> None:
    result = classify_chat_task(message)

    assert result.task_type == "explain"
    assert result.rule == "code_anchor"


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


@pytest.mark.parametrize(
    "message",
    [
        "中间的 planned 记录具体在哪里产生？",
        "total_cents 里面包含税或折扣吗？",
        "对比 customer 和 admin 两个 find_record，它们的 scope 为什么不同？",
        "文档里的 standard lookup 能覆盖 admin 版本吗？",
    ],
)
def test_active_code_goal_resolves_semantic_follow_up_without_new_path(
    message: str,
) -> None:
    result = classify_chat_task(message, has_active_code_goal=True)

    assert result.task_type == "explain"
    assert result.rule == "active_code_goal_follow_up"


def test_thanks_and_return_to_code_remains_general_chat() -> None:
    result = classify_chat_task("Thanks，接下来先回到代码。", has_active_code_goal=True)

    assert result.task_type == "general_chat"
    assert result.rule == "smalltalk_or_capability"
