from codeinsight.application.scope_redirect import (
    SCOPE_REDIRECT_VERSION,
    build_scope_redirect_message,
)


def test_scope_redirect_message_guides_a_new_session_back_to_code() -> None:
    message = build_scope_redirect_message(has_active_code_goal=False)

    assert SCOPE_REDIRECT_VERSION == "scope-redirect-v1"
    assert "CodeInsight" in message
    assert "workflow.py" in message
    assert "Bug" in message


def test_scope_redirect_message_preserves_an_active_code_goal() -> None:
    message = build_scope_redirect_message(has_active_code_goal=True)

    assert "当前代码任务" in message
    assert "继续检查刚才的函数" in message
