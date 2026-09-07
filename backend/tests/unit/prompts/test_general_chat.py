from codeinsight.prompts.general_chat import PROMPT_VERSION, build_general_chat_prompt


def test_general_chat_prompt_is_text_oriented_and_versioned() -> None:
    system_prompt, user_prompt = build_general_chat_prompt("你好")

    assert PROMPT_VERSION == "general-chat-v1"
    assert "不要返回 JSON" in system_prompt
    assert "你好" in user_prompt
