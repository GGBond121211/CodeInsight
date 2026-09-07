"""普通自然语言对话的版本化 Prompt。"""

PROMPT_VERSION = "general-chat-v1"

SYSTEM_PROMPT = """你是 CodeInsight 的普通对话助手。
请使用自然、简洁的语言回答用户，不要返回 JSON，不要使用代码仓库证据格式。
你可以回答问候、能力介绍和一般性问题；如果用户实际想理解或修改代码，应请
用户提供文件、函数或仓库问题，交由代码理解/修改入口处理。
不要编造当前仓库的事实、运行结果、模型调用结果或未提供的上下文。"""


def build_general_chat_prompt(message: str) -> tuple[str, str]:
    """构造只承载用户当前消息的自然语言 Prompt；Session 历史由编排层注入。"""
    if not message.strip():
        raise ValueError("message 不能为空")
    return SYSTEM_PROMPT, f"用户消息：\n{message.strip()}"
