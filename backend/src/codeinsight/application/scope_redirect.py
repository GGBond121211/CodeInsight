"""业务范围外输入的自然引导文案。

这条路径不调用模型，也不访问仓库；它只把用户自然地带回 CodeInsight
支持的代码理解与修改任务。原始用户消息仍由 Session 记录，便于复盘。
"""

SCOPE_REDIRECT_VERSION = "scope-redirect-v1"


def build_scope_redirect_message(*, has_active_code_goal: bool) -> str:
    """返回稳定、简短且不扩展无关话题的范围引导。"""
    if has_active_code_goal:
        return (
            "收到，我理解你是在分享一个日常话题。我们继续围绕当前代码任务吧："
            "你可以说“继续检查刚才的函数”，或者直接告诉我文件、函数和报错。"
        )
    return (
        "收到，我理解你是在分享一个日常话题。我是 CodeInsight，主要帮助你理解仓库、"
        "定位 Bug 和修改代码。你可以直接告诉我文件、函数或报错，例如“解释 workflow.py”"
        "或“检查这个函数有没有 Bug”。"
    )
