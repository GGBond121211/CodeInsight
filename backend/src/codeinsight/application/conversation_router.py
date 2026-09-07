"""统一对话的确定性入口分类器。

分类器只决定业务编排入口，不替代仓库查询 Router：普通寒暄走自然语言回答，
代码问题走只读理解，明确写意图走受审批保护的修改流程，无法判断的短句先澄清。
规则优先保证高风险修改不会因为模型误判而越过门禁；后续若引入专门意图模型，
只需替换这个纯函数并保留同一输出契约。
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ChatTaskClassification:
    task_type: str
    confidence: float
    rule: str


_CHANGE_WORDS = re.compile(
    r"(修改|改成|改为|修复|增加|新增|删除|移除|实现|重构|替换|应用补丁|apply|change|fix|add|remove|implement|refactor)",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(
    r"(如何|怎么|为什么|哪里|什么|解释|说明|吗|是否|能否|能不能|是不是|可不可以|"
    r"how|why|where|what)",
    re.IGNORECASE,
)
_CODE_ANCHORS = re.compile(
    r"(代码|仓库|文件|函数|方法|类|模块|接口|调用|参数|返回值|入口|校验|验证|检查|"
    r"错误|异常|堆栈|日志|测试|编译|运行|工作流|派送|线路|bug|checkout|"
    r"python|javascript|typescript|java|sql|api|\.py\b|\.ts\b|\.tsx\b|\.js\b|"
    r"\.java\b|\.sql\b|[A-Za-z0-9_.-]+[/\\][A-Za-z0-9_./\\-]+)",
    re.IGNORECASE,
)
_CODE_IDENTIFIER = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])")
_AMBIGUOUS_REQUEST = re.compile(
    r"^(?:(?:那|那就|然后|所以)?\s*)?"
    r"(?:这个|那个|它|刚才(?:那个|的)?|上一轮|继续|再看看|再看一下|还有吗|"
    r"这(?:个|样)?(?:怎么样|可以吗|对吗|有问题吗|有没有问题|是不是有问题|好吗)?|"
    r"(?:这个|那个)(?:地方|东西)?(?:怎么样|可以吗|对吗|有问题吗|有没有问题|是不是有问题|好吗)|"
    r"(?:这样|那样)(?:可以吗|对吗|好吗)?)[\s，。！？!?；;]*$",
    re.IGNORECASE,
)
_RESUME_WORDS = re.compile(r"(继续|刚才|上一轮|再看看|再看一下|接着)", re.IGNORECASE)
_GENERAL_CHAT = re.compile(
    r"(你好|您好|嗨|哈喽|hello|hi|hey|谢谢|感谢|辛苦了|早上好|晚上好|"
    r"你是谁|你能做什么|在吗|好的|明白了|收到|再见)",
    re.IGNORECASE,
)


def classify_chat_task(
    message: str, *, has_active_code_goal: bool = False
) -> ChatTaskClassification:
    """把用户消息映射为产品级入口，不在这里调用模型或仓库工具。

    ``has_active_code_goal`` 只用于解释同一 Session 内的指代词。普通寒暄不会
    覆盖活动中的代码目标；有代码锚点的消息始终优先于寒暄词，例如“你好，顺便
    解释 workflow.py”仍然进入代码理解路线。
    """
    if not message.strip():
        raise ValueError("message 不能为空")
    normalized = message.strip()
    has_code_anchor = bool(_CODE_ANCHORS.search(normalized))
    has_write_target = has_code_anchor or bool(_CODE_IDENTIFIER.search(normalized))
    if _CHANGE_WORDS.search(normalized) and not _QUESTION_WORDS.search(normalized):
        if has_write_target or has_active_code_goal:
            return ChatTaskClassification("change", 0.90, "explicit_write_intent")
        return ChatTaskClassification("clarify", 0.55, "write_without_code_anchor")
    if _AMBIGUOUS_REQUEST.fullmatch(normalized):
        if has_active_code_goal:
            return ChatTaskClassification("explain", 0.80, "active_code_goal_reference")
        return ChatTaskClassification("clarify", 0.60, "ambiguous_reference")
    if has_code_anchor:
        return ChatTaskClassification("explain", 0.85, "code_anchor")
    if has_active_code_goal and _RESUME_WORDS.search(normalized):
        return ChatTaskClassification("explain", 0.78, "active_code_goal_reference")
    if _GENERAL_CHAT.search(normalized):
        return ChatTaskClassification("general_chat", 0.98, "smalltalk_or_capability")
    # 没有代码锚点的自然语言默认仍是普通对话；只有明显无法指代目标的短句才澄清。
    return ChatTaskClassification("general_chat", 0.75, "natural_language_without_code_anchor")
