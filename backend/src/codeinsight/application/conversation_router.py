"""统一对话的保守任务分类器。

明确的写意图才进入 change 路线；不确定表达一律走 explain/read-only。这样
业务编排层可以把「解释之后再修改」放在同一 Session，同时不因模型猜错而越过
修改审批门禁。后续若引入专门意图模型，只需替换这个纯函数并保留同一输出契约。
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
    r"(如何|怎么|为什么|哪里|什么|解释|说明|how|why|where|what)", re.IGNORECASE
)


def classify_chat_task(message: str) -> ChatTaskClassification:
    if not message.strip():
        raise ValueError("message 不能为空")
    if _CHANGE_WORDS.search(message) and not _QUESTION_WORDS.search(message):
        return ChatTaskClassification("change", 0.90, "explicit_write_intent")
    return ChatTaskClassification("explain", 0.85, "read_only_default")
