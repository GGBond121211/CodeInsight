"""统一对话入口的轻量领域值对象。

ChatTurn 是用户可见的一轮交互状态；它不承载模型隐藏推理。调试 reasoning
只通过进程内实时通道短暂传输，不进入这个对象、Session 或持久事件日志。
"""

from __future__ import annotations

from dataclasses import dataclass

CHAT_QUEUED = "QUEUED"
CHAT_RUNNING = "RUNNING"
CHAT_WAITING_APPROVAL = "WAITING_APPROVAL"
CHAT_COMPLETED = "COMPLETED"
CHAT_FAILED = "FAILED"
CHAT_CANCELLED = "CANCELLED"

SUPPORTED_CHAT_STATUSES = frozenset(
    {
        CHAT_QUEUED,
        CHAT_RUNNING,
        CHAT_WAITING_APPROVAL,
        CHAT_COMPLETED,
        CHAT_FAILED,
        CHAT_CANCELLED,
    }
)
TERMINAL_CHAT_STATUSES = frozenset({CHAT_COMPLETED, CHAT_FAILED, CHAT_CANCELLED})
SUPPORTED_CHAT_TASK_TYPES = frozenset({"general_chat", "clarify", "explain", "change"})


@dataclass(frozen=True)
class ChatTurn:
    turn_id: str
    session_id: str
    run_id: str
    task_type: str
    status: str
    user_message: str
    assistant_message: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    reasoning_available: bool = False
    created_at_epoch_ms: int = 0
    updated_at_epoch_ms: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.turn_id, "turn_id"),
            (self.session_id, "session_id"),
            (self.run_id, "run_id"),
            (self.user_message, "user_message"),
        ):
            if not value.strip():
                raise ValueError(f"{label} 不能为空")
        if self.task_type not in SUPPORTED_CHAT_TASK_TYPES:
            raise ValueError(f"不支持的聊天任务类型：{self.task_type}")
        if self.status not in SUPPORTED_CHAT_STATUSES:
            raise ValueError(f"不支持的聊天状态：{self.status}")
        if self.created_at_epoch_ms <= 0 or self.updated_at_epoch_ms <= 0:
            raise ValueError("聊天时间戳必须为正")

    def as_dict(self) -> dict[str, object]:
        """返回前端可展示的公开状态，不包含 approval token 或 reasoning 原文。"""
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "run_id": self.run_id,
            "task_type": self.task_type,
            "status": self.status,
            "user_message": self.user_message,
            "assistant_message": self.assistant_message,
            "result": self.result,
            "error": self.error,
            "reasoning_available": self.reasoning_available,
            "created_at_epoch_ms": self.created_at_epoch_ms,
            "updated_at_epoch_ms": self.updated_at_epoch_ms,
        }
