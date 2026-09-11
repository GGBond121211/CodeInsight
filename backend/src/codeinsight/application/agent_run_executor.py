"""Worker 侧重建并执行一次 Agent Run。

受理与执行分离之后，两边只能通过事实 Store 交流：API 只留下 session/turn/run 标识
和「这一轮被要求做什么」，Worker 拿到 task 之后自己去把仓库根、历史与本轮问题读回来。
这个模块负责那次读，以及执行前后的边界。

依赖方向刻意保持干净：这里不 import conversation_service，需要的仓库指纹函数由装配
方注入，因此加载器在真正的跨进程 Worker 里可以原样复用。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codeinsight.application.conversation_router import (
    ChatTaskClassification,
    classify_chat_task,
)
from codeinsight.application.session_service import SessionService
from codeinsight.domain.agent_run import AgentRunRecord, AgentRunTask, RunRequestOptions
from codeinsight.domain.errors import (
    ModelCallError,
    ModelConfigurationError,
    ModelResponseError,
)
from codeinsight.domain.trace import CONTEXT_LOADED
from codeinsight.infrastructure.chat_runtime import ChatExecution
from codeinsight.infrastructure.gateway_errors import GatewayError
from codeinsight.infrastructure.model_gateway import resolve_route_budget

# 上下文重建失败的公开分类。它们都属于「重试也没用」：会话没了、仓库根没落库、
# 目录不在了——再跑一次结果一样，所以 Worker 直接判失败，不留队列。
CONTEXT_SESSION_NOT_FOUND = "CONTEXT_SESSION_NOT_FOUND"
CONTEXT_REPOSITORY_UNKNOWN = "CONTEXT_REPOSITORY_UNKNOWN"
CONTEXT_REPOSITORY_MISSING = "CONTEXT_REPOSITORY_MISSING"
CONTEXT_MESSAGE_NOT_FOUND = "CONTEXT_MESSAGE_NOT_FOUND"
# 执行体抛出的异常分类。
EXECUTION_FAILED = "EXECUTION_FAILED"
DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"


class AgentRunContextError(RuntimeError):
    """无法只凭事实 Store 重建这一轮的执行上下文。"""

    def __init__(self, error_class: str) -> None:
        super().__init__(error_class)
        self.error_class = error_class


class AgentRunExecutionError(RuntimeError):
    """执行体未返回结果就失败。retryable 表示重试是否有意义。"""

    def __init__(self, error_class: str, *, retryable: bool) -> None:
        super().__init__(error_class)
        self.error_class = error_class
        self.retryable = retryable


@dataclass(frozen=True)
class AgentRunContext:
    """Worker 重建出来的执行输入。全部字段都来自事实 Store。"""

    run_id: str
    turn_id: str
    session_id: str
    task_id: str
    repo_id: str
    repo_root: str
    repo_fingerprint: str
    user_message: str
    task_type: str
    confidence: float
    rule: str
    options: RunRequestOptions
    # 隔离工作区里是否已经有未验证的改动。只读 Run 恒为 False；change 路径在
    # _execute_turn 内部按 task_type 自行置 True，因此这里不必猜。
    contains_workspace_state: bool = False

    @property
    def classification(self) -> ChatTaskClassification:
        return ChatTaskClassification(
            task_type=self.task_type, confidence=self.confidence, rule=self.rule
        )


class AgentRunContextLoader:
    """只凭 task 与事实 Store 重建一轮执行。

    这里没有任何 API 进程的内存依赖：会话、仓库根、历史、本轮问题全部来自
    Session/Memory 事实，请求参数来自 Run 记录。换一个进程、换一台机器，
    加载出来的结果必须一样——这正是它和「把上下文塞进任务消息」的区别。
    """

    def __init__(
        self,
        *,
        session_service: SessionService,
        index_version: str,
        fingerprint: Callable[[str], str],
    ) -> None:
        self._session_service = session_service
        self._index_version = index_version
        self._fingerprint = fingerprint

    def load(self, task: AgentRunTask, record: AgentRunRecord) -> AgentRunContext:
        session = self._session_service.load_session(task.session_id)
        if session is None:
            raise AgentRunContextError(CONTEXT_SESSION_NOT_FOUND)
        repo_root = session.repo_root.strip()
        if not repo_root:
            # 会话没有绑定仓库根。它必须在受理时就落库——否则跨进程的 Worker
            # 无从知道该读哪个目录，这一轮就永久卡住。
            raise AgentRunContextError(CONTEXT_REPOSITORY_UNKNOWN)
        if not Path(repo_root).is_dir():
            raise AgentRunContextError(CONTEXT_REPOSITORY_MISSING)
        message = self._last_user_message(
            session_id=session.session_id,
            scope=session.scope,
            repo_id=session.repo_id,
        )
        classification = classify_chat_task(
            message, has_active_code_goal=session.active_goal_id is not None
        )
        return AgentRunContext(
            run_id=task.run_id,
            turn_id=task.turn_id,
            session_id=task.session_id,
            task_id=task.task_id,
            repo_id=session.repo_id,
            repo_root=repo_root,
            repo_fingerprint=self._fingerprint(repo_root),
            user_message=message,
            task_type=classification.task_type,
            confidence=classification.confidence,
            rule=classification.rule,
            options=record.options,
        )

    def _last_user_message(self, *, session_id: str, scope, repo_id: str) -> str:
        memory = self._session_service.load_session_memory(
            session_id=session_id, scope=scope, repo_id=repo_id
        )
        if memory is None:
            raise AgentRunContextError(CONTEXT_MESSAGE_NOT_FOUND)
        for turn in reversed(memory.recent_turns):
            if turn.role == "user":
                return turn.content
        raise AgentRunContextError(CONTEXT_MESSAGE_NOT_FOUND)


class AgentRunExecutor:
    """执行一次已重建的只读 Agent Run，并守住执行前后的边界。

    它只做两件有实质意义的事：

    1. 在模型被调用之前，把这一轮实际采用的 Context Lifecycle 预算写进事件。
       预算来自统一的 resolve_route_budget，不是调用方临时算的——这样
       「每一次模型调用都经过同一套策略」才是可核对的事实，而不是约定。
    2. 把执行体的失败翻译成公开错误类别，让 Worker 不必认识领域异常。

    业务分支（普通对话 / 澄清 / 范围引导 / 代码理解 / 修改）不在这里再分叉一次：
    分叉逻辑只有一份，放在 ConversationService；这里只按 task_type 决定场景。
    """

    def __init__(
        self,
        *,
        emit: Callable[[str, str, dict[str, str]], None],
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        self._emit = emit
        self._clock_ms = clock_ms or _now_ms

    @staticmethod
    def scene_for(task_type: str) -> str:
        return "change-plan" if task_type == "change" else "explain"

    def execute(
        self,
        context: AgentRunContext,
        *,
        runner: Callable[[AgentRunContext], ChatExecution],
    ) -> ChatExecution:
        scene = self.scene_for(context.task_type)
        budget = resolve_route_budget(scene)
        self._emit(
            context.run_id,
            CONTEXT_LOADED,
            {
                "scene": scene,
                "task_type": context.task_type,
                "context_window_tokens": str(budget.context_window_tokens),
                "input_allowance": str(budget.input_allowance),
                "reserved_output_tokens": str(budget.reserved_output_tokens),
                "session_history_budget": str(budget.session_history_budget),
                "policy_version": budget.policy_version,
                "rule": context.rule,
            },
        )
        try:
            return runner(context)
        except (ModelCallError, GatewayError, ModelResponseError) as error:
            raise AgentRunExecutionError(type(error).__name__, retryable=True) from error
        except (ModelConfigurationError, ValueError, OSError) as error:
            raise AgentRunExecutionError(type(error).__name__, retryable=False) from error


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)

