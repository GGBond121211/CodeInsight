"""统一对话入口与业务编排层。

一轮聊天在这里被拆成：恢复 Session -> 分类解释/修改 -> 组装上下文 ->
执行既有只读或变更工作流 -> 记录公开回答。前端不需要知道两条内部路径，
但每一步都通过同一个 run_id 进入实时事件流。
"""

from __future__ import annotations

import hashlib
import inspect
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from codeinsight.agent.agent_run_worker import (
    RUN_RECORD_MISSING,
    AgentRunOutcome,
    AgentRunWorker,
)
from codeinsight.agent.change_workflow import run_change_workflow_to_preview
from codeinsight.agent.tool_loop import ToolLoopConfig
from codeinsight.application.agent_run_dispatcher import (
    AgentRunDispatcher,
    AgentRunTransport,
    CallbackAgentRunTransport,
    TransportRejected,
)
from codeinsight.application.agent_run_executor import (
    AgentRunContext,
    AgentRunContextError,
    AgentRunContextLoader,
    AgentRunExecutor,
)
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.code_understanding_route import (
    run_code_understanding_answer,
    to_auto_answer,
    uses_code_understanding,
)
from codeinsight.application.context_budget import estimate_tokens
from codeinsight.application.conversation_router import (
    ChatTaskClassification,
    classify_chat_task,
)
from codeinsight.application.query_router import route_question
from codeinsight.application.scope_redirect import (
    SCOPE_REDIRECT_VERSION,
    build_scope_redirect_message,
)
from codeinsight.application.session_service import SessionContext, SessionService
from codeinsight.application.validation_coordinator import (
    CallbackValidationTransport,
    ValidationCoordinator,
    ValidationTaskQueue,
    ValidationTransport,
    ValidationWorker,
)
from codeinsight.domain.agent_run import (
    CANCELLED,
    COMPLETED,
    FAILED,
    MANUAL_REQUIRED,
    OPEN_AGENT_RUN_STATUSES,
    QUEUED,
    RUNNING,
    TASK_AGENT_RUN,
    TASK_RESUME_AFTER_APPROVAL,
    TASK_RESUME_AFTER_VALIDATION,
    UNKNOWN,
    WAITING_APPROVAL,
    WAITING_VALIDATION,
    AgentRunRecord,
    AgentRunTask,
    RunRequestOptions,
)
from codeinsight.domain.answer import AutoAnswer
from codeinsight.domain.change import MODE_ISOLATED_WRITE, MODE_READ_ONLY, TenantScope
from codeinsight.domain.chat import (
    CHAT_CANCELLED,
    CHAT_COMPLETED,
    CHAT_FAILED,
    CHAT_MANUAL_REQUIRED,
    CHAT_QUEUED,
    CHAT_RUNNING,
    CHAT_UNKNOWN,
    CHAT_WAITING_APPROVAL,
    CHAT_WAITING_VALIDATION,
    SETTLED_CHAT_STATUSES,
    ChatTurn,
)
from codeinsight.domain.errors import ModelCallError, ModelConfigurationError, ModelResponseError
from codeinsight.domain.ports import AgentRunStore
from codeinsight.domain.trace import (
    ANSWER_READY,
    APPROVAL_GRANTED,
    APPROVAL_REQUESTED,
    CONTEXT_ASSEMBLED,
    CONTEXT_COMPACTED,
    INTENT_CLASSIFIED,
    MODEL_GENERATING,
    PATCH_REJECTED,
    REPAIR_ATTEMPTED,
    RETRIEVAL_FINISHED,
    RETRIEVAL_STARTED,
    SESSION_LOADED,
    STEP_STARTED,
    TASK_QUEUED,
    VALIDATION_PREFLIGHT,
)
from codeinsight.infrastructure.chat_runtime import (
    ChatExecution,
    ChatRuntime,
    ChatRuntimeError,
)
from codeinsight.infrastructure.gateway_errors import GatewayError
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.model_gateway import CacheContext, resolve_route_budget
from codeinsight.infrastructure.redaction import redact_sensitive
from codeinsight.infrastructure.redis_cache import RedisCache
from codeinsight.infrastructure.reranker import Reranker
from codeinsight.infrastructure.run_store import InMemorySessionStore
from codeinsight.infrastructure.runtime_policy import DevelopmentPolicy
from codeinsight.infrastructure.task_queue import TaskEnvelope, default_validation_queue
from codeinsight.ingestion.scanner import scan_repository

# 2026-09-11：旧 LangGraph 路线整体冻结后，本文件不再需要这些符号；保留原名
# 以便回退：AgentRepositoryAnswer、QueryRouterResult、AutoAnswerEvent、
# SubQuestionAnswer。

INDEX_VERSION = "conversation-scan-v1"

# 一次 Chat Agent Run 的策略版本与受理期限。期限只约束「这一轮多久之内必须
# 跑完」，不是模型的超时；超时后的处理见 recovery_decision。
CHAT_RUN_POLICY_VERSION = "chat-agent-run-v1"
DEFAULT_RUN_DEADLINE_MS = 10 * 60 * 1000
# 一次校验失败之后允许自动重修几轮。这个数字是登记过的工程基线，不是调优结果：
# 预算用完还是不过，就停下来交给人，不做没有上限的自动循环。
MAX_CHANGE_REPAIR_ROUNDS = 1

_CHAT_STATUS_BY_RUN_STATUS: dict[str, str] = {
    QUEUED: CHAT_QUEUED,
    RUNNING: CHAT_RUNNING,
    WAITING_APPROVAL: CHAT_WAITING_APPROVAL,
    WAITING_VALIDATION: CHAT_WAITING_VALIDATION,
    COMPLETED: CHAT_COMPLETED,
    FAILED: CHAT_FAILED,
    CANCELLED: CHAT_CANCELLED,
    UNKNOWN: CHAT_UNKNOWN,
    MANUAL_REQUIRED: CHAT_MANUAL_REQUIRED,
}


class ChatDispatchError(RuntimeError):
    """这一轮没有成功交给 Worker。

    Run 已经留下失败事实（FAILED / MANUAL_REQUIRED），所以调用方看到的是
    「受理失败但可查」，而不是一个悬在半空的 202。
    """
ModelFactory = Callable[[], object]
EmbeddingFactory = Callable[[], object]
RerankerFactory = Callable[[], Reranker]


@dataclass(frozen=True)
class _TurnInput:
    repository_root: str
    repo_id: str
    repo_fingerprint: str
    validation_profile: str
    limit: int
    contains_workspace_state: bool = False


@dataclass(frozen=True)
class _SessionWorkspace:
    repo_id: str
    source_root: Path
    effective_root: Path


def _turn_id_for(session_id: str, client_turn_id: str | None) -> str:
    """本轮的 turn_id。

    带 client_turn_id 时按它推导，而不是随机生成：这样「同一个请求重发」在
    不同进程、不同时刻都会算出同一个 turn_id，agent_runs 上的唯一约束就替
    我们挡住了第二个 Run。随机 ID 只能靠进程内字典去重，重启即失效。
    """

    if client_turn_id is None:
        return f"turn-{uuid4().hex[:16]}"
    digest = hashlib.sha256(f"{session_id}:{client_turn_id}".encode()).hexdigest()
    return f"turn-{digest[:16]}"


def _turn_from_run_record(
    record: AgentRunRecord, *, user_message: str, task_type: str
) -> ChatTurn:
    """把持久化的 Run 事实翻译成一轮对话的公开状态。

    只在「本进程没见过这一轮，但 Store 里确实有它」时使用：这时返回 404 会让
    界面以为消息丢了，而事实其实一直在。分类结果来自重发时的同一条消息。
    """

    return ChatTurn(
        turn_id=record.turn_id,
        session_id=record.session_id,
        run_id=record.run_id,
        task_type=task_type,
        status=_CHAT_STATUS_BY_RUN_STATUS.get(record.status, CHAT_FAILED),
        user_message=user_message,
        error=record.error_class,
        created_at_epoch_ms=record.updated_at_epoch_ms,
        updated_at_epoch_ms=record.updated_at_epoch_ms,
        task_id=record.task_id,
    )


@dataclass(frozen=True)
class _ChangeRecovery:
    """上一轮校验失败后，供同一 Session 继续修复的公开摘要。"""

    run_id: str
    patch_id: str
    validation: dict[str, object]


def default_session_service(*, max_recent_turns: int = 12) -> SessionService:
    """按环境装配 Session 事实层。

    配了 MySQL 就用真表，否则退回进程内存。Redis 只作为可失效热点缓存，
    永远不是事实来源：它挂掉时 ``cache_aside`` 会回源 Store。

    为什么默认要读环境：公开多轮会话如果只存在进程内存里，API 一重启，
    用户看到的会话就消失了，而日志里不会有任何错误——恢复链路没闭合是
    静默故障，不是性能问题。
    """
    cache = RedisCache.from_environment()
    from codeinsight.infrastructure.db.engine import (
        MySqlConfig,
        create_all_tables,
        create_db_engine,
        create_session_factory,
    )

    mysql = MySqlConfig.from_env()
    if mysql is None:
        return SessionService(
            InMemorySessionStore(),
            InMemoryMemoryStore(),
            cache,
            max_recent_turns=max_recent_turns,
        )
    from codeinsight.infrastructure.db.stores import MySqlMemoryStore, MySqlSessionStore

    engine = create_db_engine(mysql)
    create_all_tables(engine)
    session_factory = create_session_factory(engine)
    return SessionService(
        MySqlSessionStore(session_factory),
        MySqlMemoryStore(session_factory),
        cache,
        max_recent_turns=max_recent_turns,
    )


def _turn_input_from_context(context: AgentRunContext) -> _TurnInput:
    """把 Worker 重建出来的上下文翻译成执行为所需的输入。

    执行体（_execute_turn）只认 _TurnInput；这里做一次翻译，避免让执行体同时
    认识两种形状。
    """

    return _TurnInput(
        repository_root=context.repo_root,
        repo_id=context.repo_id,
        repo_fingerprint=context.repo_fingerprint,
        validation_profile=context.options.validation_profile,
        limit=context.options.result_limit,
        contains_workspace_state=context.contains_workspace_state,
    )



def default_agent_run_store() -> AgentRunStore:
    """按环境装配 Agent Run 事实层。配了 MySQL 就用真表，否则退回进程内存。

    与 default_session_service 同样的理由：Run 的事实只存在进程内存里时，
    API 一重启，「这一轮到底跑没跑」就没人答得上来。
    """

    from codeinsight.infrastructure.db.engine import (
        MySqlConfig,
        create_all_tables,
        create_db_engine,
        create_session_factory,
    )
    from codeinsight.infrastructure.db.stores import MySqlAgentRunStore
    from codeinsight.infrastructure.run_store import InMemoryAgentRunStore

    mysql = MySqlConfig.from_env()
    if mysql is None:
        return InMemoryAgentRunStore()
    engine = create_db_engine(mysql)
    create_all_tables(engine)
    return MySqlAgentRunStore(create_session_factory(engine))


class ConversationService:
    """把用户体验上的一个对话映射到内部多种 Agent 路径。"""

    def __init__(
        self,
        model_factory: ModelFactory,
        embedding_factory: EmbeddingFactory,
        *,
        reranker_factory: RerankerFactory,
        change_service,
        session_service: SessionService | None = None,
        runtime: ChatRuntime | None = None,
        mcp_client_factory=None,
        agent_run_dispatcher: AgentRunDispatcher | None = None,
        agent_run_store: AgentRunStore | None = None,
        validation_queue: ValidationTaskQueue | None = None,
        agent_run_transport: AgentRunTransport | None = None,
        validation_transport: ValidationTransport | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._embedding_factory = embedding_factory
        self._reranker_factory = reranker_factory
        # 只读代码理解 Tool Loop 的 MCP Client 工厂；测试注入 Fake，生产走 stdio。
        self._mcp_client_factory = mcp_client_factory
        self.change_service = change_service
        selected_policy = getattr(change_service, "development_policy", None)
        self.development_policy = (
            selected_policy
            if isinstance(selected_policy, DevelopmentPolicy)
            else DevelopmentPolicy.from_environment()
        )
        self.session_service = session_service or default_session_service()
        self.runtime = runtime or ChatRuntime()
        self.agent_run_store = agent_run_store or default_agent_run_store()
        if agent_run_dispatcher is None:
            # 默认是进程内回调：单进程部署里投递与执行在同一个进程，事件流也是
            # 同一份。跨进程部署必须显式传入 broker 通道，不能悄悄换掉默认行为。
            agent_run_transport = agent_run_transport or CallbackAgentRunTransport(
                self._schedule_turn
            )
            agent_run_dispatcher = AgentRunDispatcher(
                store=self.agent_run_store, transport=agent_run_transport
            )
        self.agent_run_dispatcher = agent_run_dispatcher
        self.agent_run_context_loader = AgentRunContextLoader(
            session_service=self.session_service,
            index_version=INDEX_VERSION,
            fingerprint=lambda root: _unscanned_repository_fingerprint(Path(root)),
        )
        self.agent_run_executor = AgentRunExecutor(emit=self.runtime.emit)
        self.agent_run_worker = AgentRunWorker(
            store=self.agent_run_store,
            loader=self.agent_run_context_loader,
            executor=self.agent_run_executor,
            emit=self.runtime.emit,
        )
        # 投递那一刻的受理快照。202 要回答的是「我刚受理了哪一轮」，
        # 而不是「它现在跑到哪了」——后者会在响应组装时被 Worker 抢先改写。
        self._accepted_turns: dict[str, ChatTurn] = {}
        self._turn_inputs: dict[str, _TurnInput] = {}
        self._client_turns: dict[tuple[str, str], ChatTurn] = {}
        self._client_turn_lock = threading.RLock()
        self._session_workspaces: dict[str, _SessionWorkspace] = {}
        self._session_workspace_lock = threading.RLock()
        # 「上一次修改失败、等人决定」的会话内记忆：用户说「继续修复」时指的是
        # 上一次那一轮，而那是跨轮次的界面事实。自动修复走的是事实层（见
        # _finalize_validation），这里只负责把用户的追问接到正确的那一轮上。
        self._change_recoveries: dict[str, _ChangeRecovery] = {}
        # 同一轮上「正在登记审批」的占位。重复点击必须在动 approval token 之前
        # 就被挡住，否则第二下会带着自己的 task_id 覆盖掉第一下已经写好的续跑。
        self._approvals_in_flight: set[str] = set()
        self._approval_lock = threading.RLock()
        # 校验任务有自己的一条队列和工人：应用完补丁的那一轮登记完就结束，Docker
        # 在别的线程里跑。单进程用内存队列加本地回调；换成 Redis 队列与 Celery 任务
        # 时语义不变（按标识领取、租约、有限尝试）。
        self.validation_queue = validation_queue or default_validation_queue()
        self.validation_worker = ValidationWorker(
            queue=self.validation_queue,
            change_service=self.change_service,
            emit=self.runtime.emit,
        )
        self.validation_coordinator = ValidationCoordinator(
            queue=self.validation_queue,
            worker=self.validation_worker,
            transport=validation_transport
            or CallbackValidationTransport(self._schedule_validation),
            dispatcher=self.agent_run_dispatcher,
            store=self.agent_run_store,
            emit=self.runtime.emit,
        )
        self._validation_executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="codeinsight-validation"
        )
        add_event_sink = getattr(self.change_service, "add_event_sink", None)
        if callable(add_event_sink):
            add_event_sink(self._mirror_change_event)

    def create_session(
        self, repository_root: str, *, session_id: str | None = None
    ) -> dict[str, object]:
        session_key = session_id or f"session-{uuid4().hex[:16]}"
        root, repo_id, repo_fingerprint, _ = self._resolve_session_repository(
            session_key, repository_root
        )
        context = self.session_service.get_or_create_session(
            session_id=session_key,
            scope=TenantScope(),
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        self._remember_session_workspace(session_key, repo_id, root)
        return _session_payload(context)

    def get_session(self, session_id: str, repository_root: str) -> dict[str, object]:
        root, repo_id, repo_fingerprint, _ = self._resolve_session_repository(
            session_id, repository_root
        )
        context = self.session_service.get_or_create_session(
            session_id=session_id,
            scope=TenantScope(),
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        self._remember_session_workspace(session_id, repo_id, root)
        return _session_payload(context)

    def submit_turn(
        self,
        *,
        session_id: str,
        repository_root: str,
        message: str,
        client_turn_id: str | None = None,
        limit: int = 5,
        validation_profile: str = "python_compile",
        show_debug_reasoning: bool = False,
    ) -> ChatTurn:
        """受理一轮对话：先把 Run 写成事实，再把它交给 Worker。

        返回值代表「这一轮已被受理」，不代表回答已经产生。模型与工具调用在 Worker
        侧执行——放在 API 线程里，一个慢模型就能把整个 HTTP 服务拖住，而进程一重启，
        这一轮连痕迹都不会留下。
        """

        if not message.strip():
            raise ValueError("message 不能为空")
        root, repo_id, repo_fingerprint, contains_workspace_state = (
            self._resolve_session_repository(
                session_id, repository_root, refresh_fingerprint=False
            )
        )
        context = self.session_service.get_or_create_session(
            session_id=session_id,
            scope=TenantScope(),
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        self._remember_session_workspace(session_id, repo_id, root)
        classification = classify_chat_task(
            message,
            has_active_code_goal=context.active_goal is not None,
        )

        turn_id = _turn_id_for(session_id, client_turn_id)
        existing = self.agent_run_store.find_run_by_turn(turn_id)
        if existing is not None:
            # 同一条消息重发：返回同一条事实，不再跑一遍模型。本进程没见过它时
            # （例如 API 重启过），就用 Store 里的事实回答，而不是报 404。
            known = self.runtime.get_turn_or_none(turn_id)
            if known is not None:
                return known
            return _turn_from_run_record(
                existing,
                user_message=message.strip(),
                task_type=classification.task_type,
            )

        turn_input = _TurnInput(
            str(root),
            repo_id,
            repo_fingerprint,
            validation_profile.strip(),
            limit,
            contains_workspace_state,
        )
        task = AgentRunTask(
            task_id=f"task-{uuid4().hex[:16]}",
            session_id=session_id,
            turn_id=turn_id,
            run_id=f"chat-run-{uuid4().hex[:16]}",
            task_kind=TASK_AGENT_RUN,
            idempotency_key=client_turn_id or turn_id,
            policy_version=CHAT_RUN_POLICY_VERSION,
            deadline_epoch_ms=int(time.time() * 1000) + DEFAULT_RUN_DEADLINE_MS,
            options=RunRequestOptions(
                validation_profile=validation_profile.strip(),
                result_limit=limit,
                show_debug_reasoning=show_debug_reasoning,
            ),
        )
        # 受理即事实：用户这一轮的消息在返回 202 之前写进会话。Worker 只凭
        # session_id/run_id 重建执行，读到的必须已经是这句话；晚一步写，
        # 另一个进程的 Worker 就会答上一轮的问题。
        previous_compactions = context.memory.compaction_count
        context = self.session_service.append_turn(
            context,
            role="user",
            content=message.strip(),
            repo_fingerprint=repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        if context.memory.compaction_count > previous_compactions:
            self.runtime.emit(
                task.run_id,
                CONTEXT_COMPACTED,
                {
                    "compaction_count": str(context.memory.compaction_count),
                    "through_sequence": str(context.memory.compacted_through_sequence),
                },
            )
        record = self.agent_run_dispatcher.dispatch(task)
        if record.status != QUEUED:
            raise ChatDispatchError(record.error_class or "DISPATCH_FAILED")

        turn = self._accepted_turns.pop(task.task_id, None)
        if turn is None:
            turn = self.runtime.get_turn_or_none(turn_id)
        if turn is None:
            raise ChatDispatchError("DISPATCH_FAILED")
        self._turn_inputs[turn.turn_id] = turn_input
        self.runtime.emit(
            turn.run_id,
            TASK_QUEUED,
            {
                "turn_id": turn.turn_id,
                "run_id": turn.run_id,
                "task_id": task.task_id,
                "task_kind": task.task_kind,
                "status": QUEUED,
            },
        )
        self.runtime.emit(
            turn.run_id,
            INTENT_CLASSIFIED,
            {
                "task_type": classification.task_type,
                "confidence": f"{classification.confidence:.2f}",
                "rule": classification.rule,
                "previous_task_type": (
                    context.active_goal.task_type if context.active_goal is not None else "none"
                ),
                "goal_action": _goal_action(classification.task_type, context.active_goal),
            },
        )
        return turn

    def _schedule_turn(self, task: AgentRunTask) -> str:
        """进程内投递：先把这一轮的事实读回来，再交给本进程的线程池执行。

        这里读一遍不是为了把上下文送给 Worker——Worker 自己会再读一遍，它只能
        依据事实层工作。这一步是为了构造用户看到的状态投影：API 若凭空编一个
        task_type 或消息，界面显示的就会是另一轮的内容。
        """

        record = self.agent_run_store.get_run(task.run_id)
        if record is None:
            raise TransportRejected(f"Run {task.run_id} 不在事实层，无法投递")
        try:
            context = self.agent_run_context_loader.load(task, record)
        except AgentRunContextError as error:
            raise TransportRejected(error.error_class) from error
        if task.task_kind in {TASK_RESUME_AFTER_APPROVAL, TASK_RESUME_AFTER_VALIDATION}:
            return self._resume_existing_turn(task, context)
        try:
            turn = self.runtime.submit(
                session_id=task.session_id,
                task_type=context.task_type,
                user_message=context.user_message,
                show_debug_reasoning=context.options.show_debug_reasoning,
                worker=lambda current, debug: self._execute_loaded_run(task, current, debug),
                turn_id=task.turn_id,
                run_id=task.run_id,
                task_id=task.task_id,
            )
        except ChatRuntimeError as error:
            raise TransportRejected(str(error)) from error
        self._accepted_turns[task.task_id] = turn
        return turn.turn_id

    def _execute_loaded_run(
        self, task: AgentRunTask, turn: ChatTurn, show_debug_reasoning: bool
    ) -> ChatExecution:
        """把一个执行结果交给 AgentRunWorker，让事实层和用户没有第二种说法。

        Worker 负责领取租约、重建上下文、写状态；这里只回答「这一轮该跑什么」。
        分流只看任务类型：新建 Run 跑一次对话，续跑只做已经批准的那个补丁。
        """

        outcome = self.agent_run_worker.handle(
            task, runner=self._runner_for(task, turn, show_debug_reasoning)
        )
        if outcome.execution is None:
            # Worker 自己终止了这一轮。事实层里的状态（UNKNOWN、MANUAL_REQUIRED、
            # FAILED）必须原样变成用户看到的说法：一律显示「失败」，「需要人工对账」
            # 这层信息就在界面上消失了。
            return ChatExecution(
                status=_CHAT_STATUS_BY_RUN_STATUS.get(outcome.status, CHAT_FAILED),
                assistant_message=_stopped_message(outcome.status),
                error=outcome.error_class,
            )
        return outcome.execution

    def run_agent_task(self, task: AgentRunTask) -> AgentRunOutcome:
        """Worker 进程入口：消息本身就是执行，不经过本进程的线程池。

        与进程内路径唯一的区别是「这一轮是谁受理的」：消息可能来自另一个进程，
        本地运行时里没有这一轮，执行体需要的 turn 就从事实 Store 重建。重建只用
        标识、用户原话和公开状态，别的一概不搬。
        """

        record = self.agent_run_store.get_run(task.run_id)
        if record is None:
            # 任务指向一个不存在的 Run。凭一条孤立消息去跑模型，只会产出一份
            # 无法和任何事实对账的结果。
            return AgentRunOutcome(
                run_id=task.run_id,
                status=MANUAL_REQUIRED,
                execution=None,
                error_class=RUN_RECORD_MISSING,
                claimed=False,
            )

        def runner(loaded: AgentRunContext) -> ChatExecution:
            turn = self._adopt_turn(record, loaded)
            return self._runner_for(task, turn, loaded.options.show_debug_reasoning)(loaded)

        return self.agent_run_worker.handle(task, runner=runner)

    def _adopt_turn(self, record: AgentRunRecord, context: AgentRunContext) -> ChatTurn:
        """把事实层里的这一轮收进本地运行时。

        Worker 进程和 API 进程各有一个运行时。执行体会按 turn_id 读写本地运行时
        的状态，所以 Worker 侧必须先有这一轮；它的内容全部来自事实层，而不是从
        另一个进程的内存里搬过来。
        """

        return self.runtime.adopt_turn(
            _turn_from_run_record(
                record, user_message=context.user_message, task_type=context.task_type
            )
        )

    def resolve_run_id(self, turn_id: str) -> str | None:
        """这一轮对应的 run_id：本地运行时优先，其次事实层。

        SSE 断线重连和 API 重启后都要能接着读事件，所以「本进程没见过这一轮」
        不能等于「没有这一轮」。
        """

        turn = self.runtime.get_turn_or_none(turn_id)
        if turn is not None:
            return turn.run_id
        record = self.agent_run_store.find_run_by_turn(turn_id)
        return record.run_id if record is not None else None

    def run_is_settled(self, run_id: str) -> bool:
        """这一轮还会不会自动产生新事件。本地运行时不知道时回退到事实层。

        事件流靠它决定「还等不等新事件」：答错会留下一条永远挂着的连接，或者
        提前关掉一条还有事件要写的连接。等审批与等校验都不算停下——前者等用户
        动作，后者有 Worker 在推。
        """

        turn = self.runtime.get_turn_or_none_by_run_id(run_id)
        if turn is not None:
            return turn.status in SETTLED_CHAT_STATUSES
        record = self.agent_run_store.get_run(run_id)
        return record is None or record.status not in OPEN_AGENT_RUN_STATUSES

    def _adopt_turn(self, record: AgentRunRecord, context: AgentRunContext) -> ChatTurn:
        """把事实层里的这一轮收进本地运行时。

        Worker 进程和 API 进程各有一个运行时。执行体会按 turn_id 读写本地运行时
        的状态，所以 Worker 侧必须先有这一轮；它的内容全部来自事实层，而不是从
        另一个进程的内存里搬过来。
        """

        return self.runtime.adopt_turn(
            _turn_from_run_record(
                record, user_message=context.user_message, task_type=context.task_type
            )
        )

    def _runner_for(
        self, task: AgentRunTask, turn: ChatTurn, show_debug_reasoning: bool
    ) -> Callable[[AgentRunContext], ChatExecution]:
        """这一轮该跑什么：分流只看任务类型。

        新建 Run 跑一次对话，审批续跑只做已经批准的那个补丁，校验续跑只负责把
        校验结论翻译成终态。进程内路径和 Worker 进程路径共用它，语义不分叉。
        """

        if task.task_kind == TASK_RESUME_AFTER_APPROVAL:
            return lambda loaded: self._resume_change(
                turn, self._turn_input_for(loaded), task
            )
        if task.task_kind == TASK_RESUME_AFTER_VALIDATION:
            return lambda loaded: self._finalize_validation(
                turn, loaded, show_debug_reasoning, task
            )
        return lambda loaded: self._execute_turn(
            turn,
            self._turn_input_for(loaded),
            loaded.classification,
            show_debug_reasoning,
        )

    def _finalize_validation(
        self,
        turn: ChatTurn,
        loaded: AgentRunContext,
        show_debug_reasoning: bool,
        task: AgentRunTask,
    ) -> ChatExecution:
        """校验结论的执行体：通过就收尾，失败按登记过的预算决定修还是交人。

        这一步同样不跑沙箱：结论已经由 ValidationWorker 写进事实，这里只负责把
        事实翻译成用户能看到的终态。
        """

        record = self.agent_run_store.get_run(task.run_id)
        patch_id = (record.patch_id or "").strip() if record is not None else ""
        if not patch_id:
            raise ValueError("校验续跑缺少补丁标识")
        turn_input = self._turn_input_for(loaded)
        result = self.change_service.get_result(task.run_id, patch_id)
        if result is None or result.status == "WAITING_VALIDATION":
            # 固定校验没有给出结论（环境不可用或 Worker 崩了）。这既不是「校验
            # 通过」也不是「代码有问题」：只能停下来让人确认隔离 workspace。
            message = "固定校验没有给出结论：后台校验没有完成，需要人工确认隔离 workspace。"
            return ChatExecution(
                status=CHAT_MANUAL_REQUIRED,
                assistant_message=message,
                result={
                    "kind": "change_validation_inconclusive",
                    "validation": result.validation if result is not None else None,
                },
                error="VALIDATION_INCONCLUSIVE",
            )
        if result.status == "COMPLETED":
            return self._finish_change_turn(
                turn,
                turn_input,
                result,
                (
                    "修改已应用；当前开发模式跳过了 Docker 固定校验。"
                    if result.validation and result.validation.get("skipped")
                    else "修改已应用并完成校验。"
                ),
            )
        failure = self._review_required_result(task.run_id)
        if failure is not None and not _is_sandbox_failure(failure.validation or {}):
            # 修复轮要带着失败摘要去跑：摘要由 _repair_task 从这一条事实生成。
            self._change_recoveries[turn.session_id] = _ChangeRecovery(
                run_id=failure.run_id,
                patch_id=failure.patch_id,
                validation=dict(failure.validation or {}),
            )
            if self._repair_budget_left(task.run_id):
                # 先把预算记下来再动手：反过来的顺序在崩溃时会变成无限重修。
                self.change_service.record_repair_round(task.run_id, patch_id=patch_id)
                self.runtime.emit(
                    turn.run_id,
                    REPAIR_ATTEMPTED,
                    {"stage": "auto_repair", "status": "running"},
                )
                return self._execute_turn(
                    replace(turn, user_message=_auto_repair_message(turn.user_message)),
                    turn_input,
                    loaded.classification,
                    show_debug_reasoning,
                )
        message = f"修改流程结束，状态为 {result.status}。"
        if result.status == "REVIEW_REQUIRED":
            message = "固定检查没有通过，需要人工处理。"
        return self._finish_change_turn(turn, turn_input, result, message)

    def _remember_change_recovery(self, session_id: str, result) -> None:
        """把「这一轮还要不要人来决定」记在会话上，供用户追问时接续。"""

        if result.status == "REVIEW_REQUIRED" and result.validation is not None:
            self._change_recoveries[session_id] = _ChangeRecovery(
                run_id=result.run_id,
                patch_id=result.patch_id,
                validation=dict(result.validation),
            )
            return
        if result.status == "COMPLETED":
            self._change_recoveries.pop(session_id, None)

    def _finish_change_turn(
        self, turn: ChatTurn, turn_input: _TurnInput, result, assistant_message: str
    ) -> ChatExecution:
        """收尾一次修改轮：把助手答复写进会话，再返回用户可见的结果。"""

        self._remember_change_recovery(turn.session_id, result)

        context = self.session_service.get_or_create_session(
            session_id=turn.session_id,
            scope=TenantScope(),
            repo_id=turn_input.repo_id,
            repo_fingerprint=turn_input.repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        self.session_service.append_turn(
            context,
            role="assistant",
            content=assistant_message,
            repo_fingerprint=turn_input.repo_fingerprint,
            index_version=INDEX_VERSION,
        )
        return _change_result_execution(result, assistant_message=assistant_message)

    def _register_validation(self, turn: ChatTurn, result) -> ChatExecution:
        """把这一轮的固定校验登记出去，然后立刻结束当前 attempt。"""

        profile, workspace_path = self.change_service.registered_validation(
            result.run_id, result.patch_id
        )
        record = self.agent_run_store.get_run(result.run_id)
        if record is None:
            raise ValueError("找不到这次修改的 Run 事实，无法登记校验")
        self.validation_coordinator.register(
            record,
            patch_id=result.patch_id,
            profile=profile,
            workspace_path=workspace_path,
        )
        return ChatExecution(
            status=CHAT_WAITING_VALIDATION,
            assistant_message="补丁已应用到隔离 workspace，正在等待固定校验。",
            result={
                "kind": "change_pending_validation",
                "validation": {"profile": profile, "status": "QUEUED"},
            },
        )

    def _schedule_validation(self, task: TaskEnvelope) -> str:
        """投递一条校验任务：立刻返回，沙箱在别的线程里跑。

        Agent Worker 在登记完之后马上结束自己的 attempt，所以这里绝不能同步跑
        Docker——那等于把 Agent 队列又堵回去。
        """

        self._validation_executor.submit(self._run_validation_task, task.task_id)
        return task.task_id

    def _run_validation_task(self, task_id: str) -> None:
        outcome = self.validation_worker.handle(task_id)
        self.validation_coordinator.after_validation(outcome)

    def _turn_input_for(self, context: AgentRunContext) -> _TurnInput:
        """执行前定下「这次在哪个目录里干活」。

        会话事实里存的是源仓库（write-once）；一旦这个会话此前已经产生过隔离
        workspace，进程里记着的是更近的事实。Worker 重建时优先用后者，否则模型
        会去看源仓库，而改动其实都落在隔离 workspace 里。
        """

        base = _turn_input_from_context(context)
        with self._session_workspace_lock:
            workspace = self._session_workspaces.get(context.session_id)
        if workspace is None or workspace.effective_root == workspace.source_root:
            return base
        return replace(
            base,
            repository_root=str(workspace.effective_root),
            contains_workspace_state=True,
        )

    def _review_required_result(self, run_id: str):
        finder = getattr(self.change_service, "last_review_required", None)
        if not callable(finder):
            return None
        return finder(run_id)

    def _repair_budget_left(self, run_id: str) -> bool:
        """自动修复还有没有预算。预算记在事实里，进程重启也算数。"""

        counter = getattr(self.change_service, "repair_rounds", None)
        if not callable(counter):
            return False
        return counter(run_id) < MAX_CHANGE_REPAIR_ROUNDS

    def _resume_existing_turn(self, task: AgentRunTask, context: AgentRunContext) -> str:
        """续跑投递：复用停在等待审批的那一轮，而不是新开一轮对话。

        审批续跑和一次新消息共用 run_id / turn_id，区别只在执行体，所以这里
        不能走 submit——submit 会因为同一会话已有活跃轮次而拒绝。真正把状态从
        等待审批推走的动作在 runtime.resume 里，重复投递会在那里输掉。
        """

        try:
            turn = self.runtime.resume(
                task.turn_id,
                show_debug_reasoning=context.options.show_debug_reasoning,
                worker=lambda current, debug: self._execute_loaded_run(task, current, debug),
            )
        except ChatRuntimeError as error:
            raise TransportRejected(str(error)) from error
        return turn.turn_id

    def _resume_change(
        self, turn: ChatTurn, turn_input: _TurnInput, task: AgentRunTask
    ) -> ChatExecution:
        """续跑执行体：凭 run_id 从事实层取回补丁标识与一次性审批令牌再 apply。

        令牌既不放队列载荷，也不留在 API 进程内存里，Worker 侧读事实层拿。
        这样 API 重启或换一台机器跑 Worker，续跑照样能执行，也不需要把秘密
        塞进消息队列。
        """

        record = self.agent_run_store.get_run(task.run_id)
        patch_id = (record.patch_id or "").strip() if record is not None else ""
        token = (record.approval_token or "").strip() if record is not None else ""
        if not patch_id or not token:
            raise ValueError("续跑缺少补丁标识或审批令牌")
        return self._apply_change(
            turn, turn_input, patch_id, token, defer_validation=True
        )

    def _resolve_session_repository(
        self,
        session_id: str,
        repository_root: str,
        *,
        refresh_fingerprint: bool = True,
    ) -> tuple[Path, str, str, bool]:
        if refresh_fingerprint:
            requested_root, requested_repo_id, requested_fingerprint = _repository_metadata(
                repository_root
            )
        else:
            requested_root, requested_repo_id = _repository_identity(repository_root)
            requested_fingerprint = _unscanned_repository_fingerprint(requested_root)
        with self._session_workspace_lock:
            workspace = self._session_workspaces.get(session_id)
        if workspace is None:
            return requested_root, requested_repo_id, requested_fingerprint, False
        if requested_root not in {workspace.source_root, workspace.effective_root}:
            raise ValueError("session_id 已绑定其他仓库")
        effective_root = workspace.effective_root
        if not effective_root.is_dir():
            effective_root = workspace.source_root
        if refresh_fingerprint:
            effective_root, _, effective_fingerprint = _repository_metadata(
                str(effective_root)
            )
        else:
            effective_root, _ = _repository_identity(str(effective_root))
            effective_fingerprint = _unscanned_repository_fingerprint(effective_root)
        return (
            effective_root,
            workspace.repo_id,
            effective_fingerprint,
            effective_root != workspace.source_root,
        )

    def _remember_session_workspace(
        self, session_id: str, repo_id: str, source_root: Path
    ) -> None:
        resolved = source_root.resolve()
        with self._session_workspace_lock:
            first_seen = session_id not in self._session_workspaces
            self._session_workspaces.setdefault(
                session_id, _SessionWorkspace(repo_id, resolved, resolved)
            )
        if not first_seen:
            return
        # 会话绑定的仓库根要落成持久事实：受理与执行分开之后，另一个进程的
        # Worker 只有 session_id，没有这张进程内的表。
        self.session_service.bind_repository_root(session_id, str(resolved))

    def _remember_effective_workspace(
        self,
        session_id: str,
        repo_id: str,
        source_root: Path,
        effective_root: Path,
    ) -> None:
        with self._session_workspace_lock:
            current = self._session_workspaces.get(session_id)
            if current is not None and current.repo_id != repo_id:
                raise ValueError("session_id 已绑定其他仓库")
            source = current.source_root if current is not None else source_root.resolve()
            self._session_workspaces[session_id] = _SessionWorkspace(
                repo_id,
                source,
                effective_root.resolve(),
            )

    def approve_turn(self, turn_id: str) -> ChatTurn:
        """登记一次审批续跑，不在 API 线程里应用补丁。

        这里只做三件事：确认这一轮确实停在等待审批、把审批换成一次性令牌、
        把续跑任务写进事实层并投给后台 Worker。真正写 workspace 的 apply /
        reconcile / validation 全部发生在 Worker 里——API 线程一旦开始改磁盘，
        超时、断线或重复点击都会变成「不知道改到哪儿了」。

        同一轮的重复点击由 _approvals_in_flight 挡住：第二下在动令牌之前就
        失败，而不是等 Worker 侧才发现自己输了。
        """

        with self._approval_lock:
            if turn_id in self._approvals_in_flight:
                raise ValueError("这一轮的审批正在处理中")
            self._approvals_in_flight.add(turn_id)
        try:
            turn = self.runtime.get_turn(turn_id)
            if turn.status != CHAT_WAITING_APPROVAL:
                raise ValueError("当前轮次不在等待审批状态")
            record = self.agent_run_store.get_run(turn.run_id)
            if record is None:
                raise ValueError("当前轮次没有对应的 Run 事实，无法登记续跑")
            if record.status != WAITING_APPROVAL:
                raise ValueError(f"Run 当前是 {record.status}，不接受审批")
            preview = _preview_from_result(turn.result)
            patch_id = str(preview["patch_id"])
            token = self.change_service.approve(turn.run_id, patch_id)
            if not callable(getattr(self.change_service, "add_event_sink", None)):
                self.runtime.emit(
                    turn.run_id,
                    APPROVAL_GRANTED,
                    {"patch_id": patch_id, "source": "chat_ui"},
                )
            task = AgentRunTask(
                task_id=f"task-{uuid4().hex[:16]}",
                session_id=turn.session_id,
                turn_id=turn.turn_id,
                run_id=turn.run_id,
                task_kind=TASK_RESUME_AFTER_APPROVAL,
                idempotency_key=f"{record.idempotency_key}:approval:{patch_id}",
                policy_version=record.policy_version,
                deadline_epoch_ms=int(time.time() * 1000) + DEFAULT_RUN_DEADLINE_MS,
                options=record.options,
            )
            try:
                self.agent_run_dispatcher.dispatch_continuation(
                    task,
                    expected_status=WAITING_APPROVAL,
                    patch_id=patch_id,
                    approval_token=token,
                )
            except TransportRejected as error:
                raise ChatDispatchError(str(error)) from error
            return self.runtime.get_turn(turn_id)
        finally:
            with self._approval_lock:
                self._approvals_in_flight.discard(turn_id)

    def cancel_turn(self, turn_id: str) -> ChatTurn:
        turn = self.runtime.get_turn(turn_id)
        if turn.status == CHAT_WAITING_APPROVAL:
            self.change_service.cancel(turn.run_id)
            return self.runtime.cancel_waiting(turn_id)
        raise ValueError("运行中的聊天任务只能通过后端自然收敛，暂不支持强制终止")

    def _enforce_session_budget(
        self,
        turn: ChatTurn,
        context: SessionContext,
        *,
        scene: str,
        repo_fingerprint: str,
    ) -> SessionContext:
        """这轮请求发出之前，按 token 预算压缩 Session 历史。

        Q-010 第一版的 95% 触发点乘的是原始窗口（121600），比输出预留后的
        输入上限（87040）还高，而且没有任何调用方——压缩实际只由「保留轮数」
        驱动。这里改成按可用输入预算计算，并且在发请求之前真正执行。

        重试次数取策略的 ``compaction_retries``。仍然超预算时不再静默重试，
        交给 Gateway 的 overflow guard 拒收并留下事件：继续压缩一个已经压到
        保留窗口的历史，只会既丢掉上下文又换不来空间。
        """
        budget = resolve_route_budget(scene)
        policy = budget.policy
        limit = budget.session_history_budget
        surface = estimate_tokens(_render_session_context(context))
        if surface < limit:
            return context
        for _ in range(policy.compaction_retries + 1):
            compacted, result = self.session_service.compact_session(
                context,
                repo_fingerprint=repo_fingerprint,
                index_version=INDEX_VERSION,
                max_context_tokens=policy.retained_recent_tokens,
            )
            if not result.dropped_turn_sequences:
                self.runtime.emit(
                    turn.run_id,
                    CONTEXT_COMPACTED,
                    {
                        "trigger": "session_history_budget",
                        "outcome": "no_progress",
                        "surface_tokens": str(surface),
                        "limit_tokens": str(limit),
                        "failure_class": str(result.failure_class or ""),
                    },
                )
                return compacted
            context = compacted
            after = estimate_tokens(_render_session_context(context))
            self.runtime.emit(
                turn.run_id,
                CONTEXT_COMPACTED,
                {
                    "trigger": "session_history_budget",
                    "outcome": "compacted",
                    "surface_tokens_before": str(surface),
                    "surface_tokens_after": str(after),
                    "limit_tokens": str(limit),
                    "retained_tokens": str(policy.retained_recent_tokens),
                    "dropped_turns": str(len(result.dropped_turn_sequences)),
                    "compaction_count": str(context.memory.compaction_count),
                    "boundary_id": str(result.boundary_id or ""),
                },
            )
            if after < limit:
                return context
            surface = after
        return context

    def _execute_turn(
        self,
        turn: ChatTurn,
        turn_input: _TurnInput,
        classification: ChatTaskClassification,
        show_debug_reasoning: bool,
    ) -> ChatExecution:
        context: SessionContext | None = None
        try:
            if classification.task_type in {"explain", "change"}:
                refreshed_root, _, refreshed_fingerprint = _repository_metadata(
                    turn_input.repository_root
                )
                turn_input = replace(
                    turn_input,
                    repository_root=str(refreshed_root),
                    repo_fingerprint=refreshed_fingerprint,
                )
            context = self.session_service.get_or_create_session(
                session_id=turn.session_id,
                scope=TenantScope(),
                repo_id=turn_input.repo_id,
                repo_fingerprint=turn_input.repo_fingerprint,
                index_version=INDEX_VERSION,
            )
            self.runtime.emit(
                turn.run_id,
                SESSION_LOADED,
                {
                    "cache_hit": str(context.cache_hit).lower(),
                    "cache_fallback": str(context.cache_fallback).lower(),
                    "recent_turns": str(len(context.memory.recent_turns)),
                    "summary_present": str(bool(context.memory.summary)).lower(),
                },
            )
            # 用户这一轮的消息在受理时就已落库（见 submit_turn），压缩结论也在
            # 那时写进了事件；这里不再追加，否则同一个问题会进会话两次。
            goal_type = classification.task_type
            request_scene = "change-plan" if goal_type == "change" else "explain"
            if goal_type in {"change", "explain"}:
                mode = MODE_ISOLATED_WRITE if goal_type == "change" else MODE_READ_ONLY
                context = self.session_service.continue_or_create_goal(
                    context,
                    user_goal=turn.user_message,
                    task_type=goal_type,
                    mode=mode,
                    validation_profile=(
                        turn_input.validation_profile if goal_type == "change" else None
                    ),
                    start_new=(
                        context.active_goal is None or context.active_goal.task_type != goal_type
                    ),
                    repo_fingerprint=turn_input.repo_fingerprint,
                    index_version=INDEX_VERSION,
                )
            context = self._enforce_session_budget(
                turn,
                context,
                scene=request_scene,
                repo_fingerprint=turn_input.repo_fingerprint,
            )
            current_user_sequence = (
                context.memory.recent_turns[-1].sequence
                if context.memory.recent_turns
                else None
            )
            self.runtime.emit(
                turn.run_id,
                CONTEXT_ASSEMBLED,
                {
                    "history_turns": str(len(context.memory.recent_turns)),
                    "history_tokens": str(estimate_tokens(_render_session_context(context))),
                    "history_budget_tokens": str(
                        resolve_route_budget(request_scene).session_history_budget
                    ),
                    "summary_present": str(bool(context.memory.summary)).lower(),
                    "active_goal": str(context.active_goal is not None).lower(),
                },
            )
            if goal_type == "scope_redirect":
                execution = self._execute_scope_redirect(
                    turn, has_active_code_goal=context.active_goal is not None
                )
            else:
                bound_model = _RunBoundModel(
                    self._model_factory(),
                    history=_render_session_context(
                        context, exclude_sequence=current_user_sequence
                    ),
                    turn_id=turn.turn_id,
                    runtime=self.runtime,
                    show_debug_reasoning=show_debug_reasoning,
                    repo_id=turn_input.repo_id,
                    repo_fingerprint=turn_input.repo_fingerprint,
                    contains_workspace_state=(
                        turn_input.contains_workspace_state
                        or classification.task_type == "change"
                    ),
                    compaction_boundary_id=context.memory.compaction_boundary_id,
                )
                if goal_type == "change":
                    execution = self._execute_change(turn, turn_input, bound_model)
                elif goal_type == "general_chat":
                    execution = self._execute_general_chat(turn, bound_model)
                elif goal_type == "clarify":
                    execution = self._execute_clarify(turn)
                else:
                    execution = self._execute_explain(turn, turn_input, bound_model)
            assistant = execution.assistant_message
            if assistant:
                self.session_service.append_turn(
                    context,
                    role="assistant",
                    content=assistant,
                    repo_fingerprint=turn_input.repo_fingerprint,
                    index_version=INDEX_VERSION,
                )
            return execution
        except (
            ModelCallError,
            ModelResponseError,
            ModelConfigurationError,
            GatewayError,
            ValueError,
            OSError,
        ) as error:
            if context is not None:
                try:
                    self.session_service.append_turn(
                        context,
                        role="assistant",
                        content="本轮执行失败，请检查事件详情。",
                        repo_fingerprint=turn_input.repo_fingerprint,
                        index_version=INDEX_VERSION,
                    )
                except (ValueError, OSError):
                    pass
            safe = redact_sensitive(str(error)).strip()[:240] or type(error).__name__
            return ChatExecution(
                status=CHAT_FAILED,
                assistant_message=f"本轮执行失败：{safe}",
                error=safe,
            )

    def _execute_general_chat(
        self, turn: ChatTurn, model: _RunBoundModel
    ) -> ChatExecution:
        from codeinsight.prompts.general_chat import PROMPT_VERSION, build_general_chat_prompt

        self.runtime.emit(
            turn.run_id,
            MODEL_GENERATING,
            {"route": "general_chat", "status": "started"},
        )
        system_prompt, user_prompt = build_general_chat_prompt(turn.user_message)
        completion = model.complete_text(system_prompt, user_prompt)
        answer = completion.content.strip()
        if not answer:
            raise ModelResponseError("普通对话返回空回答")
        self.runtime.emit(
            turn.run_id,
            MODEL_GENERATING,
            {"route": "general_chat", "status": "completed"},
        )
        self.runtime.emit(turn.run_id, ANSWER_READY, {"kind": "general_chat"})
        payload = {
            "kind": "general_chat",
            "outcome": "answered",
            "route": "general_chat",
            "model": completion.model,
            "prompt_version": PROMPT_VERSION,
            "usage": {
                "input_tokens": completion.input_tokens or 0,
                "output_tokens": completion.output_tokens or 0,
            },
            "observability": _usage_payload(
                self.runtime.event_log.read_events(turn.run_id),
                fallback_input=completion.input_tokens or 0,
                fallback_output=completion.output_tokens or 0,
            ),
        }
        return ChatExecution(
            status=CHAT_COMPLETED,
            assistant_message=answer,
            result=payload,
        )

    def _execute_clarify(self, turn: ChatTurn) -> ChatExecution:
        message = (
            "我还不能确定你希望我做什么。请说明具体文件、函数或目标，"
            "例如“解释 workflow.py”或“修复这个函数”；如果是在追问上一轮，"
            "也可以说清楚要继续分析还是修改。"
        )
        self.runtime.emit(
            turn.run_id,
            INTENT_CLASSIFIED,
            {"execution_route": "clarify", "confidence": "0.60", "fallback": "false"},
        )
        self.runtime.emit(turn.run_id, ANSWER_READY, {"kind": "clarify"})
        return ChatExecution(
            status=CHAT_COMPLETED,
            assistant_message=message,
            result={
                "kind": "clarify",
                "outcome": "clarification_required",
                "route": "clarify",
            },
        )

    def _execute_scope_redirect(
        self, turn: ChatTurn, *, has_active_code_goal: bool
    ) -> ChatExecution:
        """自然承接业务外闲聊，但不调用模型、检索仓库或创建新 Goal。"""
        message = build_scope_redirect_message(
            has_active_code_goal=has_active_code_goal,
        )
        self.runtime.emit(
            turn.run_id,
            ANSWER_READY,
            {"kind": "scope_redirect", "model_called": "false"},
        )
        return ChatExecution(
            status=CHAT_COMPLETED,
            assistant_message=message,
            result={
                "kind": "scope_redirect",
                "outcome": "redirected",
                "route": "scope_redirect",
                "reason": "out_of_scope",
                "model_called": False,
                "prompt_version": SCOPE_REDIRECT_VERSION,
            },
        )

    def _execute_explain(
        self, turn: ChatTurn, turn_input: _TurnInput, model: _RunBoundModel
    ) -> ChatExecution:
        self.runtime.emit(turn.run_id, RETRIEVAL_STARTED, {"route": "auto"})
        router_result = route_question(turn.user_message, complete=model.complete)
        self.runtime.emit(
            turn.run_id,
            INTENT_CLASSIFIED,
            {
                "execution_route": router_result.plan.execution_route,
                "confidence": f"{router_result.plan.confidence:.2f}",
                "fallback": str(router_result.used_fallback).lower(),
            },
        )
        # 2026-09-11：linear 也归并进只读 Tool Loop 后，父进程不再需要自己
        # 构造 Embedding / Rerank——检索在 MCP Server 子进程里按环境变量自建。
        # 保留原实现以便回退：
        # embedding_model = (
        #     self._embedding_factory()
        #     if router_result.plan.execution_route != "insufficient"
        #     else None
        # )
        # reranker = (
        #     self._reranker_factory()
        #     if router_result.plan.execution_route != "insufficient"
        #     else None
        # )
        if uses_code_understanding(router_result.plan.execution_route):
            # explain 只有这一条路径：linear 与 agent 都在这里收敛到只读 Tool Loop。
            result, tool_loop_payload = self._run_code_understanding(
                turn, turn_input, model, router_result
            )
        else:
            # 只剩 insufficient：不检索、不调用工具，直接返回证据不足。
            result = auto_answer_repository(
                turn_input.repository_root,
                router_result=router_result,
                generate=model.generate,
                limit=turn_input.limit,
            )
            tool_loop_payload = None
        self.runtime.emit(
            turn.run_id,
            RETRIEVAL_FINISHED,
            {"outcome": result.outcome, "citations": str(len(result.citations))},
        )
        self.runtime.emit(
            turn.run_id,
            MODEL_GENERATING,
            {"route": router_result.plan.execution_route, "status": "completed"},
        )
        self.runtime.emit(
            turn.run_id,
            ANSWER_READY,
            {"outcome": result.outcome, "citations": str(len(result.citations))},
        )
        payload = _auto_payload(result)
        if tool_loop_payload is not None:
            payload["tool_loop"] = tool_loop_payload
        payload["observability"] = _usage_payload(
            self.runtime.event_log.read_events(turn.run_id),
            fallback_input=result.input_tokens + result.router_input_tokens,
            fallback_output=result.output_tokens + result.router_output_tokens,
        )
        return ChatExecution(
            status=CHAT_COMPLETED,
            assistant_message=result.answer,
            result=payload,
        )

    def _run_code_understanding(
        self,
        turn: ChatTurn,
        turn_input: _TurnInput,
        model: _RunBoundModel,
        router_result,
    ) -> tuple[AutoAnswer, dict[str, object]]:
        """跑一次只读代码理解 Tool Loop，并映射回公开的 AutoAnswer 形状。

        实际执行委托给 ``application/code_understanding_route.py`，
        与 HTTP、CLI 三个入口共用同一实现，避免各自漂移。
        """
        loop_result = run_code_understanding_answer(
            turn_input.repository_root,
            turn.user_message,
            model=model,
            run_id=turn.run_id,
            event_log=self.runtime.event_log,
            mcp_client_factory=self._mcp_client_factory,
            emit=lambda event_type, detail: self.runtime.emit(
                turn.run_id, event_type, detail
            ),
        )
        return (
            to_auto_answer(loop_result, router_result, model_name=model.model),
            loop_result.as_dict(),
        )

    def _execute_change(
        self, turn: ChatTurn, turn_input: _TurnInput, model: _RunBoundModel
    ) -> ChatExecution:
        preflight_checker = getattr(self.change_service, "preflight_validation", None)
        if callable(preflight_checker):
            preflight = preflight_checker(turn_input.validation_profile)
            preflight_payload = {
                "profile": str(getattr(preflight, "profile", turn_input.validation_profile)),
                "available": str(bool(getattr(preflight, "available", False))).lower(),
                "error_class": str(getattr(preflight, "error_class", None) or ""),
                "message_excerpt": str(getattr(preflight, "message_excerpt", ""))[:500],
                "development_mode": str(self.development_policy.enabled).lower(),
                "validation_skipped": str(
                    self.development_policy.skip_sandbox_validation
                ).lower(),
            }
            self.runtime.emit(turn.run_id, VALIDATION_PREFLIGHT, preflight_payload)
            if (
                not bool(getattr(preflight, "available", False))
                and not self.development_policy.skip_sandbox_validation
            ):
                error_class = preflight_payload["error_class"] or "SANDBOX_UNAVAILABLE"
                detail = preflight_payload["message_excerpt"] or (
                    "请检查 Docker Desktop、docker_engine 权限和校验镜像。"
                )
                message = f"修改前无法运行固定校验：{error_class}；{detail}"
                return ChatExecution(
                    status=CHAT_FAILED,
                    assistant_message=message,
                    result={
                        "kind": "validation_blocked",
                        "validation": {
                            **preflight_payload,
                            "next_action": "请先恢复 Sandbox，再重新提交修改。",
                        },
                    },
                    error=message,
                )

        recovery = self._change_recoveries.get(turn.session_id)
        if recovery is not None and _is_repair_continuation(turn.user_message):
            retry_validation = getattr(self.change_service, "retry_validation", None)
            if callable(retry_validation) and _is_sandbox_failure(recovery.validation):
                self.runtime.emit(
                    turn.run_id,
                    STEP_STARTED,
                    {"stage": "validation_retry", "status": "running"},
                )
                result = retry_validation(
                    recovery.run_id,
                    recovery.patch_id,
                    event_run_id=turn.run_id,
                )
                self._remember_change_recovery(turn.session_id, result)
                return _change_result_execution(
                    result,
                    assistant_message=(
                        "已重新运行固定校验，修改通过。"
                        if result.status == "COMPLETED"
                        else f"修改流程结束，状态为 {result.status}。"
                    ),
                )
        task = turn.user_message
        if recovery is not None and _is_repair_continuation(turn.user_message):
            task = _repair_task(turn.user_message, recovery)
        self.runtime.emit(
            turn.run_id,
            STEP_STARTED,
            {"stage": "tool_exploration", "status": "running"},
        )
        from codeinsight.infrastructure.mcp_client import StdioMCPClient

        workflow_config = None
        if self.development_policy.enabled:
            workflow_config = ToolLoopConfig(
                max_steps=12,
                deadline_seconds=120.0,
                max_tool_calls=96,
                repeated_error_limit=5,
            )
        try:
            with StdioMCPClient(turn_input.repository_root) as client:
                workflow = run_change_workflow_to_preview(
                    task,
                    turn_input.repository_root,
                    model=model,
                    mcp_client=client,
                    change_service=self.change_service,
                    run_id=turn.run_id,
                    validation_profile=turn_input.validation_profile,
                    config=workflow_config,
                    event_log=self.runtime.event_log,
                )
        except ValueError as error:
            if str(error) != "补丁没有实际变化":
                raise
            self.runtime.emit(
                turn.run_id,
                PATCH_REJECTED,
                {"reason": "no_effective_change", "stage": "preview"},
            )
            message = (
                "本轮未生成新的可应用补丁：模型提交的内容与当前隔离 workspace 相同。"
                "如果上一轮校验失败，请先查看校验摘要或恢复 Sandbox 后再继续。"
            )
            return ChatExecution(
                status=CHAT_FAILED,
                assistant_message=message,
                result={
                    "kind": "change_blocked",
                    "reason": "no_effective_change",
                    "validation": recovery.validation if recovery is not None else None,
                },
                error=message,
            )
        preview = workflow.preview.as_dict()
        preview["development_mode"] = self.development_policy.as_dict()
        preview["requires_approval"] = not self.development_policy.auto_approve_changes
        self.runtime.emit(
            turn.run_id,
            APPROVAL_REQUESTED,
            {
                "patch_id": workflow.preview.patch_id,
                "status": (
                    "auto_approved"
                    if self.development_policy.auto_approve_changes
                    else "waiting"
                ),
                "source": "dev_mode"
                if self.development_policy.auto_approve_changes
                else "chat_ui",
            },
        )
        if self.development_policy.auto_approve_changes:
            token = self.change_service.approve(
                turn.run_id,
                workflow.preview.patch_id,
                actor="dev_mode",
                source="dev_auto_approve",
            )
            return self._apply_change(
                turn,
                turn_input,
                workflow.preview.patch_id,
                token,
            )
        return ChatExecution(
            status=CHAT_WAITING_APPROVAL,
            assistant_message="已生成修改预览；请检查 diff，确认后再应用。",
            result={
                "kind": "change_preview",
                "preview": preview,
                "tool_loop_status": workflow.loop.status,
                "tool_steps": workflow.loop.steps,
            },
        )

    def _apply_change(
        self,
        turn: ChatTurn,
        turn_input: _TurnInput,
        patch_id: str,
        approval_token: str,
        *,
        defer_validation: bool = False,
    ) -> ChatExecution:
        self.runtime.emit(turn.run_id, STEP_STARTED, {"stage": "apply", "status": "running"})
        result = self.change_service.apply(
            turn.run_id, patch_id, approval_token, defer_validation=defer_validation
        )
        workspace_manager = getattr(self.change_service, "workspaces", None)
        managed = (
            workspace_manager.get(turn.run_id)
            if workspace_manager is not None
            else None
        )
        if managed is not None:
            self._remember_effective_workspace(
                turn.session_id,
                turn_input.repo_id,
                Path(managed.run.source_repo_path),
                Path(managed.run.workspace_path),
            )
        if defer_validation and result.status == "WAITING_VALIDATION":
            return self._register_validation(turn, result)
        if result.status == "COMPLETED" and result.validation and result.validation.get(
            "skipped"
        ):
            assistant_message = "修改已应用；当前开发模式跳过了 Docker 固定校验。"
        elif result.status == "COMPLETED":
            assistant_message = "修改已应用并完成校验。"
        else:
            assistant_message = f"修改流程结束，状态为 {result.status}。"
        return self._finish_change_turn(turn, turn_input, result, assistant_message)

    def _mirror_change_event(self, event) -> None:
        if event.event_type == "run_finished":
            return
        self.runtime.emit(event.run_id, event.event_type, dict(event.payload))


class _RunBoundModel:
    """把 Session 上下文和实时调试回调绑定到一次模型调用。"""

    def __init__(
        self,
        base: object,
        *,
        history: str,
        turn_id: str,
        runtime: ChatRuntime,
        show_debug_reasoning: bool,
        repo_id: str,
        repo_fingerprint: str,
        contains_workspace_state: bool,
        compaction_boundary_id: str | None = None,
    ) -> None:
        self._base = base
        self._history = history
        self._turn_id = turn_id
        self._runtime = runtime
        self._show_debug_reasoning = show_debug_reasoning
        self._cache_context = CacheContext(
            repo_id=repo_id,
            repo_fingerprint=repo_fingerprint,
            index_version=INDEX_VERSION,
            contains_workspace_state=contains_workspace_state,
            personalized=True,
        )
        self.model = getattr(base, "model", "unknown")
        self._compaction_boundary_id = compaction_boundary_id

    def complete(self, system_prompt: str, user_prompt: str):
        result = _call_with_run_context(
            getattr(self._base, "complete"),
            system_prompt,
            self._with_history(user_prompt),
            run_id=self._runtime.get_turn(self._turn_id).run_id,
            event_log=self._runtime.event_log,
            cache_context=self._cache_context,
            compaction_boundary_id=self._compaction_boundary_id,
        )
        self._publish_reasoning(result)
        return result

    def generate(self, system_prompt: str, user_prompt: str):
        from codeinsight.infrastructure.openai_chat import parse_model_answer

        completion = self.complete(system_prompt, user_prompt)
        return parse_model_answer(
            completion.content,
            model=completion.model,
            input_tokens=completion.input_tokens,
            output_tokens=completion.output_tokens,
        )

    def complete_text(self, system_prompt: str, user_prompt: str):
        method = getattr(self._base, "complete_text", None)
        if not callable(method):
            raise ModelConfigurationError("当前模型适配器不支持普通文本对话")
        result = _call_with_run_context(
            method,
            system_prompt,
            self._with_history(user_prompt),
            run_id=self._runtime.get_turn(self._turn_id).run_id,
            event_log=self._runtime.event_log,
            cache_context=self._cache_context,
            compaction_boundary_id=self._compaction_boundary_id,
        )
        self._publish_reasoning(result)
        return result

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
    ):
        enriched = list(messages)
        if self._history:
            for index in range(len(enriched) - 1, -1, -1):
                if enriched[index].get("role") == "user":
                    original = str(enriched[index].get("content", ""))
                    enriched[index] = {
                        **enriched[index],
                        "content": self._with_history(original),
                    }
                    break
        result = _call_with_run_context(
            getattr(self._base, "complete_with_tools"),
            tuple(enriched),
            tuple(tools),
            run_id=self._runtime.get_turn(self._turn_id).run_id,
            event_log=self._runtime.event_log,
            cache_context=self._cache_context,
            compaction_boundary_id=self._compaction_boundary_id,
        )
        self._publish_reasoning(result)
        return result

    def _with_history(self, user_prompt: str) -> str:
        if not self._history.strip():
            return user_prompt
        return (
            "[SESSION_CONTEXT]\n"
            + self._history
            + "\n\n[CURRENT_USER_REQUEST]\n"
            + user_prompt
        )

    def _publish_reasoning(self, result: object) -> None:
        if not self._show_debug_reasoning:
            return
        content = getattr(result, "reasoning_content", None)
        if isinstance(content, str) and content.strip():
            self._runtime.publish_reasoning(
                self._turn_id,
                content,
                model=str(getattr(result, "model", self.model)),
            )


def _call_with_run_context(
    method, *args, run_id: str, event_log, cache_context=None, compaction_boundary_id=None
):
    """兼容旧 FakeModel，同时给新版 Gateway 传递 run 绑定信息。"""
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, object] = {}
    if accepts_kwargs or "run_id" in parameters:
        kwargs["run_id"] = run_id
    if accepts_kwargs or "event_log" in parameters:
        kwargs["event_log"] = event_log
    if accepts_kwargs or "cache_context" in parameters:
        kwargs["cache_context"] = cache_context
    if accepts_kwargs or "compaction_boundary_id" in parameters:
        kwargs["compaction_boundary_id"] = compaction_boundary_id
    return method(*args, **kwargs)


def _repository_metadata(repository_root: str) -> tuple[Path, str, str]:
    root, repo_id = _repository_identity(repository_root)
    scan = scan_repository(root)
    digest = hashlib.sha256()
    for source in scan.files:
        digest.update(source.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.text.encode("utf-8"))
    return root, repo_id, digest.hexdigest()


def _repository_identity(repository_root: str) -> tuple[Path, str]:
    root = Path(repository_root).resolve()
    repo_id = hashlib.sha256(str(root).encode("utf-8")).hexdigest()
    return root, repo_id


def _unscanned_repository_fingerprint(root: Path) -> str:
    """为不需要代码事实的普通聊天生成稳定隔离标识，不读取仓库内容。"""
    return hashlib.sha256(f"unscanned:{root}".encode()).hexdigest()


def _render_session_context(
    context: SessionContext, *, exclude_sequence: int | None = None
) -> str:
    blocks: list[str] = []
    if context.memory.summary:
        blocks.append("summary: " + context.memory.summary)
    if context.active_goal is not None:
        blocks.append("active_goal: " + context.active_goal.user_goal)
    for turn in context.memory.recent_turns:
        if turn.sequence == exclude_sequence:
            continue
        blocks.append(f"turn {turn.sequence} {turn.role}: {turn.content}")
    return "\n".join(blocks)


def _session_payload(context: SessionContext) -> dict[str, object]:
    return {
        "session_id": context.session.session_id,
        "repo_id": context.session.repo_id,
        "index_version": INDEX_VERSION,
        "status": "READY",
        "summary": context.session.summary,
        "compacted_through_sequence": context.session.compacted_through_sequence,
        "active_goal": (
            {
                "goal_id": context.active_goal.goal_id,
                "task_type": context.active_goal.task_type,
                "mode": context.active_goal.mode,
                "status": context.active_goal.status,
            }
            if context.active_goal is not None
            else None
        ),
        "recent_turns": [
            {"sequence": item.sequence, "role": item.role, "content": item.content}
            for item in context.memory.recent_turns
        ],
        "cache_hit": context.cache_hit,
        "cache_fallback": context.cache_fallback,
    }


def _goal_action(task_type: str, active_goal) -> str:
    if task_type in {"general_chat", "scope_redirect", "clarify"}:
        return "preserve"
    if active_goal is None:
        return "create"
    if active_goal.task_type == task_type:
        return "continue"
    return "replace"


# ---------------------------------------------------------------------------
# 旧 LangGraph 路线的映射辅助（2026-09-10 停用，2026-09-11 整块注释冻结）
#
# run_citation_agent 与 agent/workflow.py 保留在仓库里作为历史实现，
# 但没有任何运行入口再调用它们：explain 一律走只读 Tool Loop。
# 用户 2026-09-11 的要求是「相关代码全部注释掉，但不要删除」，
# 所以下面整块保持原样注释，不参与任何默认路径，也不参与测试。
# 需要回退时取消注释即可。
# ---------------------------------------------------------------------------
# def _auto_from_agent(
#     router_result: QueryRouterResult, agent_result: AgentRepositoryAnswer
# ) -> AutoAnswer:
#     result = agent_result.result
#     subquestions = agent_result.subquestions
#     if not subquestions:
#         subquestions = tuple(
#             SubQuestionAnswer(
#                 question=item.question,
#                 intent=item.intent,
#                 retrieval_mode=item.retrieval_mode,
#                 outcome=result.outcome,
#                 answer=result.answer,
#                 citations=result.citations,
#             )
#             for item in router_result.plan.subquestions
#         )
#     return AutoAnswer(
#         outcome=result.outcome,
#         answer=result.answer,
#         citations=result.citations,
#         retrieval_mode="auto",
#         model=result.model,
#         prompt_version=result.prompt_version,
#         input_tokens=agent_result.input_tokens,
#         output_tokens=agent_result.output_tokens,
#         embedding_input_tokens=agent_result.embedding_input_tokens,
#         plan=router_result.plan,
#         subquestions=subquestions,
#         router_model=router_result.model,
#         router_input_tokens=router_result.input_tokens,
#         router_output_tokens=router_result.output_tokens,
#         router_elapsed_milliseconds=router_result.elapsed_milliseconds,
#         fallback_reason=router_result.fallback_reason,
#         events=tuple(
#             AutoAnswerEvent(item.sequence, item.step, item.summary)
#             for item in agent_result.events
#         ),
#     )


def _citation_payload(citation) -> dict[str, object]:
    return {
        "evidence_id": citation.evidence_id,
        "relative_path": citation.relative_path,
        "start_line": citation.start_line,
        "end_line": citation.end_line,
    }


def _auto_payload(result: AutoAnswer) -> dict[str, object]:
    return {
        "kind": "code_answer",
        "outcome": result.outcome,
        "citations": [_citation_payload(item) for item in result.citations],
        "retrieval_mode": result.retrieval_mode,
        "model": result.model,
        "prompt_version": result.prompt_version,
        "usage": {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens},
        "plan": result.plan.to_dict(),
        "subquestions": [
            {
                "question": item.question,
                "intent": item.intent,
                "retrieval_mode": item.retrieval_mode,
                "outcome": item.outcome,
                "answer": item.answer,
                "citations": [_citation_payload(citation) for citation in item.citations],
            }
            for item in result.subquestions
        ],
        "router_model": result.router_model,
        "router_usage": {
            "input_tokens": result.router_input_tokens,
            "output_tokens": result.router_output_tokens,
        },
        "router_elapsed_milliseconds": result.router_elapsed_milliseconds,
        "embedding_input_tokens": result.embedding_input_tokens,
        "fallback_reason": result.fallback_reason,
        "events": [
            {"sequence": item.sequence, "step": item.step, "summary": item.summary}
            for item in result.events
        ],
    }


def _usage_payload(events, *, fallback_input: int, fallback_output: int) -> dict[str, object]:
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_miss = 0
    for event in events:
        if event.event_type != "model_result" or event.payload.get("outcome") != "success":
            continue
        input_tokens += _int_payload(event.payload, "input_tokens")
        output_tokens += _int_payload(event.payload, "output_tokens")
        cache_read += _int_payload(event.payload, "cache_read_tokens")
        cache_miss += _int_payload(event.payload, "cache_miss_tokens")
    input_tokens = input_tokens or fallback_input
    output_tokens = output_tokens or fallback_output
    return {
        "input_tokens": input_tokens,
        "cache_read_tokens": cache_read,
        "cache_miss_tokens": cache_miss,
        "cache_hit_ratio": cache_read / input_tokens if input_tokens else 0.0,
        "output_tokens": output_tokens,
        "usage_source": "gateway_events" if cache_read or cache_miss else "model_result_summary",
    }


def _int_payload(payload: Mapping[str, str], key: str) -> int:
    try:
        return int(payload.get(key, "0"))
    except (TypeError, ValueError):
        return 0


def _preview_from_result(result: dict[str, object] | None) -> dict[str, object]:
    if not isinstance(result, dict) or not isinstance(result.get("preview"), dict):
        raise ValueError("当前轮次没有可审批的修改预览")
    return result["preview"]  # type: ignore[return-value]


_REPAIR_CONTINUATION = re.compile(
    r"(继续|修复|修正|重试|重新|校验失败|检查失败|补丁|repair|retry|fix)",
    re.IGNORECASE,
)


def _is_repair_continuation(message: str) -> bool:
    return bool(_REPAIR_CONTINUATION.search(message))


def _is_sandbox_failure(validation: Mapping[str, object]) -> bool:
    error_class = str(validation.get("error_class", "")).upper()
    excerpt = str(validation.get("message_excerpt", "")).lower()
    return error_class.startswith("SANDBOX_") or any(
        marker in excerpt
        for marker in ("docker", "docker_engine", "daemon", "access is denied")
    )


def _auto_repair_message(message: str) -> str:
    """自动修复轮的输入：保留用户目标，只补一句「这是修复轮」。

    失败摘要由 _repair_task 从事实层生成，这里不重复拼一遍——同一份摘要在模型
    眼里会变成两条不同的指令。
    """

    return (
        f"{message.strip()}\n\n"
        "[自动修复] 固定校验没有通过，请根据失败摘要继续修复同一个目标，"
        "不要重复已经应用过的改动。"
    )


def _repair_task(message: str, recovery: _ChangeRecovery) -> str:
    validation = recovery.validation
    profile = str(validation.get("profile", "unknown"))
    error_class = str(validation.get("error_class", "CHECK_FAILED"))
    excerpt = str(validation.get("message_excerpt", ""))[:2000]
    return (
        f"{message.strip()}\n\n"
        "这是上一轮修改的有限修复尝试。上一轮补丁已经应用到当前隔离 workspace，"
        "但固定校验没有通过。请先读取当前 workspace 的实际内容和相关测试，再只针对"
        "失败原因生成新的、确实不同的补丁；不要重复提交已经存在的内容。\n"
        "<validation_failure_digest>\n"
        f"profile: {profile}\nerror_class: {error_class}\n"
        f"message_excerpt: {excerpt}\n"
        "</validation_failure_digest>"
    )


def _stopped_message(status: str) -> str:
    """Worker 停下时的用户可见说法。原因不同，说法就不该共用一句「失败」。"""

    if status == UNKNOWN:
        return "这一轮的结果不确定：后台执行中断，需要人工核对隔离 workspace。"
    if status == MANUAL_REQUIRED:
        return "这一轮需要人工处理：后台执行没能完成。"
    return "本轮执行失败，请查看事件详情。"

def _change_result_execution(result, *, assistant_message: str) -> ChatExecution:
    return ChatExecution(
        status=CHAT_COMPLETED,
        assistant_message=assistant_message,
        result={"kind": "change_result", **result.as_dict()},
    )
