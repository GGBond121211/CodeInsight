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
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from codeinsight.agent.agent_run_worker import AgentRunWorker
from codeinsight.agent.change_workflow import run_change_workflow_to_preview
from codeinsight.agent.tool_loop import ToolLoopConfig
from codeinsight.application.agent_run_dispatcher import (
    AgentRunDispatcher,
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
from codeinsight.domain.agent_run import (
    CANCELLED,
    COMPLETED,
    FAILED,
    MANUAL_REQUIRED,
    QUEUED,
    RUNNING,
    TASK_AGENT_RUN,
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
from codeinsight.ingestion.scanner import scan_repository

# 2026-09-11：旧 LangGraph 路线整体冻结后，本文件不再需要这些符号；保留原名
# 以便回退：AgentRepositoryAnswer、QueryRouterResult、AutoAnswerEvent、
# SubQuestionAnswer。

INDEX_VERSION = "conversation-scan-v1"

# 一次 Chat Agent Run 的策略版本与受理期限。期限只约束「这一轮多久之内必须
# 跑完」，不是模型的超时；超时后的处理见 recovery_decision。
CHAT_RUN_POLICY_VERSION = "chat-agent-run-v1"
DEFAULT_RUN_DEADLINE_MS = 10 * 60 * 1000

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
        self.agent_run_dispatcher = agent_run_dispatcher or AgentRunDispatcher(
            store=self.agent_run_store,
            transport=CallbackAgentRunTransport(self._schedule_turn),
        )
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
        self._change_recoveries: dict[str, _ChangeRecovery] = {}
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
        """把一轮执行交给 AgentRunWorker，并如实报告它没有产出回答的情况。

        Worker 负责领取租约、重建上下文、落状态；这里只把执行体接上去。
        """

        outcome = self.agent_run_worker.handle(
            task,
            runner=lambda loaded: self._execute_turn(
                turn,
                _turn_input_from_context(loaded),
                loaded.classification,
                show_debug_reasoning,
            ),
        )
        if outcome.execution is None:
            raise ChatRuntimeError(
                f"Agent Run 未产出回答：{outcome.error_class or outcome.status}"
            )
        return outcome.execution


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
        turn = self.runtime.get_turn(turn_id)
        if turn.status != CHAT_WAITING_APPROVAL:
            raise ValueError("当前轮次不在等待审批状态")
        turn_input = self._turn_inputs.get(turn_id)
        if turn_input is None:
            raise ValueError("当前轮次的本地执行上下文已失效")
        preview = _preview_from_result(turn.result)
        token = self.change_service.approve(
            turn.run_id,
            str(preview["patch_id"]),
        )
        if not callable(getattr(self.change_service, "add_event_sink", None)):
            self.runtime.emit(
                turn.run_id,
                APPROVAL_GRANTED,
                {"patch_id": str(preview["patch_id"]), "source": "chat_ui"},
            )
        def worker(current: ChatTurn, _debug: bool) -> ChatExecution:
            return self._apply_change(
                current,
                turn_input,
                str(preview["patch_id"]),
                token,
            )

        return self.runtime.resume(
            turn_id,
            show_debug_reasoning=False,
            worker=worker,
        )

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
                if result.status == "COMPLETED":
                    self._change_recoveries.pop(turn.session_id, None)
                else:
                    self._change_recoveries[turn.session_id] = _ChangeRecovery(
                        run_id=recovery.run_id,
                        patch_id=recovery.patch_id,
                        validation=dict(result.validation or {}),
                    )
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
    ) -> ChatExecution:
        self.runtime.emit(turn.run_id, STEP_STARTED, {"stage": "apply", "status": "running"})
        result = self.change_service.apply(turn.run_id, patch_id, approval_token)
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
        if result.status == "REVIEW_REQUIRED" and result.validation is not None:
            self._change_recoveries[turn.session_id] = _ChangeRecovery(
                run_id=turn.run_id,
                patch_id=patch_id,
                validation=dict(result.validation),
            )
        elif result.status == "COMPLETED":
            self._change_recoveries.pop(turn.session_id, None)
        if result.status == "COMPLETED" and result.validation and result.validation.get(
            "skipped"
        ):
            assistant_message = "修改已应用；当前开发模式跳过了 Docker 固定校验。"
        elif result.status == "COMPLETED":
            assistant_message = "修改已应用并完成校验。"
        else:
            assistant_message = f"修改流程结束，状态为 {result.status}。"
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
        execution = _change_result_execution(
            result,
            assistant_message=assistant_message,
        )
        return execution

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


def _change_result_execution(result, *, assistant_message: str) -> ChatExecution:
    return ChatExecution(
        status=CHAT_COMPLETED,
        assistant_message=assistant_message,
        result={"kind": "change_result", **result.as_dict()},
    )
