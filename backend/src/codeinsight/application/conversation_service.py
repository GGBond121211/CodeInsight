"""统一对话入口与业务编排层。

一轮聊天在这里被拆成：恢复 Session -> 分类解释/修改 -> 组装上下文 ->
执行既有只读或变更工作流 -> 记录公开回答。前端不需要知道两条内部路径，
但每一步都通过同一个 run_id 进入实时事件流。
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from codeinsight.agent.change_workflow import run_change_workflow_to_preview
from codeinsight.agent.tool_loop import ToolLoopConfig
from codeinsight.agent.workflow import run_citation_agent
from codeinsight.application.auto_answer_repository import auto_answer_repository
from codeinsight.application.conversation_router import (
    ChatTaskClassification,
    classify_chat_task,
)
from codeinsight.application.query_router import QueryRouterResult, route_question
from codeinsight.application.scope_redirect import (
    SCOPE_REDIRECT_VERSION,
    build_scope_redirect_message,
)
from codeinsight.application.session_service import SessionContext, SessionService
from codeinsight.domain.agent import AgentRepositoryAnswer
from codeinsight.domain.answer import (
    ANSWERED,
    INSUFFICIENT_EVIDENCE,
    AutoAnswer,
    AutoAnswerEvent,
    SubQuestionAnswer,
)
from codeinsight.domain.change import MODE_ISOLATED_WRITE, MODE_READ_ONLY, TenantScope
from codeinsight.domain.chat import (
    CHAT_COMPLETED,
    CHAT_FAILED,
    CHAT_WAITING_APPROVAL,
    ChatTurn,
)
from codeinsight.domain.errors import ModelCallError, ModelConfigurationError, ModelResponseError
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
    VALIDATION_PREFLIGHT,
)
from codeinsight.infrastructure.chat_runtime import ChatExecution, ChatRuntime
from codeinsight.infrastructure.gateway_errors import GatewayError
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.model_gateway import CacheContext
from codeinsight.infrastructure.redaction import redact_sensitive
from codeinsight.infrastructure.reranker import Reranker
from codeinsight.infrastructure.run_store import InMemorySessionStore
from codeinsight.infrastructure.runtime_policy import DevelopmentPolicy
from codeinsight.ingestion.scanner import scan_repository

INDEX_VERSION = "conversation-scan-v1"
# Q-008：explain 路线的迁移开关；默认关闭，只影响 explain。
CODE_UNDERSTANDING_ENV_FLAG = "CODEINSIGHT_CODE_UNDERSTANDING_TOOL_LOOP"
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


@dataclass(frozen=True)
class _ChangeRecovery:
    """上一轮校验失败后，供同一 Session 继续修复的公开摘要。"""

    run_id: str
    patch_id: str
    validation: dict[str, object]


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
    ) -> None:
        self._model_factory = model_factory
        self._embedding_factory = embedding_factory
        self._reranker_factory = reranker_factory
        # 只读代码理解 Tool Loop 的 MCP Client 工厂；测试注入 Fake，生产走 stdio。
        self._mcp_client_factory = mcp_client_factory
        # Q-008 迁移开关：默认关闭，直到新旧路线有同数据对照（计划 Task 5/6）。
        self.code_understanding_enabled = code_understanding_tool_loop_enabled()
        self.change_service = change_service
        selected_policy = getattr(change_service, "development_policy", None)
        self.development_policy = (
            selected_policy
            if isinstance(selected_policy, DevelopmentPolicy)
            else DevelopmentPolicy.from_environment()
        )
        self.session_service = session_service or SessionService(
            InMemorySessionStore(),
            InMemoryMemoryStore(),
            max_recent_turns=12,
        )
        self.runtime = runtime or ChatRuntime()
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
        if not message.strip():
            raise ValueError("message 不能为空")
        root, repo_id, repo_fingerprint, contains_workspace_state = (
            self._resolve_session_repository(
                session_id, repository_root, refresh_fingerprint=False
            )
        )
        if client_turn_id is not None:
            with self._client_turn_lock:
                existing = self._client_turns.get((session_id, client_turn_id))
                if existing is not None:
                    existing_input = self._turn_inputs.get(existing.turn_id)
                    if existing_input is not None and existing_input.repo_id != repo_id:
                        raise ValueError("client_turn_id 不能跨仓库复用")
                    return self.runtime.get_turn(existing.turn_id)
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
        turn_input = _TurnInput(
            str(root),
            repo_id,
            repo_fingerprint,
            validation_profile.strip(),
            limit,
            contains_workspace_state,
        )
        def worker(turn: ChatTurn, debug: bool) -> ChatExecution:
            return self._execute_turn(turn, turn_input, classification, debug)

        turn = self.runtime.submit(
            session_id=session_id,
            task_type=classification.task_type,
            user_message=message.strip(),
            show_debug_reasoning=show_debug_reasoning,
            worker=worker,
        )
        self._turn_inputs[turn.turn_id] = turn_input
        if client_turn_id is not None:
            with self._client_turn_lock:
                self._client_turns[(session_id, client_turn_id)] = turn
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
        with self._session_workspace_lock:
            self._session_workspaces.setdefault(
                session_id,
                _SessionWorkspace(repo_id, source_root.resolve(), source_root.resolve()),
            )

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
            previous_compactions = context.memory.compaction_count
            context = self.session_service.append_turn(
                context,
                role="user",
                content=turn.user_message,
                repo_fingerprint=turn_input.repo_fingerprint,
                index_version=INDEX_VERSION,
            )
            if context.memory.compaction_count > previous_compactions:
                self.runtime.emit(
                    turn.run_id,
                    CONTEXT_COMPACTED,
                    {
                        "compaction_count": str(context.memory.compaction_count),
                        "through_sequence": str(context.memory.compacted_through_sequence),
                    },
                )
            goal_type = classification.task_type
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
        embedding_model = (
            self._embedding_factory()
            if router_result.plan.execution_route != "insufficient"
            else None
        )
        reranker = (
            self._reranker_factory()
            if router_result.plan.execution_route != "insufficient"
            else None
        )
        if router_result.plan.execution_route == "agent":
            if self.code_understanding_enabled:
                result, tool_loop_payload = self._run_code_understanding(
                    turn, turn_input, model, router_result
                )
            else:
                agent_result = run_citation_agent(
                    turn_input.repository_root,
                    turn.user_message,
                    complete=model.complete,
                    limit=turn_input.limit,
                    retrieval_mode="auto",
                    semantic_embed=embedding_model.embed if embedding_model else None,
                    reranker=reranker,
                    query_plan=router_result.plan,
                )
                result = _auto_from_agent(router_result, agent_result)
                tool_loop_payload = None
        else:
            result = auto_answer_repository(
                turn_input.repository_root,
                router_result=router_result,
                generate=model.generate,
                limit=turn_input.limit,
                semantic_embed=embedding_model.embed if embedding_model else None,
                reranker=reranker,
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
        """跑一次只读代码理解 Tool Loop，并映射回公开的 AutoAnswer 形状。"""
        from codeinsight.agent.code_understanding_tool_loop import (
            ANSWERED_STATUS,
            CodeUnderstandingToolLoop,
        )

        factory = self._mcp_client_factory or _default_mcp_client_factory()
        with factory(turn_input.repository_root) as client:
            loop = CodeUnderstandingToolLoop(
                model,
                client,
                run_id=turn.run_id,
                event_log=self.runtime.event_log,
                repo_root=turn_input.repository_root,
                emit=lambda event_type, detail: self.runtime.emit(
                    turn.run_id, event_type, detail
                ),
            )
            loop_result = loop.run(turn.user_message)
        outcome = (
            ANSWERED
            if loop_result.status == ANSWERED_STATUS
            else INSUFFICIENT_EVIDENCE
        )
        subquestions = tuple(
            SubQuestionAnswer(
                question=item.question,
                intent=item.intent,
                retrieval_mode="auto",
                outcome=outcome,
                answer=loop_result.answer,
                citations=loop_result.citations,
            )
            for item in router_result.plan.subquestions
        )
        answer = AutoAnswer(
            outcome=outcome,
            answer=loop_result.answer,
            citations=loop_result.citations,
            retrieval_mode="auto",
            model=model.model,
            prompt_version=loop_result.prompt_version,
            input_tokens=loop_result.input_tokens,
            output_tokens=loop_result.output_tokens,
            # MCP Server 在子进程内做 Embedding，用量目前不回流到主进程。
            embedding_input_tokens=0,
            plan=router_result.plan,
            subquestions=subquestions,
            router_model=router_result.model,
            router_input_tokens=router_result.input_tokens,
            router_output_tokens=router_result.output_tokens,
            router_elapsed_milliseconds=router_result.elapsed_milliseconds,
            fallback_reason=router_result.fallback_reason,
            events=(),
        )
        return answer, loop_result.as_dict()

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

    def complete(self, system_prompt: str, user_prompt: str):
        result = _call_with_run_context(
            getattr(self._base, "complete"),
            system_prompt,
            self._with_history(user_prompt),
            run_id=self._runtime.get_turn(self._turn_id).run_id,
            event_log=self._runtime.event_log,
            cache_context=self._cache_context,
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


def _call_with_run_context(method, *args, run_id: str, event_log, cache_context=None):
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


def code_understanding_tool_loop_enabled(
    source: Mapping[str, str] | None = None,
) -> bool:
    """Q-008 迁移开关；只控制 explain 路线，默认关闭。

    默认关闭的原因：新路线还没有和 LangGraph 的同数据、同模型对照
    （计划 Task 5），在拿到可回放结果之前不能沉默地替换默认行为。
    """
    environment = os.environ if source is None else source
    return environment.get(CODE_UNDERSTANDING_ENV_FLAG, "").strip() == "1"


def _default_mcp_client_factory():
    from codeinsight.infrastructure.mcp_client import StdioMCPClient

    return StdioMCPClient


def _auto_from_agent(
    router_result: QueryRouterResult, agent_result: AgentRepositoryAnswer
) -> AutoAnswer:
    result = agent_result.result
    subquestions = agent_result.subquestions
    if not subquestions:
        subquestions = tuple(
            SubQuestionAnswer(
                question=item.question,
                intent=item.intent,
                retrieval_mode=item.retrieval_mode,
                outcome=result.outcome,
                answer=result.answer,
                citations=result.citations,
            )
            for item in router_result.plan.subquestions
        )
    return AutoAnswer(
        outcome=result.outcome,
        answer=result.answer,
        citations=result.citations,
        retrieval_mode="auto",
        model=result.model,
        prompt_version=result.prompt_version,
        input_tokens=agent_result.input_tokens,
        output_tokens=agent_result.output_tokens,
        embedding_input_tokens=agent_result.embedding_input_tokens,
        plan=router_result.plan,
        subquestions=subquestions,
        router_model=router_result.model,
        router_input_tokens=router_result.input_tokens,
        router_output_tokens=router_result.output_tokens,
        router_elapsed_milliseconds=router_result.elapsed_milliseconds,
        fallback_reason=router_result.fallback_reason,
        events=tuple(
            AutoAnswerEvent(item.sequence, item.step, item.summary)
            for item in agent_result.events
        ),
    )


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
