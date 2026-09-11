"""Q-011 Task 5：Worker 侧重建上下文与执行边界的单元测试。

这里锁住的是「Worker 只能凭事实层工作」这条边界：会话、仓库根、本轮问题全部来自
Store；任何一项缺失都必须明确失败，而不是拿一个凑合的默认值继续跑。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from codeinsight.application.agent_run_executor import (
    CONTEXT_MESSAGE_NOT_FOUND,
    CONTEXT_REPOSITORY_MISSING,
    CONTEXT_REPOSITORY_UNKNOWN,
    CONTEXT_SESSION_NOT_FOUND,
    AgentRunContext,
    AgentRunContextError,
    AgentRunContextLoader,
    AgentRunExecutor,
)
from codeinsight.application.session_service import SessionService
from codeinsight.domain.agent_run import (
    QUEUED,
    TASK_AGENT_RUN,
    AgentRunRecord,
    AgentRunTask,
    RunRequestOptions,
)
from codeinsight.domain.change import TenantScope
from codeinsight.domain.errors import ModelCallError
from codeinsight.infrastructure.chat_runtime import ChatExecution
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.run_store import InMemorySessionStore

INDEX_VERSION = "test-index-v1"


def _task(**overrides: object) -> AgentRunTask:
    values: dict[str, object] = {
        "task_id": "task-1",
        "session_id": "s-1",
        "turn_id": "turn-1",
        "run_id": "run-1",
        "task_kind": TASK_AGENT_RUN,
        "idempotency_key": "turn-1",
        "policy_version": "chat-agent-run-v1",
        "deadline_epoch_ms": 10_000,
    }
    values.update(overrides)
    return AgentRunTask(**values)  # type: ignore[arg-type]


def _record(**overrides: object) -> AgentRunRecord:
    values: dict[str, object] = {
        "run_id": "run-1",
        "turn_id": "turn-1",
        "session_id": "s-1",
        "task_id": "task-1",
        "task_kind": TASK_AGENT_RUN,
        "status": QUEUED,
        "policy_version": "chat-agent-run-v1",
        "idempotency_key": "turn-1",
        "deadline_epoch_ms": 10_000,
        "updated_at_epoch_ms": 1,
    }
    values.update(overrides)
    return AgentRunRecord(**values)  # type: ignore[arg-type]


def _service() -> SessionService:
    return SessionService(
        InMemorySessionStore(),
        InMemoryMemoryStore(),
        None,
        max_recent_turns=12,
    )


def _seed_session(service: SessionService, repo_root: str, message: str) -> None:
    context = service.get_or_create_session(
        session_id="s-1",
        scope=TenantScope(),
        repo_id="repo-1",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.append_turn(
        context,
        role="user",
        content=message,
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.bind_repository_root("s-1", repo_root)


def _loader(service: SessionService) -> AgentRunContextLoader:
    return AgentRunContextLoader(
        session_service=service,
        index_version=INDEX_VERSION,
        fingerprint=lambda root: f"fp:{root}",
    )


# ---------------------------------------------------------------------------
# 上下文重建
# ---------------------------------------------------------------------------


def test_loader_rebuilds_the_round_from_the_store(tmp_path: Path) -> None:
    service = _service()
    _seed_session(service, str(tmp_path), "checkout 如何校验输入？")

    context = _loader(service).load(_task(), _record())

    assert context.session_id == "s-1"
    assert context.repo_id == "repo-1"
    assert context.repo_root == str(tmp_path)
    assert context.user_message == "checkout 如何校验输入？"
    assert context.task_type == "explain"
    assert context.options == RunRequestOptions()


def test_loader_carries_the_request_options_from_the_run() -> None:
    """请求参数必须来自 Run 事实：Worker 拿不到请求对象。"""

    service = _service()
    _seed_session(service, str(Path.cwd()), "你好")
    record = _record(
        options=RunRequestOptions(
            validation_profile="python_compile",
            result_limit=3,
            show_debug_reasoning=True,
        )
    )

    context = _loader(service).load(_task(), record)

    assert context.options.result_limit == 3
    assert context.options.show_debug_reasoning is True


def test_loader_reads_the_latest_user_turn(tmp_path: Path) -> None:
    """多轮会话里，Worker 要回答的是最后那一句，不是第一句。"""

    service = _service()
    _seed_session(service, str(tmp_path), "第一问")
    context = service.get_or_create_session(
        session_id="s-1",
        scope=TenantScope(),
        repo_id="repo-1",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.append_turn(
        context,
        role="assistant",
        content="第一答",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.append_turn(
        context,
        role="user",
        content="第二问",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )

    loaded = _loader(service).load(_task(), _record())

    assert loaded.user_message == "第二问"


def test_missing_session_is_refused() -> None:
    with pytest.raises(AgentRunContextError) as error:
        _loader(_service()).load(_task(), _record())
    assert error.value.error_class == CONTEXT_SESSION_NOT_FOUND


def test_session_without_a_bound_repository_is_refused(tmp_path: Path) -> None:
    """没有落库的仓库根，跨进程的 Worker 无从知道该读哪个目录。"""

    service = _service()
    context = service.get_or_create_session(
        session_id="s-1",
        scope=TenantScope(),
        repo_id="repo-1",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.append_turn(
        context,
        role="user",
        content="问题",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )

    with pytest.raises(AgentRunContextError) as error:
        _loader(service).load(_task(), _record())
    assert error.value.error_class == CONTEXT_REPOSITORY_UNKNOWN


def test_repository_that_no_longer_exists_is_refused(tmp_path: Path) -> None:
    service = _service()
    _seed_session(service, str(tmp_path / "gone"), "问题")

    with pytest.raises(AgentRunContextError) as error:
        _loader(service).load(_task(), _record())
    assert error.value.error_class == CONTEXT_REPOSITORY_MISSING


def test_session_without_a_user_turn_is_refused(tmp_path: Path) -> None:
    service = _service()
    service.get_or_create_session(
        session_id="s-1",
        scope=TenantScope(),
        repo_id="repo-1",
        repo_fingerprint="fp-1",
        index_version=INDEX_VERSION,
    )
    service.bind_repository_root("s-1", str(tmp_path))

    with pytest.raises(AgentRunContextError) as error:
        _loader(service).load(_task(), _record())
    assert error.value.error_class == CONTEXT_MESSAGE_NOT_FOUND


# ---------------------------------------------------------------------------
# 执行边界
# ---------------------------------------------------------------------------


def _context() -> AgentRunContext:
    return AgentRunContext(
        run_id="run-1",
        turn_id="turn-1",
        session_id="s-1",
        task_id="task-1",
        repo_id="repo-1",
        repo_root="C:/repo",
        repo_fingerprint="fp:repo",
        user_message="checkout 如何校验输入？",
        task_type="explain",
        confidence=0.9,
        rule="test",
        options=RunRequestOptions(),
    )


def test_executor_records_the_context_budget_before_running() -> None:
    """每一次模型调用都必须经过统一 Context Lifecycle，且预算可核对。"""

    events: list[tuple[str, str, dict[str, str]]] = []
    calls: list[str] = []

    def runner(context: AgentRunContext) -> ChatExecution:
        calls.append(context.user_message)
        return ChatExecution(status="COMPLETED", assistant_message="答")

    AgentRunExecutor(
        emit=lambda run_id, kind, payload: events.append((run_id, kind, payload))
    ).execute(_context(), runner=runner)

    assert calls == ["checkout 如何校验输入？"]
    assert len(events) == 1
    run_id, event_type, payload = events[0]
    assert run_id == "run-1"
    assert event_type == "context_loaded"
    assert payload["scene"] == "explain"
    assert int(payload["context_window_tokens"]) > 0
    assert int(payload["session_history_budget"]) > 0
    assert int(payload["input_allowance"]) > 0


def test_change_scene_uses_the_change_plan_budget() -> None:
    """修改任务走 change-plan 预算，而不是只读场景的那一套。"""

    events: list[tuple[str, str, dict[str, str]]] = []
    AgentRunExecutor(
        emit=lambda run_id, kind, payload: events.append((run_id, kind, payload))
    ).execute(
        replace(_context(), task_type="change"),
        runner=lambda context: ChatExecution(status="COMPLETED", assistant_message="答"),
    )

    assert events[0][2]["scene"] == "change-plan"


def test_provider_failure_is_retryable() -> None:
    def runner(context: AgentRunContext) -> ChatExecution:
        raise ModelCallError("upstream 502")

    with pytest.raises(Exception) as error:
        AgentRunExecutor(emit=lambda *args: None).execute(_context(), runner=runner)

    assert getattr(error.value, "retryable", None) is True
    assert error.value.error_class == "ModelCallError"


def test_configuration_failure_is_not_retryable() -> None:
    def runner(context: AgentRunContext) -> ChatExecution:
        raise ValueError("缺配置")

    with pytest.raises(Exception) as error:
        AgentRunExecutor(emit=lambda *args: None).execute(_context(), runner=runner)

    assert getattr(error.value, "retryable", None) is False
    assert error.value.error_class == "ValueError"

