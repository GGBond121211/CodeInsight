"""Q-011 Task 9 验收：每一次 Worker 尝试都留下自己的上下文预算与压缩边界。

两条性质：

1. 一次 Run 的每个 attempt 都在事实里留下「哪一次尝试、哪套预算、从哪条压缩边界
   开始」，而不是只留一个总数；
2. 已经压过的历史不会被下一次尝试再压一遍：边界来自 Session 事实，不是进程内存。

全过程用 Fake Provider，不需要真实模型、Docker 或网络。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.application.conversation_service import ConversationService
from codeinsight.application.session_service import SessionService
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.model_gateway import resolve_route_budget
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore, InMemorySessionStore
from codeinsight.infrastructure.workspace import WorkspaceManager
from tests.integration.test_chat_change_async import (
    FakeChangeModel,
    FakeEmbedding,
    FakeReranker,
    PassingSandbox,
    _session,
    _submit_change,
    _wait_for,
)

# 每轮塞进历史的正文大小。默认窗口是 1M，测试里把它降到 128K，两轮就能压过
# Session 历史预算——压不过预算的用例证明不了「压缩真的被执行」。
TURN_CHARS = 60_000


def _client(
    tmp_path: Path,
    *,
    session_service: SessionService,
    run_store: InMemoryAgentRunStore,
    model: FakeChangeModel | None = None,
) -> TestClient:
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    changes = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=PassingSandbox(),
    )
    selected_model = model or FakeChangeModel()
    service = ConversationService(
        lambda: selected_model,
        FakeEmbedding,
        reranker_factory=FakeReranker,
        change_service=changes,
        session_service=session_service,
        agent_run_store=run_store,
    )
    return TestClient(create_app(conversation_service=service))


def _session_memory(service: ConversationService, session_id: str):
    session = service.session_service.load_session(session_id)
    assert session is not None
    memory = service.session_service.load_session_memory(
        session_id=session.session_id, scope=session.scope, repo_id=session.repo_id
    )
    assert memory is not None
    return memory


def _submit_chat(client: TestClient, *, session_id: str, repo: Path, message: str) -> str:
    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": message,
        },
    )
    assert accepted.status_code == 202
    turn_id = accepted.json()["turn_id"]
    assert _wait_for(client, turn_id, statuses={"COMPLETED", "FAILED"})["status"] == (
        "COMPLETED"
    )
    return turn_id


def test_every_attempt_records_its_budget_and_its_starting_boundary(tmp_path: Path) -> None:
    session_service = SessionService(InMemorySessionStore(), InMemoryMemoryStore())
    run_store = InMemoryAgentRunStore()
    client = _client(tmp_path, session_service=session_service, run_store=run_store)
    repo = tmp_path / "repo"
    session_id = _session(client, repo)

    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert client.post(f"/api/v2/chat/turns/{preview['turn_id']}/approve").status_code == 202
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    events = conversation.runtime.event_log.read_events(record.run_id)
    loaded = [event for event in events if event.event_type == "context_loaded"]

    # 预览、应用补丁、校验收尾：三次尝试，三次都留下自己的预算。
    assert [event.payload["attempt_id"] for event in loaded] == ["1", "2", "3"]
    budget = resolve_route_budget("change-plan")
    for event in loaded:
        assert event.payload["scene"] == "change-plan"
        assert event.payload["policy_version"] == budget.policy_version
        assert event.payload["context_window_tokens"] == str(budget.context_window_tokens)
        assert event.payload["reserved_output_tokens"] == str(budget.reserved_output_tokens)
        assert event.payload["input_allowance"] == str(budget.input_allowance)
        # 这一轮还没有压过历史，边界就是空的——不是编一个出来。
        assert event.payload["compaction_boundary_id"] == ""


def test_an_already_compacted_history_is_not_compacted_again(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CODEINSIGHT_CONTEXT_WINDOW_TOKENS", "128000")
    session_service = SessionService(InMemorySessionStore(), InMemoryMemoryStore())
    run_store = InMemoryAgentRunStore()
    first = _client(tmp_path, session_service=session_service, run_store=run_store)
    repo = tmp_path / "repo"
    session_id = _session(first, repo)
    blob = "请记住这段话。" * 10_000
    assert len(blob) >= TURN_CHARS

    _submit_chat(first, session_id=session_id, repo=repo, message=blob)
    _submit_chat(first, session_id=session_id, repo=repo, message=blob)

    service = first.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    compacted = _session_memory(service, session_id)
    assert compacted.compaction_count == 1
    assert compacted.compaction_boundary_id
    boundary = compacted.compaction_boundary_id

    # 「重启」：新的服务实例，同一份 Session 事实与 Run 事实。
    restarted = _client(tmp_path, session_service=session_service, run_store=run_store)
    _submit_chat(restarted, session_id=session_id, repo=repo, message="继续")

    after_restart = _session_memory(
        restarted.app.state.codeinsight_conversation, session_id  # type: ignore[attr-defined]
    )

    # 边界与压缩次数都没动：已经压过的历史不会再压一遍，也不会产生第二条边界。
    assert after_restart.compaction_count == 1
    assert after_restart.compaction_boundary_id == boundary


def test_the_worker_uses_the_scene_budget_not_its_own(tmp_path: Path) -> None:
    """解释与修改共用一套预算来源：场景不同，窗口与预留值来自同一个解析器。"""

    session_service = SessionService(InMemorySessionStore(), InMemoryMemoryStore())
    run_store = InMemoryAgentRunStore()
    client = _client(tmp_path, session_service=session_service, run_store=run_store)
    repo = tmp_path / "repo"
    session_id = _session(client, repo)
    turn_id = _submit_chat(
        client, session_id=session_id, repo=repo, message="这个仓库是做什么的？"
    )

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(turn_id)
    events = conversation.runtime.event_log.read_events(record.run_id)
    loaded = [event for event in events if event.event_type == "context_loaded"]

    assert loaded, "Worker 必须先留下这一轮的上下文预算"
    budget = resolve_route_budget("explain")
    assert loaded[0].payload["scene"] == "explain"
    assert loaded[0].payload["context_window_tokens"] == str(budget.context_window_tokens)
    assert loaded[0].payload["session_history_budget"] == str(budget.session_history_budget)
