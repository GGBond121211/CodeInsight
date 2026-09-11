"""Q-011 Task 4：受理必须是非阻塞的、幂等的、失败可见的。

用一个会卡住的假模型证明 API 不等它；用同一个 client_turn_id 证明重发不会跑
第二遍模型；用一次投递失败证明失败会变成 503，而不是一个悬空的 202。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.application.agent_run_dispatcher import (
    AgentRunDispatcher,
    InMemoryAgentRunTransport,
)
from codeinsight.application.change_service import ChangeService
from codeinsight.application.conversation_service import ConversationService, _turn_id_for
from codeinsight.domain.answer import ModelCompletion
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


class BlockingModel:
    """普通对话路径上的假模型：进来就卡住，直到测试放行。"""

    model = "blocking-fake"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def complete_text(self, _system_prompt: str, _user_prompt: str) -> ModelCompletion:
        self.calls += 1
        self.entered.set()
        self.release.wait(timeout=20)
        return ModelCompletion("已经放行。", self.model, 3, 4)


class ExplodingEmbedding:
    def __init__(self) -> None:
        raise AssertionError("普通对话不应创建 embedding 模型")


class UnusedReranker:
    def __init__(self) -> None:
        raise AssertionError("普通对话不应创建 reranker")


def _client(model: BlockingModel) -> TestClient:
    return TestClient(
        create_app(  # type: ignore[arg-type]
            lambda: model,
            ExplodingEmbedding,
            reranker_factory=UnusedReranker,
        )
    )


def _new_session(client: TestClient) -> str:
    created = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    )
    return str(created.json()["session_id"])


def _submit(
    client: TestClient, session_id: str, *, client_turn_id: str | None = None
):
    body: dict[str, object] = {
        "session_id": session_id,
        "repository_root": str(FIXTURE_ROOT),
        "message": "你好",
    }
    if client_turn_id is not None:
        body["client_turn_id"] = client_turn_id
    return client.post("/api/v2/chat/turns", json=body)


def _wait_for_terminal(client: TestClient, turn_id: str) -> dict:
    for _ in range(500):
        payload = client.get(f"/api/v2/chat/turns/{turn_id}").json()
        if payload["status"] in {"COMPLETED", "FAILED", "WAITING_APPROVAL"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("聊天 Run 未在测试窗口内结束")


def test_submit_returns_202_while_the_model_is_still_blocked() -> None:
    model = BlockingModel()
    client = _client(model)
    session_id = _new_session(client)

    started = time.monotonic()
    accepted = _submit(client, session_id, client_turn_id="client-turn-1")
    elapsed = time.monotonic() - started

    assert accepted.status_code == 202
    body = accepted.json()
    assert body["status"] == "QUEUED"
    assert body["task_id"].startswith("task-")
    assert model.entered.wait(timeout=10) is True
    assert model.release.is_set() is False
    assert elapsed < 5

    store = client.app.state.codeinsight_conversation.agent_run_store
    record = store.find_run_by_turn(body["turn_id"])
    assert record is not None
    assert record.status in {"QUEUED", "RUNNING"}

    model.release.set()
    assert _wait_for_terminal(client, body["turn_id"])["status"] == "COMPLETED"


def test_duplicate_submit_does_not_run_the_model_twice() -> None:
    model = BlockingModel()
    client = _client(model)
    session_id = _new_session(client)

    first = _submit(client, session_id, client_turn_id="client-turn-dup")
    assert model.entered.wait(timeout=10) is True

    duplicate = _submit(client, session_id, client_turn_id="client-turn-dup")

    assert duplicate.status_code == 202
    assert duplicate.json()["turn_id"] == first.json()["turn_id"]
    assert duplicate.json()["run_id"] == first.json()["run_id"]
    assert model.calls == 1

    model.release.set()
    assert _wait_for_terminal(client, first.json()["turn_id"])["status"] == "COMPLETED"


def test_closing_the_event_stream_early_does_not_stop_the_run() -> None:
    model = BlockingModel()
    client = _client(model)
    session_id = _new_session(client)
    turn_id = _submit(client, session_id).json()["turn_id"]
    assert model.entered.wait(timeout=10) is True

    with client.stream("GET", f"/api/v2/chat/turns/{turn_id}/events") as response:
        assert response.status_code == 200
        assert next(response.iter_lines(), None) is not None

    model.release.set()
    assert _wait_for_terminal(client, turn_id)["status"] == "COMPLETED"


def test_dispatch_failure_is_reported_instead_of_a_queued_lie() -> None:
    store = InMemoryAgentRunStore()
    transport = InMemoryAgentRunTransport(failure=RuntimeError("broker down"))
    service = ConversationService(
        lambda: BlockingModel(),
        ExplodingEmbedding,  # type: ignore[arg-type]
        reranker_factory=UnusedReranker,
        change_service=ChangeService(),
        agent_run_dispatcher=AgentRunDispatcher(store=store, transport=transport),
        agent_run_store=store,
    )
    client = TestClient(create_app(conversation_service=service))
    session_id = _new_session(client)

    refused = _submit(client, session_id, client_turn_id="client-fail-1")

    assert refused.status_code == 503
    assert "DISPATCH_FAILED" in refused.json()["detail"]
    record = store.find_run_by_turn(_turn_id_for(session_id, "client-fail-1"))
    assert record is not None
    assert record.status == "FAILED"
    assert record.error_class == "DISPATCH_FAILED"

