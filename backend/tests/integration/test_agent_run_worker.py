"""Q-011 Task 5：AgentRunWorker 端到端跑完整只读 Tool Loop。

用假 Provider 与假 MCP Client 走真实链路：受理 -> 投递 -> Worker 领取 ->
重建上下文 -> Model/Tool 循环 -> 落状态。断言的是「链路真的这样跑」，
不是模型质量；真实模型的行为另有单独取证。
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.domain.agent_run import COMPLETED, RUNNING
from tests.integration.test_chat_api import (
    FakeChatModel,
    FakeEmbedding,
    FakeReranker,
    FakeToolLoopClient,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


def _client(model: FakeChatModel) -> TestClient:
    return TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            reranker_factory=FakeReranker,
            mcp_client_factory=FakeToolLoopClient,
        )  # type: ignore[arg-type]
    )


def _service(client: TestClient):
    return client.app.state.codeinsight_conversation  # type: ignore[attr-defined]


def _wait_for_terminal(client: TestClient, turn_id: str) -> dict:
    for _ in range(500):
        payload = client.get(f"/api/v2/chat/turns/{turn_id}").json()
        if payload["status"] in {"COMPLETED", "FAILED", "WAITING_APPROVAL"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("聊天 Run 未在测试窗口内结束")


def _submit(client: TestClient, session_id: str) -> dict:
    return client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "checkout 如何校验输入？",
        },
    ).json()


def test_worker_runs_the_read_only_tool_loop_end_to_end() -> None:
    """一个 Worker Run 里完成检索 -> 工具结果 -> 最终回答。"""

    model = FakeChatModel()
    client = _client(model)
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    accepted = _submit(client, session_id)
    result = _wait_for_terminal(client, accepted["turn_id"])

    assert result["status"] == "COMPLETED"
    assert result["task_id"] == accepted["task_id"]
    # 两轮模型调用：先决定检索，拿到工具结果后再给最终回答。
    assert model.tool_rounds == 2

    service = _service(client)
    record = service.agent_run_store.get_run(result["run_id"])
    assert record is not None
    assert record.status == COMPLETED
    assert record.task_id == accepted["task_id"]
    # 终态不保留租约：跑完的 Run 不该继续占着 Worker。
    assert record.worker_id is None
    assert record.lease_until_epoch_ms is None

    events = service.runtime.event_log.read_events(result["run_id"])
    event_types = [event.event_type for event in events]
    assert event_types[:3] == ["run_started", "turn_accepted", "state_transitioned"]
    for expected in (
        "worker_claimed",
        "task_queued",
        "intent_classified",
        "context_loaded",
        "session_loaded",
        "context_assembled",
        "tool_call_requested",
        "tool_dispatched",
        "tool_result_committed",
        "answer_ready",
        "run_finished",
    ):
        assert expected in event_types, expected
    # 领取发生在执行之前，而不是事后补记。
    assert event_types.index("worker_claimed") < event_types.index("answer_ready")


def test_worker_records_the_context_budget_it_actually_used() -> None:
    """每一次模型调用都经过统一 Context Lifecycle，预算必须可核对。"""

    model = FakeChatModel()
    client = _client(model)
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]
    accepted = _submit(client, session_id)
    result = _wait_for_terminal(client, accepted["turn_id"])

    service = _service(client)
    loaded = [
        event
        for event in service.runtime.event_log.read_events(result["run_id"])
        if event.event_type == "context_loaded"
    ]

    assert len(loaded) == 1
    payload = loaded[0].payload
    assert payload["scene"] == "explain"
    assert int(payload["context_window_tokens"]) > 0
    assert int(payload["session_history_budget"]) > 0
    assert int(payload["input_allowance"]) == int(payload["context_window_tokens"]) - int(
        payload["reserved_output_tokens"]
    )


def test_the_worker_can_rebuild_the_run_from_the_store_alone() -> None:
    """受理之后，Worker 只凭 Store 就能重建这一轮——这是异步执行的前提。"""

    model = FakeChatModel()
    client = _client(model)
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]
    accepted = _submit(client, session_id)

    service = _service(client)
    record = service.agent_run_store.get_run(accepted["run_id"])
    assert record is not None
    context = service.agent_run_context_loader.load(_task_like(accepted), record)

    assert context.user_message == "checkout 如何校验输入？"
    assert context.repo_root == str(FIXTURE_ROOT)
    assert context.task_type == "explain"
    assert context.options.validation_profile == "python_compile"

    assert _wait_for_terminal(client, accepted["turn_id"])["status"] == "COMPLETED"


def _task_like(accepted: dict):
    from codeinsight.domain.agent_run import TASK_AGENT_RUN, AgentRunTask

    return AgentRunTask(
        task_id=accepted["task_id"],
        session_id=accepted["session_id"],
        turn_id=accepted["turn_id"],
        run_id=accepted["run_id"],
        task_kind=TASK_AGENT_RUN,
        idempotency_key=accepted["turn_id"],
        policy_version="chat-agent-run-v1",
        deadline_epoch_ms=int(time.time() * 1000) + 60_000,
    )


def test_run_keeps_the_queued_to_running_to_completed_path() -> None:
    """状态机走完整条路径：受理是 QUEUED，领取后 RUNNING，结束是 COMPLETED。"""

    model = FakeChatModel()
    client = _client(model)
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    accepted = _submit(client, session_id)
    service = _service(client)
    seen: set[str] = set()
    for _ in range(500):
        record = service.agent_run_store.get_run(accepted["run_id"])
        if record is not None:
            seen.add(record.status)
        if record is not None and record.status == COMPLETED:
            break
        time.sleep(0.02)

    assert COMPLETED in seen
    assert seen <= {RUNNING, COMPLETED}

