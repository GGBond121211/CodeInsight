from __future__ import annotations

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse
from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch
from codeinsight.infrastructure.reranker import RerankResult
from codeinsight.infrastructure.runtime_policy import DevelopmentPolicy
from codeinsight.infrastructure.sandbox import SandboxPreflightResult, SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


class FakeChatModel:
    model = "fake-chat"

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.text_prompts: list[str] = []

    def complete(self, _system_prompt: str, user_prompt: str) -> ModelCompletion:
        self.prompts.append(user_prompt)
        if "查询规划器" in _system_prompt:
            content = (
                '{"language":"zh","normalized_question":"checkout 如何校验输入？",'
                '"subquestions":[{"question":"checkout 如何校验输入？",'
                '"intent":"symbol_lookup","retrieval_mode":"bm25"}],'
                '"execution_route":"linear","confidence":0.95}'
            )
            return ModelCompletion(content, self.model, 12, 4, reasoning_content="正在判断查询路线")
        return ModelCompletion(
            '{"outcome":"answered","answer":"checkout 在入口处完成输入校验。",'
            '"citations":["E1"]}',
            self.model,
            18,
            7,
            reasoning_content="正在根据证据组织回答",
        )

    def complete_text(self, _system_prompt: str, user_prompt: str) -> ModelCompletion:
        self.text_prompts.append(user_prompt)
        return ModelCompletion(
            "你好！我是 CodeInsight，可以帮你理解和修改代码。", self.model, 10, 6
        )


class FakeEmbedding:
    def embed(self, texts):
        return EmbeddingBatch("fake", tuple((1.0, 0.0) for _ in texts), len(texts))


class FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


def _wait_for_terminal(client: TestClient, turn_id: str) -> dict:
    # The chat worker is intentionally asynchronous; GitHub-hosted runners can
    # take longer than the local fast path while starting the next session turn.
    # Keep polling bounded, but do not turn normal runner scheduling into a
    # false product failure.
    for _ in range(500):
        payload = client.get(f"/api/v2/chat/turns/{turn_id}").json()
        if payload["status"] in {"COMPLETED", "FAILED", "WAITING_APPROVAL"}:
            return payload
        time.sleep(0.02)
    raise AssertionError("聊天 Run 未在测试窗口内结束")


def test_chat_turns_stream_reasoning_and_restore_multi_turn_context() -> None:
    model = FakeChatModel()
    client = TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    )
    assert session.status_code == 200
    session_id = session.json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "checkout 如何校验输入？",
            "client_turn_id": "client-turn-1",
            "show_debug_reasoning": True,
        },
    )
    assert accepted.status_code == 202
    first = _wait_for_terminal(client, accepted.json()["turn_id"])
    assert first["status"] == "COMPLETED"
    assert first["task_type"] == "explain"
    assert "answer" not in first["result"]
    assert first["reasoning_available"] is True

    events = client.get(
        f"/api/v2/chat/turns/{first['turn_id']}/events"
    )
    assert events.status_code == 200
    assert "event: retrieval_started" in events.text
    assert "event: model_generating" in events.text
    assert "event: answer_ready" in events.text
    assert "event: debug_reasoning" in events.text
    assert "正在判断查询路线" in events.text
    # reasoning 未进入可回放的公开 RunEvent。
    event_log = client.app.state.codeinsight_conversation.runtime.event_log  # type: ignore[attr-defined]
    assert all(
        "正在判断查询路线" not in str(event.payload)
        for event in event_log.read_events(first["run_id"])
    )

    duplicate = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "checkout 如何校验输入？",
            "client_turn_id": "client-turn-1",
        },
    )
    assert duplicate.status_code == 202
    assert duplicate.json()["turn_id"] == first["turn_id"]
    assert len(model.prompts) == 2

    second_accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "刚才那个问题的入口文件是什么？",
        },
    )
    assert second_accepted.status_code == 202
    second = _wait_for_terminal(client, second_accepted.json()["turn_id"])
    assert second["status"] == "COMPLETED"
    assert any("turn 1 user" in prompt for prompt in model.prompts)
    assert any("turn 2 assistant" in prompt for prompt in model.prompts)

    restored = client.get(
        f"/api/v2/chat/sessions/{session_id}",
        params={"repository_root": str(FIXTURE_ROOT)},
    )
    assert restored.status_code == 200
    turns = restored.json()["recent_turns"]
    assert [item["role"] for item in turns][-4:] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_general_chat_uses_text_route_without_repository_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeChatModel()

    class ExplodingEmbedding:
        def __init__(self) -> None:
            raise AssertionError("普通对话不应创建 embedding 模型")

    client = TestClient(
        create_app(
            lambda: model,
            ExplodingEmbedding,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    def fail_if_repository_is_scanned(_root):
        raise AssertionError("Session 已绑定后，普通对话不应重新扫描仓库")

    monkeypatch.setattr(
        "codeinsight.application.conversation_service.scan_repository",
        fail_if_repository_is_scanned,
    )

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "你好",
        },
    )
    result = _wait_for_terminal(client, accepted.json()["turn_id"])

    assert result["status"] == "COMPLETED"
    assert result["task_type"] == "general_chat"
    assert result["assistant_message"] == "你好！我是 CodeInsight，可以帮你理解和修改代码。"
    assert result["result"]["kind"] == "general_chat"
    assert "answer" not in result["result"]
    assert model.text_prompts
    events = client.get(f"/api/v2/chat/turns/{result['turn_id']}/events")
    assert "event: retrieval_started" not in events.text
    assert "event: answer_ready" in events.text


def test_scope_redirect_is_natural_but_does_not_call_model() -> None:
    model = FakeChatModel()
    model_factory_calls = 0

    def model_factory():
        nonlocal model_factory_calls
        model_factory_calls += 1
        return model

    class ExplodingEmbedding:
        def __init__(self) -> None:
            raise AssertionError("范围引导不应创建 embedding 模型")

    client = TestClient(
        create_app(
            model_factory,
            ExplodingEmbedding,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "我喜欢打篮球",
        },
    )
    result = _wait_for_terminal(client, accepted.json()["turn_id"])

    assert result["status"] == "COMPLETED"
    assert result["task_type"] == "scope_redirect"
    assert result["result"] == {
        "kind": "scope_redirect",
        "outcome": "redirected",
        "route": "scope_redirect",
        "reason": "out_of_scope",
        "model_called": False,
        "prompt_version": "scope-redirect-v1",
    }
    assert "CodeInsight" in result["assistant_message"]
    assert "workflow.py" in result["assistant_message"]
    assert model_factory_calls == 0
    assert not model.text_prompts
    assert not model.prompts

    events = client.get(f"/api/v2/chat/turns/{result['turn_id']}/events")
    assert '"kind": "scope_redirect"' in events.text
    assert '"model_called": "false"' in events.text


def test_ambiguous_chat_turn_returns_clarification_without_model_call() -> None:
    model = FakeChatModel()

    class ExplodingEmbedding:
        def __init__(self) -> None:
            raise AssertionError("澄清轮次不应创建 embedding 模型")

    client = TestClient(
        create_app(
            lambda: model,
            ExplodingEmbedding,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "这个怎么样",
        },
    )
    result = _wait_for_terminal(client, accepted.json()["turn_id"])

    assert result["status"] == "COMPLETED"
    assert result["task_type"] == "clarify"
    assert result["result"]["kind"] == "clarify"
    assert "还不能确定" in result["assistant_message"]
    assert not model.prompts
    assert not model.text_prompts


def test_active_code_goal_keeps_pronoun_followup_on_code_route() -> None:
    model = FakeChatModel()
    client = TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(FIXTURE_ROOT)}
    ).json()["session_id"]

    first = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "checkout 如何校验输入？",
        },
    )
    _wait_for_terminal(client, first.json()["turn_id"])
    second = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(FIXTURE_ROOT),
            "message": "这个怎么样",
        },
    )
    result = _wait_for_terminal(client, second.json()["turn_id"])

    assert result["task_type"] == "explain"
    assert result["status"] == "COMPLETED"


class PassingSandbox:
    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class RetryableSandbox:
    def __init__(self) -> None:
        self.calls = 0

    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        self.calls += 1
        if self.calls == 1:
            return SandboxResult(
                profile,
                ("python -m compileall -q .",),
                False,
                "SANDBOX_PERMISSION_DENIED",
                "docker_engine: Access is denied",
            )
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class DevModeUnavailableSandbox:
    def preflight(self, profile):
        return SandboxPreflightResult(
            profile, False, "SANDBOX_PERMISSION_DENIED", "docker_engine: Access is denied"
        )

    def run(self, profile, workspace_path):
        raise AssertionError("开发模式跳过 Sandbox 后不应调用 run")


class FakeChangeModel:
    model = "fake-change"

    def __init__(self) -> None:
        self.round = 0

    def complete(self, system_prompt: str, _user_prompt: str) -> ModelCompletion:
        if "查询规划器" in system_prompt:
            content = (
                '{"language":"zh","normalized_question":"检查派送线路函数",'
                '"subquestions":[{"question":"检查派送线路函数",'
                '"intent":"implementation","retrieval_mode":"bm25"}],'
                '"execution_route":"linear","confidence":0.95}'
            )
        else:
            content = (
                '{"outcome":"answered","answer":"当前代码未发现需要继续处理的 bug。",'
                '"citations":["E1"]}'
            )
        return ModelCompletion(content, self.model, 18, 7)

    def complete_with_tools(self, _messages, _tools):
        self.round += 1
        if self.round == 1:
            return ToolModelResponse(
                None, (ToolCall("read-1", "read_file", {"path": "app.py"}),), self.model
            )
        if self.round == 2:
            return ToolModelResponse(
                None,
                (
                    ToolCall(
                        "patch-1",
                        "generate_patch",
                        {"path": "app.py", "new_content": "value = 2\n"},
                    ),
                ),
                self.model,
            )
        return ToolModelResponse("已生成修改预览。", (), self.model)


def test_change_turn_stops_at_preview_until_chat_approval(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_file = repo / "app.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    changes = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=PassingSandbox(),
    )
    model = FakeChangeModel()
    client = TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            changes,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(repo)}
    ).json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": "把 value 修改成 2",
        },
    )
    assert accepted.status_code == 202
    preview_turn = _wait_for_terminal(client, accepted.json()["turn_id"])
    assert preview_turn["status"] == "WAITING_APPROVAL"
    assert preview_turn["result"]["kind"] == "change_preview"
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"
    preview_events = client.get(
        f"/api/v2/chat/turns/{preview_turn['turn_id']}/events"
    )
    assert preview_events.status_code == 200
    assert "event: approval_requested" in preview_events.text
    assert '"to_status": "WAITING_APPROVAL"' in preview_events.text
    preview_cursor = max(
        int(line.rsplit(":", 1)[1])
        for line in preview_events.text.splitlines()
        if line.startswith("id: ")
    )

    approved = client.post(
        f"/api/v2/chat/turns/{preview_turn['turn_id']}/approve"
    )
    assert approved.status_code == 202
    final_turn = _wait_for_terminal(client, preview_turn["turn_id"])
    assert final_turn["status"] == "COMPLETED"
    assert final_turn["result"]["kind"] == "change_result"
    assert final_turn["result"]["status"] == "COMPLETED"
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"
    change_events = client.get(
        f"/api/v2/chat/turns/{preview_turn['turn_id']}/events",
        params={"after_sequence": preview_cursor},
    )
    assert change_events.status_code == 200
    assert "event: approval_granted" in change_events.text
    assert "event: validation_started" in change_events.text
    assert "event: validation_finished" in change_events.text
    assert "event: run_finished" in change_events.text
    restored = client.get(
        f"/api/v2/chat/sessions/{session_id}",
        params={"repository_root": str(repo)},
    )
    assert restored.status_code == 200
    assert restored.json()["recent_turns"][-1]["content"] == "修改已应用并完成校验。"

    followup = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": "现在再检查派送线路函数还有没有 bug。",
        },
    )
    assert followup.status_code == 202
    followup_turn = _wait_for_terminal(client, followup.json()["turn_id"])
    assert followup_turn["status"] == "COMPLETED"
    assert "未发现需要继续处理的 bug" in followup_turn["assistant_message"]
    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    effective_root = Path(conversation._turn_inputs[followup_turn["turn_id"]].repository_root)
    assert effective_root != repo.resolve()
    assert (effective_root / "app.py").read_text(encoding="utf-8") == "value = 2\n"


def test_development_mode_auto_approves_change_and_marks_validation_skipped(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_file = repo / "app.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    changes = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=DevModeUnavailableSandbox(),
        development_policy=DevelopmentPolicy("development", True, True, True),
    )
    client = TestClient(
        create_app(
            lambda: FakeChangeModel(),
            FakeEmbedding,
            changes,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(repo)}
    ).json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": "把 value 修改成 2",
        },
    )
    result = _wait_for_terminal(client, accepted.json()["turn_id"])

    assert result["status"] == "COMPLETED"
    assert result["result"]["kind"] == "change_result"
    assert result["result"]["validation"]["skipped"] is True
    assert "开发模式" in result["assistant_message"]
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"
    events = client.get(f"/api/v2/chat/turns/{result['turn_id']}/events")
    assert '"source": "dev_mode"' in events.text
    assert '"passed": "skipped"' in events.text


def test_continue_after_sandbox_failure_retries_validation_without_duplicate_patch(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    source_file = repo / "app.py"
    source_file.write_text("value = 1\n", encoding="utf-8")
    sandbox = RetryableSandbox()
    changes = ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=sandbox,
    )
    model = FakeChangeModel()
    client = TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            changes,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    session_id = client.post(
        "/api/v2/chat/sessions", json={"repository_root": str(repo)}
    ).json()["session_id"]

    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": "把 value 修改成 2",
        },
    )
    preview_turn = _wait_for_terminal(client, accepted.json()["turn_id"])
    approved = client.post(
        f"/api/v2/chat/turns/{preview_turn['turn_id']}/approve"
    )
    first_result = _wait_for_terminal(client, preview_turn["turn_id"])
    assert approved.status_code == 202
    assert first_result["result"]["status"] == "REVIEW_REQUIRED"
    model_calls_after_first_apply = model.round

    continued = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": "继续完成修改",
        },
    )
    continued_result = _wait_for_terminal(client, continued.json()["turn_id"])

    assert continued_result["status"] == "COMPLETED"
    assert continued_result["result"]["status"] == "COMPLETED"
    assert sandbox.calls == 2
    assert model.round == model_calls_after_first_apply
    events = client.get(
        f"/api/v2/chat/turns/{continued_result['turn_id']}/events"
    )
    assert "event: validation_finished" in events.text
    assert "event: tool_call_requested" not in events.text
    assert source_file.read_text(encoding="utf-8") == "value = 1\n"
