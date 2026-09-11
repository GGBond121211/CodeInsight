"""Q-011 Task 6 验收：审批只登记后台续跑，API 线程不碰 workspace。

四条性质对应 Task 6 的验收条件：

1. 批准之前，原仓库和隔离 workspace 都没有任何越权写入，事实层里也没有
   补丁凭据；
2. 批准只登记一次有效续跑：同一个 run_id 上 task_id 换新、attempt 加一；
3. 重复批准拿不到第二条续跑，也不会再跑一次模型；
4. Worker 在应用窗口里崩掉时，Run 停在要人工处理的 UNKNOWN，重投同一条
   消息不会被领取，也就不会把半应用的改动写第二遍。

全过程用 Fake Provider 与 Fake Sandbox，不需要网络、Docker 或真实模型。
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse
from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.domain.answer import ModelCompletion
from codeinsight.domain.semantic import EmbeddingBatch, SparseEmbedding
from codeinsight.infrastructure.reranker import RerankResult
from codeinsight.infrastructure.sandbox import SandboxPreflightResult, SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager


class FakeChangeModel:
    """只产生一次「读文件 + 生成补丁」的变更 Tool Loop。"""

    model = "fake-change"

    def __init__(self) -> None:
        self.round = 0

    def complete(self, system_prompt: str, _user_prompt: str) -> ModelCompletion:
        if "查询规划器" in system_prompt:
            content = (
                '{"language":"zh","normalized_question":"检查 app.py",'
                '"subquestions":[{"question":"检查 app.py",'
                '"intent":"implementation","retrieval_mode":"hybrid"}],'
                '"execution_route":"linear","confidence":0.95}'
            )
        else:
            content = (
                '{"outcome":"answered","answer":"当前代码未发现需要继续处理的 bug。",'
                '"citations":["E1"]}'
            )
        return ModelCompletion(content, self.model, 18, 7)

    def complete_with_tools(self, _messages, tools):
        names = {tool.get("function", {}).get("name") for tool in tools}
        if "generate_patch" not in names:
            return ToolModelResponse(
                '{"outcome":"answered","answer":"当前代码未发现需要继续处理的 bug。",'
                '"citations":["E1"]}',
                (),
                self.model,
                18,
                7,
            )
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
        return ToolModelResponse("已经生成修改预览。", (), self.model)


class FakeEmbedding:
    def embed(self, texts):
        return EmbeddingBatch(
            "fake",
            tuple((1.0, 0.0) for _ in texts),
            len(texts),
            tuple(SparseEmbedding((1,), (1.0,)) for _ in texts),
            "dense-sparse-v1",
        )


class FakeReranker:
    def rerank(self, _query, documents, *, top_n):
        return tuple(RerankResult(index, float(top_n - index)) for index in range(top_n))


class PassingSandbox:
    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class CrashingApplyService(ChangeService):
    """apply 直接崩，模拟 Worker 在应用窗口里失去控制权。"""

    def apply(self, run_id: str, patch_id: str, approval_token: str):
        raise RuntimeError("worker crashed inside the apply window")


CHANGE_MESSAGE = "把 value 改成 2"


def _build(tmp_path: Path, *, change_service: ChangeService | None = None):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    base = tmp_path / "managed"
    changes = change_service or ChangeService(
        workspace_manager=WorkspaceManager(base),
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
    return client, repo, base, model


def _workspaces(base: Path) -> list[Path]:
    return sorted(path for path in base.glob("ws-*") if path.is_dir())


def _wait_for(client: TestClient, turn_id: str, *, statuses: set[str]) -> dict:
    for _ in range(1000):
        payload = client.get(f"/api/v2/chat/turns/{turn_id}").json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"这一轮没有停在预期状态：{sorted(statuses)}")


def _submit_change(client: TestClient, *, session_id: str, repo: Path) -> dict:
    accepted = client.post(
        "/api/v2/chat/turns",
        json={
            "session_id": session_id,
            "repository_root": str(repo),
            "message": CHANGE_MESSAGE,
        },
    )
    assert accepted.status_code == 202
    return _wait_for(
        client,
        accepted.json()["turn_id"],
        statuses={"WAITING_APPROVAL", "FAILED", "COMPLETED"},
    )


def _session(client: TestClient, repo: Path) -> str:
    return client.post("/api/v2/chat/sessions", json={"repository_root": str(repo)}).json()[
        "session_id"
    ]


def _unexpected_runner(_context):
    raise AssertionError("未审批的 Run 不应进入执行体")


def test_approval_registers_continuation_and_worker_owns_apply(tmp_path: Path) -> None:
    client, repo, base, model = _build(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"
    assert preview["result"]["kind"] == "change_preview"
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert _workspaces(base) == []

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    assert record.status == "WAITING_APPROVAL"
    assert record.attempt == 1
    assert record.task_kind == "agent_run"
    # 批准之前，事实层里不该有补丁凭据：没有它们，任何重投都执行不了 apply。
    assert record.patch_id is None
    assert record.approval_token is None

    approved = client.post(f"/api/v2/chat/turns/{preview['turn_id']}/approve")
    assert approved.status_code == 202
    assert approved.json()["status"] == "QUEUED"

    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"
    assert final["result"]["kind"] == "change_result"
    assert final["result"]["status"] == "COMPLETED"
    # 源仓库始终没被改动；改动只落在 Worker 建的隔离 workspace 里
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    workspaces = _workspaces(base)
    assert len(workspaces) == 1
    assert (workspaces[0] / "app.py").read_text(encoding="utf-8") == "value = 2\n"
    assert model.round >= 2

    continued = conversation.agent_run_store.get_run(record.run_id)
    assert continued.status == "COMPLETED"
    assert continued.task_kind == "resume_after_approval"
    assert continued.attempt == 2
    assert continued.task_id != record.task_id
    assert continued.run_id == record.run_id


def test_duplicate_approval_cannot_register_a_second_continuation(tmp_path: Path) -> None:
    client, repo, base, model = _build(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"
    turn_id = preview["turn_id"]

    assert client.post(f"/api/v2/chat/turns/{turn_id}/approve").status_code == 202
    second = client.post(f"/api/v2/chat/turns/{turn_id}/approve")
    assert second.status_code == 409

    model_calls = model.round
    final = _wait_for(client, turn_id, statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"
    assert model.round == model_calls
    assert len(_workspaces(base)) == 1
    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(turn_id)
    assert record.attempt == 2


def test_redelivered_run_while_waiting_for_approval_never_applies(tmp_path: Path) -> None:
    """未批准的 Run 被重投时没人接：等待审批的状态不在可领取集合里。"""

    client, repo, base, _model = _build(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    outcome = conversation.agent_run_worker.handle(
        record.to_task(), runner=_unexpected_runner
    )
    assert outcome.claimed is False
    assert outcome.execution is None
    assert conversation.agent_run_store.get_run(record.run_id).status == "WAITING_APPROVAL"
    assert _workspaces(base) == []
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_worker_crash_in_apply_window_stops_at_unknown_and_is_not_retried(
    tmp_path: Path,
) -> None:
    changes = CrashingApplyService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=PassingSandbox(),
    )
    client, repo, base, _model = _build(tmp_path, change_service=changes)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"

    assert client.post(f"/api/v2/chat/turns/{preview['turn_id']}/approve").status_code == 202
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "FAILED"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    assert record.status == "UNKNOWN"
    assert record.error_class is not None
    assert record.error_class.startswith("UNEXPECTED_ERROR")

    # 重投同一条消息：UNKNOWN 不是可领取状态，Worker 不接手，也就不会把
    # 可能已经写了一半的改动再写一遍。
    outcome = conversation.agent_run_worker.handle(
        record.to_task(), runner=_unexpected_runner
    )
    assert outcome.claimed is False
    assert conversation.agent_run_store.get_run(record.run_id).status == "UNKNOWN"
    assert _workspaces(base) == []
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
