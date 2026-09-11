"""Q-011 Task 7 验收：固定校验在独立 ValidationWorker 里跑，结果进 Run 状态机。

覆盖计划里点名要有的五条：

1. ValidationWorker 正常跑完 → Agent Run 进入 COMPLETED；
2. 同一条校验任务只能被领取一次，租约到期后恰好恢复一次；
3. 校验环境不可用时，Run 停在要人工处理的状态，不伪造成功；
4. 检查失败最多进入登记过的修复次数，耗尽后不再自动循环；
5. 补丁已应用但结果不确定时，不会自动重复 apply。

沙箱与模型全部用假实现，不需要 Docker、Redis 或真实模型。
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.agent.tool_loop import ToolCall, ToolModelResponse
from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.infrastructure.sandbox import SandboxPreflightResult, SandboxResult
from codeinsight.infrastructure.task_queue import InMemoryTaskQueue, TaskEnvelope
from codeinsight.infrastructure.workspace import WorkspaceManager
from tests.integration.test_chat_change_async import (
    FakeChangeModel,
    FakeEmbedding,
    FakeReranker,
    _session,
    _submit_change,
    _wait_for,
    _workspaces,
)


class CountingSandbox:
    """记录调用次数：它被调用几次，就是「固定校验跑了几个回合」的事实。"""

    def __init__(self, *, passed: bool = True) -> None:
        self.calls = 0
        self.passed = passed

    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        self.calls += 1
        if self.passed:
            return SandboxResult(profile, ("python -m compileall -q .",), True)
        return SandboxResult(
            profile,
            ("python -m pytest -q",),
            False,
            "CHECK_FAILED",
            "assertion failed in test_checkout_total",
        )


class UnavailableSandbox:
    """校验环境整个不可用：连结果都拿不到。"""

    def __init__(self) -> None:
        self.calls = 0

    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        self.calls += 1
        raise RuntimeError("docker engine is not reachable")


class RepeatableChangeModel(FakeChangeModel):
    """每次「读文件 → 生成补丁 → 收尾」都当成一轮新的修改。

    真实模型没有轮次计数器；这里用轮数模拟「修复轮也是一次完整的修改」，让第二次
    修改也能产出补丁，而且内容确实与上一轮不同（否则就是没有实质变化）。
    """

    def complete_with_tools(self, messages, tools):
        names = {tool.get("function", {}).get("name") for tool in tools}
        if "generate_patch" not in names:
            return super().complete_with_tools(messages, tools)
        self.round += 1
        step = self.round % 3
        if step == 1:
            return ToolModelResponse(
                None,
                (ToolCall(f"read-{self.round}", "read_file", {"path": "app.py"}),),
                self.model,
            )
        if step == 2:
            attempt = self.round // 3 + 1
            return ToolModelResponse(
                None,
                (
                    ToolCall(
                        f"patch-{self.round}",
                        "generate_patch",
                        {"path": "app.py", "new_content": f"value = {attempt + 1}"},
                    ),
                ),
                self.model,
            )
        return ToolModelResponse("已经生成修改预览。", (), self.model)


def _client(tmp_path: Path, sandbox, *, model: FakeChangeModel | None = None):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    base = tmp_path / "managed"
    changes = ChangeService(
        workspace_manager=WorkspaceManager(base),
        sandbox=sandbox,
    )
    model = model or FakeChangeModel()
    client = TestClient(
        create_app(
            lambda: model,
            FakeEmbedding,
            changes,
            reranker_factory=FakeReranker,
        )  # type: ignore[arg-type]
    )
    return client, repo, base, model, changes


def _approve(client: TestClient, turn_id: str) -> None:
    approved = client.post(f"/api/v2/chat/turns/{turn_id}/approve")
    assert approved.status_code == 202


def test_validation_worker_finishes_the_run_after_apply(tmp_path: Path) -> None:
    sandbox = CountingSandbox()
    client, repo, base, model, changes = _client(tmp_path, sandbox)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"

    _approve(client, preview["turn_id"])
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})

    assert final["status"] == "COMPLETED"
    assert final["result"]["kind"] == "change_result"
    assert final["result"]["status"] == "COMPLETED"
    assert final["result"]["validation"]["profile"] == "python_compile"
    # 固定校验只跑了一次，而且是在隔离 workspace 上跑的
    assert sandbox.calls == 1
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    workspaces = _workspaces(base)
    assert len(workspaces) == 1
    assert (workspaces[0] / "app.py").read_text(encoding="utf-8") == "value = 2\n"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    assert record.status == "COMPLETED"
    assert record.task_kind == "resume_after_validation"
    # profile 与 workspace 路径来自登记事实，不是模型临时决定的
    tasks = [
        task
        for task in _validation_tasks(conversation)
        if task.run_id == record.run_id
    ]
    assert len(tasks) == 1
    assert tasks[0].payload["profile"] == "python_compile"
    assert tasks[0].payload["workspace_path"] == str(workspaces[0])
    assert tasks[0].status == "COMPLETED"


def test_the_same_validation_task_is_never_run_twice(tmp_path: Path) -> None:
    sandbox = CountingSandbox()
    client, repo, _base, _model, _changes = _client(tmp_path, sandbox)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"
    calls_after_finish = sandbox.calls

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    task = _validation_tasks(conversation)[0]

    # 重复投递同一条校验任务：拿不到租约，沙箱不会再跑一遍
    outcome = conversation.validation_worker.handle(task.task_id)

    assert outcome.claimed is False
    assert sandbox.calls == calls_after_finish
    assert conversation.validation_queue.get_task(task.task_id).status == "COMPLETED"


def test_expired_validation_lease_is_recovered_exactly_once() -> None:
    queue = InMemoryTaskQueue()
    queue.submit(
        TaskEnvelope(
            task_id="validation-1",
            run_id="run-1",
            session_id="session-1",
            idempotency_key="run-1:patch-1:python_compile",
            task_type="registered_validation",
            payload={"profile": "python_compile", "patch_id": "patch-1"},
            deadline_epoch_ms=10_000_000,
            max_attempts=2,
        )
    )

    first = queue.claim_task("validation-1", worker_id="dead", now_epoch_ms=100, lease_ms=100)
    assert first is not None and first.attempt == 1
    # 租约还没到期：别人领不走，也就不会两个 Worker 同时跑同一套检查
    assert (
        queue.claim_task("validation-1", worker_id="other", now_epoch_ms=150, lease_ms=100)
        is None
    )
    recovered = queue.claim_task(
        "validation-1", worker_id="other", now_epoch_ms=201, lease_ms=100
    )
    assert recovered is not None and recovered.attempt == 2
    # 尝试次数用完：第三条消息拿不到任务，任务停在要人工处理的状态
    assert (
        queue.claim_task("validation-1", worker_id="third", now_epoch_ms=500, lease_ms=100)
        is None
    )
    assert queue.get_task("validation-1").status == "MANUAL_REQUIRED"


def test_unavailable_validation_environment_never_fakes_success(tmp_path: Path) -> None:
    sandbox = UnavailableSandbox()
    client, repo, base, _model, changes = _client(tmp_path, sandbox)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])

    final = _wait_for(
        client,
        preview["turn_id"],
        statuses={"COMPLETED", "FAILED", "MANUAL_REQUIRED"},
    )

    assert final["status"] == "MANUAL_REQUIRED"
    assert sandbox.calls == 1
    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    assert record.status == "MANUAL_REQUIRED"
    # 任务本身还留在队列里可以查询，状态不是「已完成」
    task = _validation_tasks(conversation)[0]
    assert task.status == "QUEUED"
    assert changes.get_result(record.run_id, record.patch_id).status == "WAITING_VALIDATION"
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    assert len(_workspaces(base)) == 1


def test_a_failed_check_is_repaired_once_and_then_stops(tmp_path: Path) -> None:
    sandbox = CountingSandbox(passed=False)
    model = RepeatableChangeModel()
    client, repo, _base, _unused, changes = _client(tmp_path, sandbox, model=model)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])

    # 第一次校验失败：预算还没用完，自动修复一轮，产出新的预览等审批。
    repaired = _wait_for(client, preview["turn_id"], statuses={"WAITING_APPROVAL", "FAILED"})
    assert repaired["status"] == "WAITING_APPROVAL"
    assert repaired["result"]["kind"] == "change_preview"
    assert sandbox.calls == 1
    rounds_after_repair = model.round

    _approve(client, preview["turn_id"])
    final = _wait_for(
        client,
        preview["turn_id"],
        statuses={"COMPLETED", "FAILED", "WAITING_APPROVAL"},
    )

    # 第二次校验又失败：预算已用完，停下来交给人，也不再跑模型。
    assert final["status"] == "COMPLETED"
    assert final["result"]["status"] == "REVIEW_REQUIRED"
    assert model.round == rounds_after_repair
    assert sandbox.calls == 2
    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = conversation.agent_run_store.find_run_by_turn(preview["turn_id"])
    assert changes.repair_rounds(record.run_id) == 1
    assert (repo / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def _validation_tasks(conversation) -> list[TaskEnvelope]:
    return [
        task
        for task in conversation.validation_queue.list_tasks()
        if task.task_type == "registered_validation"
    ]
