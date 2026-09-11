"""Q-011 Task 8 验收：事件先落事实，再推给连接；断线重连靠 Store 补齐。

四条性质：

1. 回放与广播按 sequence 去重：重连既不重复显示，也不跳过事件；
2. 没人连接时写下的事件照样落事实，重连能补齐；
3. API 重启（换一个运行时，同一份事件事实）之后仍能回放；
4. 等审批、等校验、状态不明在界面上是三个不同的说法。

事件 Store 在本地用内存实现——它是「进程之外的事实」的替身；生产用 MySQL 的
run_events 表。全过程用 Fake Provider 与假沙箱，不需要网络、Docker 或真实模型。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.application.conversation_service import ConversationService
from codeinsight.infrastructure.chat_runtime import ChatRuntime
from codeinsight.infrastructure.event_log import InMemoryEventLog, LiveEventLog
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore
from codeinsight.infrastructure.sandbox import SandboxPreflightResult, SandboxResult
from codeinsight.infrastructure.workspace import WorkspaceManager
from tests.integration.test_chat_change_async import (
    CrashingApplyService,
    FakeChangeModel,
    FakeEmbedding,
    FakeReranker,
    _session,
    _submit_change,
    _wait_for,
)


class PassingSandbox:
    """固定校验直接通过：这里要验的是事件，不是沙箱结论。"""

    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        return SandboxResult(profile, ("python -m compileall -q .",), True)


class GateSandbox:
    """把固定校验卡住，用来观察「等校验」这个中间状态。"""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def preflight(self, profile):
        return SandboxPreflightResult(profile, True)

    def run(self, profile, workspace_path):
        self.entered.set()
        assert self.release.wait(timeout=30)
        return SandboxResult(profile, ("python -m compileall -q .",), True)


def _event_frames(body: str) -> list[dict[str, object]]:
    """SSE 文本 → 公开事件帧。keep-alive 与调试 reasoning 帧不算事件。"""

    frames: list[dict[str, object]] = []
    for block in body.split("\n\n"):
        payloads = [
            line[len("data: ") :]
            for line in block.splitlines()
            if line.startswith("data: ")
        ]
        if not payloads:
            continue
        frame = json.loads(payloads[0])
        if "sequence" in frame:
            frames.append(frame)
    return frames


def _stream(
    client: TestClient, turn_id: str, *, after_sequence: int
) -> list[dict[str, object]]:
    response = client.get(
        f"/api/v2/chat/turns/{turn_id}/events",
        params={"after_sequence": after_sequence},
    )
    assert response.status_code == 200
    return _event_frames(response.text)


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    return repo


def _app(
    tmp_path: Path,
    *,
    repo: Path,
    sandbox,
    event_store=None,
    run_store=None,
    changes: ChangeService | None = None,
) -> TestClient:
    """装配一个可以「重启」的应用：事件事实与 Run 事实都能从外面传进来。"""

    selected_changes = changes or ChangeService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=sandbox,
    )
    service = ConversationService(
        lambda: FakeChangeModel(),
        FakeEmbedding,
        reranker_factory=FakeReranker,
        change_service=selected_changes,
        runtime=ChatRuntime(event_log=LiveEventLog(event_store or InMemoryEventLog())),
        agent_run_store=run_store or InMemoryAgentRunStore(),
    )
    return TestClient(create_app(conversation_service=service))


def _simple_app(tmp_path: Path, *, sandbox=None) -> tuple[TestClient, Path]:
    repo = _make_repo(tmp_path)
    return _app(tmp_path, repo=repo, sandbox=sandbox or PassingSandbox()), repo


def _approve(client: TestClient, turn_id: str) -> None:
    assert client.post(f"/api/v2/chat/turns/{turn_id}/approve").status_code == 202


def test_reconnect_resumes_from_the_last_sequence_without_gaps_or_repeats(
    tmp_path: Path,
) -> None:
    client, repo = _simple_app(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"

    everything = _stream(client, preview["turn_id"], after_sequence=0)
    sequences = [frame["sequence"] for frame in everything]
    assert sequences == list(range(1, len(sequences) + 1))

    cut = sequences[2]
    resumed = _stream(client, preview["turn_id"], after_sequence=cut)

    assert [frame["sequence"] for frame in resumed] == [
        sequence for sequence in sequences if sequence > cut
    ]
    assert resumed[0]["sequence"] == cut + 1


def test_events_written_while_nobody_is_connected_are_replayed_on_reconnect(
    tmp_path: Path,
) -> None:
    client, repo = _simple_app(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)

    # 第一次连接只看到「等审批」就断开；断开不等于取消。
    first = _stream(client, preview["turn_id"], after_sequence=0)
    assert first[-1]["event_type"] == "state_transitioned"
    assert first[-1]["payload"]["to_status"] == "WAITING_APPROVAL"
    last_sequence = first[-1]["sequence"]

    # 断开之后才批准：apply、固定校验和收尾全部发生在没有连接的时候。
    _approve(client, preview["turn_id"])
    assert _wait_for(client, preview["turn_id"], statuses={"COMPLETED"})["status"] == (
        "COMPLETED"
    )

    resumed = _stream(client, preview["turn_id"], after_sequence=last_sequence)
    event_types = [frame["event_type"] for frame in resumed]

    assert resumed[0]["sequence"] == last_sequence + 1
    assert "validation_queued" in event_types
    assert "validation_finished" in event_types
    assert event_types[-1] == "run_finished"


def test_persisted_events_replay_after_an_api_restart(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    event_store = InMemoryEventLog()
    run_store = InMemoryAgentRunStore()
    first = _app(
        tmp_path,
        repo=repo,
        sandbox=PassingSandbox(),
        event_store=event_store,
        run_store=run_store,
    )
    session_id = _session(first, repo)
    preview = _submit_change(first, session_id=session_id, repo=repo)
    _approve(first, preview["turn_id"])
    assert _wait_for(first, preview["turn_id"], statuses={"COMPLETED"})["status"] == (
        "COMPLETED"
    )
    before = _stream(first, preview["turn_id"], after_sequence=0)

    # 重启：新的应用、新的运行时、新的进程内状态；只有两份事实留下来。
    second = _app(
        tmp_path,
        repo=repo,
        sandbox=PassingSandbox(),
        event_store=event_store,
        run_store=run_store,
    )
    after = _stream(second, preview["turn_id"], after_sequence=0)

    assert [frame["sequence"] for frame in after] == [
        frame["sequence"] for frame in before
    ]
    assert [frame["event_type"] for frame in after] == [
        frame["event_type"] for frame in before
    ]


def test_waiting_for_approval_is_its_own_status(tmp_path: Path) -> None:
    client, repo = _simple_app(tmp_path)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)

    assert preview["status"] == "WAITING_APPROVAL"
    frames = _stream(client, preview["turn_id"], after_sequence=0)
    assert frames[-1]["payload"]["to_status"] == "WAITING_APPROVAL"


def test_waiting_for_validation_is_its_own_status(tmp_path: Path) -> None:
    gate = GateSandbox()
    client, repo = _simple_app(tmp_path, sandbox=gate)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])

    waiting = _wait_for(
        client, preview["turn_id"], statuses={"WAITING_VALIDATION", "FAILED"}
    )

    assert waiting["status"] == "WAITING_VALIDATION"
    assert waiting["result"]["kind"] == "change_pending_validation"
    assert waiting["result"]["validation"]["profile"] == "python_compile"
    assert gate.entered.wait(timeout=10) is True

    gate.release.set()
    final = _wait_for(client, preview["turn_id"], statuses={"COMPLETED", "FAILED"})
    assert final["status"] == "COMPLETED"


def test_an_unknown_apply_window_is_shown_as_unknown(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    changes = CrashingApplyService(
        workspace_manager=WorkspaceManager(tmp_path / "managed"),
        sandbox=PassingSandbox(),
    )
    client = _app(tmp_path, repo=repo, sandbox=PassingSandbox(), changes=changes)
    session_id = _session(client, repo)
    preview = _submit_change(client, session_id=session_id, repo=repo)
    _approve(client, preview["turn_id"])

    final = _wait_for(
        client, preview["turn_id"], statuses={"COMPLETED", "FAILED", "UNKNOWN"}
    )

    assert final["status"] == "UNKNOWN"
    assert "UNEXPECTED_ERROR" in final["error"]
