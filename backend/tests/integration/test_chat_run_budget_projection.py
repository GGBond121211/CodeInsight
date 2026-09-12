"""Q-012 U6 验收：花费闸的结论要能在一行事实里读到。

上游已经有两条原始事实（budget_limit / budget_used 事件）；Run 行上的
token_budget / tokens_used 是它们的摘要，用来回答「这一轮闸值多少、用了多少」，
而不必为每个 Run 翻整条事件流。

这条用例走真实 change 链路（Fake Provider + Fake Sandbox，不需要网络、Docker
或真实模型），断言摘要等于事件里的最后一条累计值——不是写入时顺手编的数字。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from codeinsight.api.app import create_app
from codeinsight.application.change_service import ChangeService
from codeinsight.application.conversation_service import ConversationService
from codeinsight.application.session_service import SessionService
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.model_gateway import configured_change_token_budget
from codeinsight.infrastructure.run_store import InMemoryAgentRunStore, InMemorySessionStore
from codeinsight.infrastructure.workspace import WorkspaceManager
from tests.integration.test_chat_change_async import (
    FakeChangeModel,
    FakeEmbedding,
    FakeReranker,
    PassingSandbox,
    _session,
    _submit_change,
)


class TokenReportingFakeChangeModel(FakeChangeModel):
    """行为与 FakeChangeModel 相同，只多一件事：每一轮都报一个非零用量。

    用量全为 0 时，「记下来了」和「根本没记」在断言上分不出来。
    """

    def complete_with_tools(self, messages, tools):
        response = super().complete_with_tools(messages, tools)
        return replace(
            response,
            input_tokens=(response.input_tokens or 0) + 1_200,
            output_tokens=(response.output_tokens or 0) + 300,
        )


def _client(tmp_path: Path, run_store: InMemoryAgentRunStore) -> TestClient:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("value = 1\n", encoding="utf-8")
    # 同一轮里 Router 与 Tool Loop 用的是同一个模型实例：换成每次新建，
    # FakeChangeModel 的轮次计数会从 1 重新开始，补丁就生成不出来。
    model = TokenReportingFakeChangeModel()
    service = ConversationService(
        lambda: model,
        FakeEmbedding,
        reranker_factory=FakeReranker,
        change_service=ChangeService(
            workspace_manager=WorkspaceManager(tmp_path / "managed"),
            sandbox=PassingSandbox(),
        ),
        session_service=SessionService(InMemorySessionStore(), InMemoryMemoryStore()),
        agent_run_store=run_store,
    )
    return TestClient(create_app(conversation_service=service))


def test_a_gated_turn_writes_its_budget_summary_onto_the_run_row(tmp_path: Path) -> None:
    run_store = InMemoryAgentRunStore()
    client = _client(tmp_path, run_store)
    repo = tmp_path / "repo"
    session_id = _session(client, repo)

    preview = _submit_change(client, session_id=session_id, repo=repo)
    assert preview["status"] == "WAITING_APPROVAL"

    conversation = client.app.state.codeinsight_conversation  # type: ignore[attr-defined]
    record = run_store.find_run_by_turn(preview["turn_id"])
    assert record is not None
    events = conversation.runtime.event_log.read_events(record.run_id)
    reported = [
        int(event.payload["budget_used"])
        for event in events
        if event.event_type == "budget_used"
    ]
    assert reported, "这一轮没有写下任何 budget_used 事件，摘要无从谈起"
    assert record.token_budget == configured_change_token_budget()
    assert record.tokens_used == reported[-1]
    # 非零用量证明它是从事件里读出来的，不是默认值。
    assert record.tokens_used > 0
