"""Q-011 收尾：执行换到别的进程后，读接口仍要能回答。

API 进程与 Worker 进程的内存互不可见。这一组用例把「两个进程」摆在同一份事实层上：
一个只受理（broker 通道，本进程不执行），另一个执行。断言的是跨过进程边界仍然成立的
三件事——202 说的是受理的那一轮、读接口能拿回另一个进程写下的答案、审批能读到事实层
里的预览。

它不能替代真实 Redis + 独立 Worker 进程的取证（那个在验收记录里单独留痕），但它能挡住
「按本进程内存回答」这类回归：那种实现单进程测试全绿，换进程就全部失效。
"""

from __future__ import annotations

import time
from pathlib import Path

from codeinsight.application.agent_run_dispatcher import InMemoryAgentRunTransport
from codeinsight.application.conversation_service import (
    CHAT_RUN_POLICY_VERSION,
    INDEX_VERSION,
    ConversationService,
)
from codeinsight.application.session_service import SessionService
from codeinsight.domain.agent_run import (
    TASK_AGENT_RUN,
    TASK_RESUME_AFTER_APPROVAL,
    WAITING_APPROVAL,
    AgentRunRecord,
    RunOutput,
)
from codeinsight.domain.chat import (
    CHAT_COMPLETED,
    CHAT_QUEUED,
    CHAT_WAITING_APPROVAL,
)
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.redis_cache import InMemoryCache
from codeinsight.infrastructure.run_store import (
    InMemoryAgentRunStore,
    InMemorySessionStore,
)
from tests.integration.test_chat_api import (
    FakeChatModel,
    FakeEmbedding,
    FakeReranker,
    FakeToolLoopClient,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "sample_repo"


class RecordingChangeService:
    """只记下审批登记。这一组用例要证的是「预览从事实层读到」，不是变更流程本身。"""

    def __init__(self) -> None:
        self.approvals: list[tuple[str, str]] = []

    def approve(self, run_id: str, patch_id: str) -> str:
        self.approvals.append((run_id, patch_id))
        return "token-from-approval"


def _service(
    *,
    store: InMemoryAgentRunStore,
    sessions: SessionService,
    transport=None,
    change_service=None,
) -> ConversationService:
    return ConversationService(
        FakeChatModel,
        FakeEmbedding,
        reranker_factory=FakeReranker,
        change_service=change_service,
        session_service=sessions,
        agent_run_store=store,
        mcp_client_factory=FakeToolLoopClient,
        agent_run_transport=transport,
    )


def _shared_facts() -> tuple[InMemoryAgentRunStore, SessionService]:
    """两个「进程」共用的那一份事实层。"""

    return InMemoryAgentRunStore(), SessionService(
        InMemorySessionStore(), InMemoryMemoryStore()
    )


def _shared_facts_with_cache() -> tuple[InMemoryAgentRunStore, SessionService]:
    """共用事实层 + 共享热点缓存。真机上这份缓存是 Redis，本机不配 Redis 时是 None。

    只有开着缓存时才会出现「同一会话两套缓存键」，所以这一类回归必须带缓存跑。
    """

    return InMemoryAgentRunStore(), SessionService(
        InMemorySessionStore(), InMemoryMemoryStore(), InMemoryCache()
    )


def _seed_user_turn(sessions: SessionService, session_id: str, message: str) -> None:
    """按受理路径的方式把用户这一轮写进会话事实。

    受理是「先写用户消息、再投递」：Run 一旦存在，这句话在会话事实里就已经在了。
    手工造 Run 的用例必须照做，否则造出来的是一个真实链路到不了的状态。
    """

    session = sessions.load_session(session_id)
    assert session is not None
    context = sessions.get_or_create_session(
        session_id=session_id,
        scope=session.scope,
        repo_id=session.repo_id,
        repo_fingerprint="seed-fingerprint",
        index_version=INDEX_VERSION,
    )
    sessions.append_turn(
        context,
        role="user",
        content=message,
        repo_fingerprint="seed-fingerprint",
        index_version=INDEX_VERSION,
    )


def test_accepted_turn_is_projected_from_facts_not_from_local_memory() -> None:
    """受理进程内存里没有这一轮时，202 仍要说出受理的是哪一轮。"""

    store, sessions = _shared_facts()
    transport = InMemoryAgentRunTransport()
    api = _service(store=store, sessions=sessions, transport=transport)

    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    accepted = api.submit_turn(
        session_id=session_id,
        repository_root=str(FIXTURE_ROOT),
        message="checkout 如何校验输入？",
    )

    # 只受理、不执行：这一轮只存在于事实层，本进程运行时里没有它。
    assert accepted.status == CHAT_QUEUED
    assert accepted.run_id
    assert api.runtime.get_turn_or_none(accepted.turn_id) is None

    # 事实层里必须已经记下「谁在跑、跑的是哪一轮」。
    record = store.find_run_by_turn(accepted.turn_id)
    assert record is not None
    assert record.run_id == accepted.run_id
    assert record.status == CHAT_QUEUED
    assert transport.published[-1].turn_id == accepted.turn_id


def test_reader_returns_answer_written_by_another_process() -> None:
    """执行在别的进程，读接口要靠 agent_runs + agent_run_outputs 回答。"""

    store, sessions = _shared_facts()
    transport = InMemoryAgentRunTransport()
    api = _service(store=store, sessions=sessions, transport=transport)
    worker = _service(store=store, sessions=sessions)

    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    accepted = api.submit_turn(
        session_id=session_id,
        repository_root=str(FIXTURE_ROOT),
        message="checkout 如何校验输入？",
    )
    task = transport.published[-1]

    outcome = worker.run_agent_task(task)
    assert outcome.claimed is True

    # 执行进程见过这一轮；受理进程到现在仍然没见过——跨进程只能靠事实。
    assert worker.runtime.get_turn_or_none(accepted.turn_id) is not None
    assert api.runtime.get_turn_or_none(accepted.turn_id) is None

    projected = api.resolve_turn(accepted.turn_id)
    assert projected.status == CHAT_COMPLETED
    assert projected.assistant_message
    assert projected.user_message == "checkout 如何校验输入？"
    assert projected.error is None

    # run_id 与 turn_id 是同一轮的两个入口，必须指向同一个答案。
    by_run = api.resolve_turn_by_run_id(accepted.run_id)
    assert by_run.turn_id == accepted.turn_id
    assert by_run.assistant_message == projected.assistant_message


def test_approval_reads_its_preview_from_facts() -> None:
    """审批进程内存里没有这一轮时，预览与补丁标识只能来自产出事实。"""

    store, sessions = _shared_facts()
    transport = InMemoryAgentRunTransport()
    change_service = RecordingChangeService()
    api = _service(
        store=store, sessions=sessions, transport=transport, change_service=change_service
    )

    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    _seed_user_turn(sessions, session_id, "把 value 改成 2")
    now = int(time.time() * 1000)
    run_id = "chat-run-cross-process"
    turn_id = "turn-cross-process"
    store.save_run(
        AgentRunRecord(
            run_id=run_id,
            turn_id=turn_id,
            session_id=session_id,
            task_id="task-cross-process",
            task_kind=TASK_AGENT_RUN,
            status=WAITING_APPROVAL,
            policy_version=CHAT_RUN_POLICY_VERSION,
            idempotency_key=turn_id,
            deadline_epoch_ms=now + 60_000,
            updated_at_epoch_ms=now,
        )
    )
    store.save_output(
        RunOutput(
            run_id=run_id,
            attempt=1,
            task_type="change",
            assistant_message="准备把 value 改成 2，等你确认。",
            result={"preview": {"patch_id": "patch-from-facts", "summary": "value: 1 -> 2"}},
            updated_at_epoch_ms=now,
        )
    )
    assert api.runtime.get_turn_or_none(turn_id) is None

    approved = api.approve_turn(turn_id)

    assert change_service.approvals == [(run_id, "patch-from-facts")]
    # 续跑已经登记进事实层：状态回到 QUEUED，补丁标识来自产出事实。
    record = store.get_run(run_id)
    assert record is not None
    assert record.status == CHAT_QUEUED
    assert record.patch_id == "patch-from-facts"
    assert record.approval_token == "token-from-approval"
    assert approved.status == CHAT_QUEUED
    assert transport.published[-1].task_kind == TASK_RESUME_AFTER_APPROVAL


def test_waiting_approval_turn_is_projected_for_a_reader_without_local_state() -> None:
    """等待审批这一轮在别的进程时，读出来必须还是「等待审批」而不是失败。"""

    store, sessions = _shared_facts()
    api = _service(store=store, sessions=sessions, transport=InMemoryAgentRunTransport())
    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    _seed_user_turn(sessions, session_id, "把 value 改成 2")
    now = int(time.time() * 1000)
    run_id = "chat-run-waiting"
    turn_id = "turn-waiting"
    store.save_run(
        AgentRunRecord(
            run_id=run_id,
            turn_id=turn_id,
            session_id=session_id,
            task_id="task-waiting",
            task_kind=TASK_AGENT_RUN,
            status=WAITING_APPROVAL,
            policy_version=CHAT_RUN_POLICY_VERSION,
            idempotency_key=turn_id,
            deadline_epoch_ms=now + 60_000,
            updated_at_epoch_ms=now,
        )
    )
    store.save_output(
        RunOutput(
            run_id=run_id,
            attempt=1,
            task_type="change",
            assistant_message="等你确认。",
            result={"preview": {"patch_id": "patch-1"}},
            updated_at_epoch_ms=now,
        )
    )

    projected = api.resolve_turn(turn_id)

    assert projected.status == CHAT_WAITING_APPROVAL
    assert projected.assistant_message == "等你确认。"


def test_unreadable_user_message_is_marked_instead_of_crashing() -> None:
    """会话事实里已经找不到这一轮的提问时，读出来的是一句标记，不是 500。

    这一刻真实存在：会话又聊了足够多轮之后，这一轮的用户消息会被压缩挤出保留
    窗口，而 Run 与产出事实还在。标记必须明确说「读不出来」，不能用一句空话
    冒充用户提问。
    """

    store, sessions = _shared_facts()
    api = _service(store=store, sessions=sessions, transport=InMemoryAgentRunTransport())
    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    now = int(time.time() * 1000)
    run_id = "chat-run-old"
    turn_id = "turn-old"
    store.save_run(
        AgentRunRecord(
            run_id=run_id,
            turn_id=turn_id,
            session_id=session_id,
            task_id="task-old",
            task_kind=TASK_AGENT_RUN,
            status="COMPLETED",
            policy_version=CHAT_RUN_POLICY_VERSION,
            idempotency_key=turn_id,
            deadline_epoch_ms=now + 60_000,
            updated_at_epoch_ms=now,
        )
    )
    store.save_output(
        RunOutput(
            run_id=run_id,
            attempt=1,
            task_type="explain",
            assistant_message="旧一轮的回答仍然可读。",
            updated_at_epoch_ms=now,
        )
    )

    projected = api.resolve_turn(turn_id)

    assert projected.assistant_message == "旧一轮的回答仍然可读。"
    assert "找不到" in projected.user_message


def test_accepted_user_turn_survives_execution_with_a_shared_cache() -> None:
    """受理写下的用户消息，不能被执行体读到的旧快照抹掉。

    同一会话在缓存里曾经有两套键：受理走「未扫描」指纹，执行走扫描指纹。执行体
    命中旧快照后，会把那份「还没有这一轮」的记忆写回事实层，刚受理的问题就从会话
    里消失（真 Redis 上实测过）。这条用例把「执行完之后这句话还在」钉住。
    """

    store, sessions = _shared_facts_with_cache()
    transport = InMemoryAgentRunTransport()
    api = _service(store=store, sessions=sessions, transport=transport)
    worker = _service(store=store, sessions=sessions)

    session_id = str(api.create_session(str(FIXTURE_ROOT))["session_id"])
    accepted = api.submit_turn(
        session_id=session_id,
        repository_root=str(FIXTURE_ROOT),
        message="checkout 如何校验输入？",
    )
    assert worker.run_agent_task(transport.published[-1]).claimed is True

    session = sessions.load_session(session_id)
    assert session is not None
    memory = sessions.load_session_memory(
        session_id=session_id, scope=session.scope, repo_id=session.repo_id
    )
    assert memory is not None
    # 用户那一轮不能被抹掉：它是这一轮唯一记录过「问了什么」的事实；
    # 助手那一轮跟在后面，属于同一次执行。
    assert [(turn.role, turn.content) for turn in memory.recent_turns] == [
        ("user", "checkout 如何校验输入？"),
        ("assistant", "checkout 在入口处完成输入校验。"),
    ]

    # 读接口也要能读到这句话：事实层没被抹掉，投影才立得住。
    assert api.resolve_turn(accepted.turn_id).user_message == "checkout 如何校验输入？"
