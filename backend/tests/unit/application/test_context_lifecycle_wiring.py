"""Q-010 第五阶段：Session 历史预算在运行时真正被执行。

这些测试针对 Q-010 第一版的失败模式：策略对象数字齐全、单元测试全绿，
却没有任何调用方，压缩只在「保留轮数」超限时发生。
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeinsight.application.change_service import ChangeService
from codeinsight.application.context_budget import (
    ContextLifecyclePolicy,
    estimate_tokens,
)
from codeinsight.application.conversation_service import (
    INDEX_VERSION,
    ConversationService,
    _render_session_context,
)
from codeinsight.application.session_service import SessionService
from codeinsight.domain.change import TenantScope
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.model_gateway import resolve_route_budget
from codeinsight.infrastructure.run_store import InMemorySessionStore

REPO_ID = "repo-q010"
FINGERPRINT = "fp-q010"
# 每轮塞进历史的正文大小；用于在不依赖真实模型的情况下把历史推过预算。
TURN_CHARS = 40_000


@pytest.fixture(autouse=True)
def _shrunken_working_window(monkeypatch) -> None:
    """测试用的小窗口：默认 1M 下要铺四百万字符才能压过预算，那不叫实验。

    窗口大小是部署属性，这些用例只验证「历史压过预算时压缩真的被执行」。
    """
    monkeypatch.setenv("CODEINSIGHT_CONTEXT_WINDOW_TOKENS", "128000")


@dataclass(frozen=True)
class _StubTurn:
    """只需要 run_id：_enforce_session_budget 只用它发事件。"""

    run_id: str = "run-q010"


def _build_service(*, max_recent_turns: int = 200) -> ConversationService:
    # max_recent_turns 刻意放大：让「按轮数压缩」不可能触发，任何一次压缩
    # 都只能由 token 预算解释。这正是 Q-010 第一版缺失的区分。
    session_service = SessionService(
        InMemorySessionStore(),
        InMemoryMemoryStore(),
        max_recent_turns=max_recent_turns,
    )
    return ConversationService(
        lambda: None,
        lambda: None,
        reranker_factory=lambda: None,
        change_service=ChangeService(),
        session_service=session_service,
    )


def _seed_history(service: ConversationService, turns: int):
    context = service.session_service.get_or_create_session(
        session_id="session-q010",
        scope=TenantScope(),
        repo_id=REPO_ID,
        repo_fingerprint=FINGERPRINT,
        index_version=INDEX_VERSION,
    )
    body = "x" * TURN_CHARS
    for index in range(turns):
        context = service.session_service.append_turn(
            context,
            role="user",
            content=f"第 {index} 轮问题 {body}",
            repo_fingerprint=FINGERPRINT,
            index_version=INDEX_VERSION,
        )
        context = service.session_service.append_turn(
            context,
            role="assistant",
            content=f"第 {index} 轮回答 {body}",
            repo_fingerprint=FINGERPRINT,
            index_version=INDEX_VERSION,
        )
    return context


def test_history_under_budget_is_left_alone() -> None:
    service = _build_service()
    context = _seed_history(service, turns=1)
    limit = resolve_route_budget("explain").session_history_budget
    assert estimate_tokens(_render_session_context(context)) < limit

    result = service._enforce_session_budget(
        _StubTurn(), context, scene="explain", repo_fingerprint=FINGERPRINT
    )

    assert result.memory.compaction_count == context.memory.compaction_count
    assert result.memory.summary is None


def test_history_over_budget_is_compacted_by_tokens_alone() -> None:
    """核心回归：轮数完全不超限，压缩仍然必须发生。"""
    service = _build_service()
    context = _seed_history(service, turns=6)
    budget = resolve_route_budget("explain")
    surface_before = estimate_tokens(_render_session_context(context))
    assert len(context.memory.recent_turns) <= 200
    assert surface_before >= budget.session_history_budget

    result = service._enforce_session_budget(
        _StubTurn(), context, scene="explain", repo_fingerprint=FINGERPRINT
    )

    surface_after = estimate_tokens(_render_session_context(result))
    assert result.memory.compaction_count > context.memory.compaction_count
    assert result.memory.summary is not None
    assert result.memory.compacted_through_sequence > 0
    assert surface_after < budget.session_history_budget


def test_compaction_event_records_before_and_after_tokens() -> None:
    service = _build_service()
    context = _seed_history(service, turns=6)
    turn = _StubTurn(run_id="run-q010-events")

    service._enforce_session_budget(
        turn, context, scene="explain", repo_fingerprint=FINGERPRINT
    )

    events = service.runtime.event_log.read_events(turn.run_id)
    compacted = [event for event in events if event.event_type == "context_compacted"]
    assert compacted, "压缩必须留下可观测事件"
    payload = compacted[-1].payload
    assert payload["trigger"] == "session_history_budget"
    assert payload["outcome"] == "compacted"
    assert int(payload["surface_tokens_before"]) > int(payload["surface_tokens_after"])
    assert int(payload["dropped_turns"]) > 0
    assert payload["boundary_id"]


def test_summary_stays_within_its_token_cap_after_repeated_compaction() -> None:
    """反复压缩时 previous_text 会累积；摘要必须始终受上限约束。"""
    policy = ContextLifecyclePolicy()
    service = _build_service()
    context = _seed_history(service, turns=6)
    for _ in range(4):
        context = service._enforce_session_budget(
            _StubTurn(), context, scene="explain", repo_fingerprint=FINGERPRINT
        )
        context = service.session_service.append_turn(
            context,
            role="user",
            content="追加 " + "y" * TURN_CHARS,
            repo_fingerprint=FINGERPRINT,
            index_version=INDEX_VERSION,
        )
    summary = context.memory.summary
    assert summary is not None
    assert estimate_tokens(summary) <= policy.summary_max_tokens


def test_output_cap_wide_enough_to_break_the_invariant_is_refused() -> None:
    """把输出预留放大到吃掉历史预算时，构造路由预算必须直接报错。"""
    policy = ContextLifecyclePolicy(context_window_tokens=20_000)
    with pytest.raises(ValueError):
        policy.verify_trigger_reachable(15_000)


def test_adversarial_content_does_not_blow_up_compaction() -> None:
    """回归：路径正则曾经无界，长词字符段会退化成二次回溯。

    实测 40KB 的连续词字符会让一次压缩从 0.2 秒涨到 32 秒。这里给出宽松的
    上界，只用来抓住「又变回二次复杂度」这种量级的倒退。
    """
    import time

    service = _build_service()
    context = _seed_history(service, turns=6)
    started = time.perf_counter()
    result = service._enforce_session_budget(
        _StubTurn(), context, scene="explain", repo_fingerprint=FINGERPRINT
    )
    elapsed = time.perf_counter() - started
    assert result.memory.summary is not None
    assert elapsed < 10.0, f"压缩耗时 {elapsed:.1f}s，疑似正则回溯退化"


def test_summary_still_extracts_real_paths_from_dropped_turns() -> None:
    """加扫描上界之后，正常正文里的文件路径仍必须被提取出来。"""
    service = _build_service()
    context = service.session_service.get_or_create_session(
        session_id="session-q010-paths",
        scope=TenantScope(),
        repo_id=REPO_ID,
        repo_fingerprint=FINGERPRINT,
        index_version=INDEX_VERSION,
    )
    context = service.session_service.append_turn(
        context,
        role="user",
        content="解释 src/shop/pricing.py 与 src/shop/refunds.py 的关系 " + "z" * TURN_CHARS,
        repo_fingerprint=FINGERPRINT,
        index_version=INDEX_VERSION,
    )
    context = service.session_service.append_turn(
        context,
        role="assistant",
        content="pricing.py 负责定价 " + "z" * TURN_CHARS,
        repo_fingerprint=FINGERPRINT,
        index_version=INDEX_VERSION,
    )
    # 垫到超预算；每一对轮次约两万 token，最多四轮就够。
    limit = resolve_route_budget("explain").session_history_budget
    padding = 0
    while estimate_tokens(_render_session_context(context)) <= limit:
        padding += 1
        assert padding < 10, "预算没被推过阈值，用例前提失效"
        for role in ("user", "assistant"):
            context = service.session_service.append_turn(
                context,
                role=role,
                content=f"补充 {padding} {role} " + "z" * TURN_CHARS,
                repo_fingerprint=FINGERPRINT,
                index_version=INDEX_VERSION,
            )

    result = service._enforce_session_budget(
        _StubTurn(), context, scene="explain", repo_fingerprint=FINGERPRINT
    )

    assert result.memory.summary is not None
    # 路径来自第一批被丢掉的轮次，扫描上界不能把正常路径一起截掉。
    assert "src/shop/pricing.py" in result.memory.summary
    assert "src/shop/refunds.py" in result.memory.summary
