from __future__ import annotations

import pytest

from codeinsight.application.session_service import SessionService
from codeinsight.domain.change import MODE_READ_ONLY, TenantScope
from codeinsight.domain.memory import (
    SESSION_SUMMARY_VERSION,
    SemanticMemory,
    SessionCompactionError,
    WorkingMemory,
)
from codeinsight.infrastructure.memory_store import InMemoryMemoryStore
from codeinsight.infrastructure.redis_cache import CacheUnavailableError, InMemoryCache
from codeinsight.infrastructure.run_store import InMemorySessionStore

SCOPE = TenantScope(tenant_id="tenant-a", user_id="user-a")


def _service(cache: object | None = None) -> SessionService:
    return SessionService(
        InMemorySessionStore(),
        InMemoryMemoryStore(),
        cache,  # type: ignore[arg-type]
    )


def test_session_is_restored_with_turns_and_the_same_goal() -> None:
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    cache = InMemoryCache()
    service = SessionService(session_store, memory_store, cache)
    kwargs = {
        "session_id": "session-1",
        "scope": SCOPE,
        "repo_id": "repo-1",
        "repo_fingerprint": "repo-fingerprint-1",
        "index_version": "index-1",
    }

    context = service.get_or_create_session(**kwargs)
    context = service.append_turn(
        context,
        role="user",
        content="先定位入口",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    context = service.continue_or_create_goal(
        context,
        user_goal="理解请求入口",
        task_type="explain",
        mode=MODE_READ_ONLY,
        goal_id="goal-1",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    context = service.append_turn(
        context,
        role="assistant",
        content="入口在 routes.py",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    service.update_session_memory(
        context,
        summary="已经定位入口",
        confirmed_conclusions=("入口在 routes.py",),
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )

    restored = service.get_or_create_session(**kwargs)
    assert restored.cache_hit is True
    assert restored.session.recent_turns[0].content == "先定位入口"
    assert restored.session.summary == "已经定位入口"
    assert restored.active_goal is not None
    assert restored.active_goal.goal_id == "goal-1"

    # 第二次表达不同措辞，默认继续当前 Goal；显式 start_new 才创建新 Goal。
    continued = service.continue_or_create_goal(
        restored,
        user_goal="再确认一下调用链",
        task_type="explain",
        mode=MODE_READ_ONLY,
        goal_id="goal-2",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    assert continued.active_goal is not None
    assert continued.active_goal.goal_id == "goal-1"

    new_goal = service.continue_or_create_goal(
        continued,
        user_goal="开始另一个目标",
        task_type="locate",
        mode=MODE_READ_ONLY,
        start_new=True,
        goal_id="goal-2",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    assert new_goal.active_goal is not None
    assert new_goal.active_goal.goal_id == "goal-2"


def test_working_and_semantic_memory_roundtrip_and_forget() -> None:
    memory_store = InMemoryMemoryStore()
    service = SessionService(InMemorySessionStore(), memory_store)
    working = WorkingMemory(
        run_id="run-1",
        user_goal="修复测试",
        selected_evidence_ids=("E1",),
        rejected_paths=("先改配置",),
        steps_remaining=2,
        structured_problem="将重试次数提取为配置",
        pending_tool="read_file",
        patch_ref="patch-1",
        check_status="waiting",
    )
    service.save_working_memory(working, scope=SCOPE, repo_id="repo-1")
    assert service.load_working_memory(
        run_id="run-1", scope=SCOPE, repo_id="repo-1"
    ) == working

    semantic = SemanticMemory(
        repo_id="repo-1",
        index_version="index-1",
        module_summaries=(("routes.py", "HTTP 入口"),),
        symbol_names=("answer",),
    )
    service.save_semantic_memory(
        semantic,
        scope=SCOPE,
        source="explicit-confirmation",
        confidence=0.9,
        consent=True,
    )
    assert service.load_semantic_memory(
        repo_id="repo-1", index_version="index-1", scope=SCOPE
    ) == semantic
    assert service.load_semantic_memory(
        repo_id="repo-1", index_version="index-2", scope=SCOPE
    ) is None

    service.forget_working_memory(run_id="run-1", scope=SCOPE, repo_id="repo-1")
    assert service.load_working_memory(
        run_id="run-1", scope=SCOPE, repo_id="repo-1"
    ) is None


class BrokenCache:
    def get(self, key: str) -> str | None:
        raise CacheUnavailableError("Redis stopped")

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        raise CacheUnavailableError("Redis stopped")

    def delete(self, key: str) -> None:
        raise CacheUnavailableError("Redis stopped")


def test_redis_failure_falls_back_to_session_and_memory_stores() -> None:
    service = _service(BrokenCache())
    context = service.get_or_create_session(
        session_id="session-1",
        scope=SCOPE,
        repo_id="repo-1",
        repo_fingerprint="fingerprint-1",
        index_version="index-1",
    )
    assert context.cache_fallback is True
    assert context.session.session_id == "session-1"


def test_public_session_factory_reads_the_environment(monkeypatch) -> None:
    from codeinsight.application.conversation_service import default_session_service

    # 没有 MySQL 配置时必须退回进程内存，而不是在启动时抛错。
    monkeypatch.delenv("CODEINSIGHT_MYSQL_HOST", raising=False)
    monkeypatch.delenv("CODEINSIGHT_MYSQL_USER", raising=False)
    monkeypatch.delenv("CODEINSIGHT_MYSQL_DATABASE", raising=False)
    monkeypatch.delenv("CODEINSIGHT_REDIS_URL", raising=False)

    service = default_session_service()
    context = service.get_or_create_session(
        session_id="session-factory",
        scope=SCOPE,
        repo_id="repo-factory",
        repo_fingerprint="fingerprint-f",
        index_version="index-f",
    )

    assert context.memory.recent_turns == ()
    assert context.session.session_id == "session-factory"


def test_repository_version_change_does_not_reuse_the_cached_surface() -> None:
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    service = SessionService(session_store, memory_store, InMemoryCache(), max_recent_turns=2)
    base: dict[str, object] = {
        "session_id": "session-versioned",
        "scope": SCOPE,
        "repo_id": "repo-versioned",
        "repo_fingerprint": "fingerprint-old",
        "index_version": "index-old",
    }
    context = service.get_or_create_session(**base)
    service.append_turn(
        context,
        role="user",
        content="旧仓库版本的问题",
        repo_fingerprint="fingerprint-old",
        index_version="index-old",
    )

    same = service.get_or_create_session(**base)
    assert [turn.content for turn in same.memory.recent_turns] == ["旧仓库版本的问题"]

    # 仓库指纹或索引版本一变，缓存键就不同，不能复用旧的 Session surface。
    newer = service.get_or_create_session(
        **{**base, "repo_fingerprint": "fingerprint-new", "index_version": "index-new"}
    )
    assert newer.cache_hit is False


def test_long_session_auto_compacts_old_turns_and_keeps_sequence_continuity() -> None:
    service = SessionService(
        InMemorySessionStore(),
        InMemoryMemoryStore(),
        max_recent_turns=2,
    )
    kwargs = {
        "session_id": "session-long",
        "scope": SCOPE,
        "repo_id": "repo-1",
        "repo_fingerprint": "fingerprint-1",
        "index_version": "index-1",
    }
    context = service.get_or_create_session(**kwargs)
    for number in range(1, 5):
        context = service.append_turn(
            context,
            role="user" if number % 2 else "assistant",
            content=f"第 {number} 轮需要保留的上下文",
            repo_fingerprint=kwargs["repo_fingerprint"],
            index_version=kwargs["index_version"],
        )

    assert [turn.sequence for turn in context.memory.recent_turns] == [3, 4]
    assert context.memory.compacted_through_sequence == 2
    assert context.memory.compaction_count == 2
    assert context.memory.summary is not None
    assert "第 1 轮" in context.memory.summary

    context = service.append_turn(
        context,
        role="user",
        content="第 5 轮继续",
        repo_fingerprint=kwargs["repo_fingerprint"],
        index_version=kwargs["index_version"],
    )
    assert [turn.sequence for turn in context.memory.recent_turns] == [4, 5]

    restored = service.get_or_create_session(**kwargs)
    assert restored.session.compacted_through_sequence == 3
    assert [turn.sequence for turn in restored.session.recent_turns] == [4, 5]


def _long_session(
    service: SessionService, contents: list[str]
) -> tuple[dict[str, object], object]:
    kwargs: dict[str, object] = {
        "session_id": "session-boundary",
        "scope": SCOPE,
        "repo_id": "repo-boundary",
        "repo_fingerprint": "fingerprint-b",
        "index_version": "index-b",
    }
    context = service.get_or_create_session(**kwargs)
    for number, content in enumerate(contents, start=1):
        context = service.append_turn(
            context,
            role="user" if number % 2 else "assistant",
            content=content,
            repo_fingerprint=str(kwargs["repo_fingerprint"]),
            index_version=str(kwargs["index_version"]),
        )
    return kwargs, context


def test_compaction_writes_a_boundary_id_and_matching_summary_hash() -> None:
    service = SessionService(
        InMemorySessionStore(), InMemoryMemoryStore(), max_recent_turns=2
    )
    _, context = _long_session(
        service, [f"第 {number} 轮内容" for number in range(1, 5)]
    )

    memory = context.memory  # type: ignore[attr-defined]
    assert memory.compaction_boundary_id is not None
    assert memory.structured_summary is not None
    assert memory.structured_summary.boundary_id == memory.compaction_boundary_id
    assert memory.structured_summary.summary_hash == memory.summary_hash
    assert memory.structured_summary.summary_version == SESSION_SUMMARY_VERSION
    assert memory.summary == memory.structured_summary.render()


def test_structured_summary_extracts_files_evidence_and_failures() -> None:
    service = SessionService(
        InMemorySessionStore(), InMemoryMemoryStore(), max_recent_turns=2
    )
    # 只压缩第 1 轮，让结构化字段与它一一对应，不受后续边界干扰。
    _, context = _long_session(
        service,
        [
            "请看 backend/src/codeinsight/domain/memory.py 的 E12 证据"
            + "\n"
            + "FAILED tests/unit/test_x.py::test_y"
            + "\n"
            + "还要不要保留 BM25？",
            "继续",
            "结束",
        ],
    )

    summary = context.memory.structured_summary  # type: ignore[attr-defined]
    assert summary is not None
    assert summary.source_sequence_range == (1, 1)
    assert "backend/src/codeinsight/domain/memory.py" in summary.important_files
    assert "E12" in summary.evidence_ids
    assert any("FAILED" in line for line in summary.test_failures)
    assert any("BM25" in question for question in summary.open_questions)
    # 逐轮片段也随摘要保留，被丢弃的原文不是彻底消失。
    assert any("E12" in snippet for snippet in summary.turn_snippets)


def test_second_compaction_keeps_the_previous_summary_verbatim() -> None:
    service = SessionService(
        InMemorySessionStore(), InMemoryMemoryStore(), max_recent_turns=2
    )
    kwargs, context = _long_session(
        service, [f"第 {number} 轮内容" for number in range(1, 4)]
    )
    first = context.memory.structured_summary  # type: ignore[attr-defined]
    assert first is not None
    first_hash = first.summary_hash

    context = service.append_turn(
        context,
        role="user",
        content="第 4 轮内容",
        repo_fingerprint=str(kwargs["repo_fingerprint"]),
        index_version=str(kwargs["index_version"]),
    )
    second = context.memory.structured_summary  # type: ignore[attr-defined]
    assert second is not None
    assert second.boundary_id != first.boundary_id
    # 上一段摘要原样保留，没有在下一次压缩里被重新摘要一次。
    assert second.previous_text == first.render()
    assert first.summary_hash == first_hash
    for snippet in first.turn_snippets:
        assert snippet in second.render()


def test_rebuilt_service_restores_summary_boundary_and_recent_turns() -> None:
    session_store = InMemorySessionStore()
    memory_store = InMemoryMemoryStore()
    kwargs: dict[str, object] = {
        "session_id": "session-restore",
        "scope": SCOPE,
        "repo_id": "repo-restore",
        "repo_fingerprint": "fingerprint-r",
        "index_version": "index-r",
    }
    service = SessionService(session_store, memory_store, max_recent_turns=2)
    context = service.get_or_create_session(**kwargs)
    for number in range(1, 5):
        context = service.append_turn(
            context,
            role="user" if number % 2 else "assistant",
            content=f"第 {number} 轮内容",
            repo_fingerprint=str(kwargs["repo_fingerprint"]),
            index_version=str(kwargs["index_version"]),
        )

    rebuilt = SessionService(session_store, memory_store, max_recent_turns=2)
    restored = rebuilt.get_or_create_session(**kwargs)

    assert restored.memory.compaction_boundary_id == context.memory.compaction_boundary_id
    assert restored.memory.summary_hash == context.memory.summary_hash
    assert restored.memory.structured_summary == context.memory.structured_summary
    assert restored.memory.summary == context.memory.summary
    assert [turn.sequence for turn in restored.memory.recent_turns] == [3, 4]


def test_summary_build_failure_keeps_history_and_reports_the_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from codeinsight.application import session_service as module

    def explode(**_: object) -> None:
        raise SessionCompactionError("注入的摘要失败")

    monkeypatch.setattr(module, "_build_structured_summary", explode)
    service = SessionService(
        InMemorySessionStore(), InMemoryMemoryStore(), max_recent_turns=2
    )
    _, context = _long_session(
        service, [f"第 {number} 轮内容" for number in range(1, 5)]
    )

    # 失败时历史保持完整：既没有丢轮次，也没有产生半截摘要。
    assert [turn.sequence for turn in context.memory.recent_turns] == [1, 2, 3, 4]  # type: ignore[attr-defined]
    assert context.memory.summary is None  # type: ignore[attr-defined]
    assert context.memory.compaction_boundary_id is None  # type: ignore[attr-defined]


def test_compact_session_reports_the_failure_class(monkeypatch: pytest.MonkeyPatch) -> None:
    from codeinsight.application import session_service as module

    service = SessionService(InMemorySessionStore(), InMemoryMemoryStore())
    kwargs, context = _long_session(
        service, [f"第 {number} 轮内容" for number in range(1, 5)]
    )

    def explode(**_: object) -> None:
        raise SessionCompactionError("注入的摘要失败")

    monkeypatch.setattr(module, "_build_structured_summary", explode)
    compacted, result = service.compact_session(
        context,
        repo_fingerprint=str(kwargs["repo_fingerprint"]),
        index_version=str(kwargs["index_version"]),
        max_recent_turns=2,
    )

    assert result.failure_class == "SUMMARY_BUILD_FAILED"
    assert result.dropped_turn_sequences == ()
    assert result.summary_updated is False
    assert result.boundary_id is None
    assert [turn.sequence for turn in compacted.memory.recent_turns] == [1, 2, 3, 4]


def test_fresh_read_ignores_the_other_version_snapshot() -> None:
    """执行体读会话时必须看到事实，而不是另一个版本键上的旧快照。

    受理走「未扫描」指纹、执行走扫描指纹，两条键各有一份上下文。执行体如果直接
    命中扫描键上的旧快照，它写回的记忆里就没有刚受理的用户消息——真 Redis 上实测
    发生过这件事。fresh=True 就是「这次读完之后要改写记忆，先回源事实」的说法。
    """

    cache = InMemoryCache()
    service = _service(cache)
    base = {"session_id": "session-surface", "scope": SCOPE, "repo_id": "repo-1"}
    created = service.get_or_create_session(
        **base, repo_fingerprint="scanned-version", index_version="index-1"
    )
    service.append_turn(
        created,
        role="user",
        content="checkout 如何校验输入？",
        repo_fingerprint="unscanned-version",
        index_version="index-1",
    )

    on_the_other_key = service.get_or_create_session(
        **base, repo_fingerprint="scanned-version", index_version="index-1"
    )
    read_from_facts = service.get_or_create_session(
        **base,
        repo_fingerprint="scanned-version",
        index_version="index-1",
        fresh=True,
    )

    assert on_the_other_key.memory.recent_turns == ()
    assert [turn.content for turn in read_from_facts.memory.recent_turns] == [
        "checkout 如何校验输入？"
    ]
