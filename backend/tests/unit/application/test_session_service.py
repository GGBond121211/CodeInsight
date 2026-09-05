from __future__ import annotations

from codeinsight.application.session_service import SessionService
from codeinsight.domain.change import MODE_READ_ONLY, TenantScope
from codeinsight.domain.memory import SemanticMemory, WorkingMemory
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
