"""公开会话的事实层恢复：进程重建、仓库版本隔离、Redis 故障回源。

需要配置好的 MySQL 测试库，未配置时整体跳过（见 conftest）。
"""

from __future__ import annotations

from codeinsight.application.session_service import SessionService
from codeinsight.domain.change import TenantScope
from codeinsight.infrastructure.db.stores import MySqlMemoryStore, MySqlSessionStore
from codeinsight.infrastructure.redis_cache import CacheUnavailableError, InMemoryCache

SCOPE = TenantScope(tenant_id="tenant-int", user_id="user-int")


def _service(session_factory, cache=None) -> SessionService:
    return SessionService(
        MySqlSessionStore(session_factory),
        MySqlMemoryStore(session_factory),
        cache,
        max_recent_turns=2,
    )


def _kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "session_id": "session-persist",
        "scope": SCOPE,
        "repo_id": "repo-persist",
        "repo_fingerprint": "fingerprint-a",
        "index_version": "index-a",
    }
    values.update(overrides)
    return values


def test_rebuilt_service_restores_summary_boundary_and_recent_tail(session_factory) -> None:
    kwargs = _kwargs()
    service = _service(session_factory)
    context = service.get_or_create_session(**kwargs)
    for number in range(1, 5):
        context = service.append_turn(
            context,
            role="user" if number % 2 else "assistant",
            content=f"第 {number} 轮需要跨进程保留",
            repo_fingerprint=str(kwargs["repo_fingerprint"]),
            index_version=str(kwargs["index_version"]),
        )
    assert context.memory.compaction_boundary_id is not None

    # 换一个全新的服务实例和全新的 Store 对象，模拟 API 进程重启。
    rebuilt = _service(session_factory).get_or_create_session(**kwargs)

    assert rebuilt.memory.compaction_boundary_id == context.memory.compaction_boundary_id
    assert rebuilt.memory.summary_hash == context.memory.summary_hash
    assert rebuilt.memory.summary == context.memory.summary
    assert [turn.sequence for turn in rebuilt.memory.recent_turns] == [3, 4]


def test_repository_version_change_does_not_reuse_the_old_surface(session_factory) -> None:
    service = _service(session_factory)
    first = service.get_or_create_session(**_kwargs())
    service.append_turn(
        first,
        role="user",
        content="仓库 A 的问题",
        repo_fingerprint="fingerprint-a",
        index_version="index-a",
    )

    other = service.get_or_create_session(
        **_kwargs(repo_fingerprint="fingerprint-b", index_version="index-b")
    )

    assert other.memory.recent_turns == ()
    assert other.memory.summary is None


def test_redis_failure_falls_back_to_the_fact_store(session_factory) -> None:
    kwargs = _kwargs()
    healthy = _service(session_factory, InMemoryCache())
    context = healthy.get_or_create_session(**kwargs)
    healthy.append_turn(
        context,
        role="user",
        content="Redis 挂掉也要能读回来",
        repo_fingerprint=str(kwargs["repo_fingerprint"]),
        index_version=str(kwargs["index_version"]),
    )

    broken = _service(session_factory, _BrokenCache())
    recovered = broken.get_or_create_session(**kwargs)

    assert [turn.content for turn in recovered.memory.recent_turns] == [
        "Redis 挂掉也要能读回来"
    ]
    assert recovered.cache_fallback is True


class _BrokenCache:
    """模拟 Redis 不可用；事实必须仍然来自 Store。"""

    def get(self, key: str) -> str | None:
        raise CacheUnavailableError("模拟 Redis 故障")

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        raise CacheUnavailableError("模拟 Redis 故障")

    def delete(self, key: str) -> None:
        raise CacheUnavailableError("模拟 Redis 故障")
