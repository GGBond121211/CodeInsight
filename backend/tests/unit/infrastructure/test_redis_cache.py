from __future__ import annotations

from codeinsight.domain.memory import MemoryCacheKey
from codeinsight.infrastructure.redis_cache import (
    CacheUnavailableError,
    InMemoryCache,
    cache_aside,
)


def _key() -> MemoryCacheKey:
    return MemoryCacheKey.for_query(
        tenant_id="tenant-a",
        user_id="user-a",
        session_id="session-1",
        repo_fingerprint="fingerprint-a",
        index_version="index-1",
        query="如何创建订单？",
        top_k=5,
    )


def test_cache_key_contains_isolation_dimensions_but_not_raw_query() -> None:
    key = _key()
    assert "tenant-a" in key.value
    assert "user-a" in key.value
    assert "session-1" in key.value
    assert "fingerprint-a" in key.value
    assert "index-1" in key.value
    assert "如何创建订单" not in key.value
    assert key.value.startswith("codeinsight:v1:")


def test_cache_aside_cold_then_warm() -> None:
    cache = InMemoryCache()
    calls = 0

    def loader() -> str:
        nonlocal calls
        calls += 1
        return "候选证据"

    first = cache_aside(cache, _key(), loader, ttl_seconds=30)
    second = cache_aside(cache, _key(), loader, ttl_seconds=30)
    assert first == type(first)(value="候选证据", hit=False, used_fallback=False)
    assert second == type(second)(value="候选证据", hit=True, used_fallback=False)
    assert calls == 1


class BrokenCache:
    def get(self, key: str) -> str | None:
        raise CacheUnavailableError("Redis stopped")

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        raise CacheUnavailableError("Redis stopped")

    def delete(self, key: str) -> None:
        raise CacheUnavailableError("Redis stopped")


def test_cache_failure_returns_fact_store_value() -> None:
    calls = 0

    def loader() -> str:
        nonlocal calls
        calls += 1
        return "事实层结果"

    result = cache_aside(BrokenCache(), _key(), loader)
    assert result.value == "事实层结果"
    assert result.hit is False
    assert result.used_fallback is True
    assert calls == 1
