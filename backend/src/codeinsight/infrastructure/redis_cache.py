"""Redis 与内存缓存适配器。

Redis 只做热点加速和短期共享状态；事实数据仍由 Memory/Session Store 保存。
这样 Redis 停止时可以回源，而不是把缓存误当成唯一数据库。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from redis import Redis
from redis.exceptions import RedisError

from codeinsight.domain.memory import MemoryCacheKey
from codeinsight.infrastructure.otel import get_telemetry

ENV_REDIS_URL = "CODEINSIGHT_REDIS_URL"


class CacheUnavailableError(Exception):
    """缓存服务不可用，调用方应回退事实 Store。"""


@dataclass(frozen=True)
class CacheAsideResult:
    """一次 cache-aside 读取的结果。"""

    value: str
    hit: bool
    used_fallback: bool


class InMemoryCache:
    """带 TTL 的内存缓存，用于测试和无 Redis 的本地开发。"""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, float]] = {}

    def get(self, key: str) -> str | None:
        found = self._values.get(key)
        if found is None:
            return None
        value, expires_at = found
        if time.monotonic() >= expires_at:
            self._values.pop(key, None)
            return None
        return value

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须为正")
        self._values[key] = (value, time.monotonic() + ttl_seconds)

    def delete(self, key: str) -> None:
        self._values.pop(key, None)


class RedisCache:
    """redis-py 的最小适配器。"""

    def __init__(self, client: Redis[Any]) -> None:
        self._client = client

    @classmethod
    def from_environment(cls) -> RedisCache | None:
        url = os.environ.get(ENV_REDIS_URL, "").strip()
        if not url:
            return None
        return cls(Redis.from_url(url, decode_responses=True))

    def get(self, key: str) -> str | None:
        try:
            with get_telemetry().span("redis", "get"):
                value = self._client.get(key)
        except RedisError as error:
            raise CacheUnavailableError(f"Redis get 失败：{error}") from error
        if value is None:
            return None
        return str(value)

    def set(self, key: str, value: str, *, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds 必须为正")
        try:
            with get_telemetry().span("redis", "set"):
                self._client.set(name=key, value=value, ex=ttl_seconds)
        except RedisError as error:
            raise CacheUnavailableError(f"Redis set 失败：{error}") from error

    def delete(self, key: str) -> None:
        try:
            with get_telemetry().span("redis", "delete"):
                self._client.delete(key)
        except RedisError as error:
            raise CacheUnavailableError(f"Redis delete 失败：{error}") from error


def cache_aside(
    cache: Any | None,
    key: MemoryCacheKey,
    loader: Any,
    *,
    ttl_seconds: int = 300,
) -> CacheAsideResult:
    """先查缓存，未命中或 Redis 故障时回源并尝试写回。"""
    if cache is None:
        return CacheAsideResult(value=str(loader()), hit=False, used_fallback=True)

    try:
        cached = cache.get(key.value)
    except CacheUnavailableError:
        return CacheAsideResult(value=str(loader()), hit=False, used_fallback=True)
    if cached is not None:
        return CacheAsideResult(value=cached, hit=True, used_fallback=False)

    value = str(loader())
    try:
        cache.set(key.value, value, ttl_seconds=ttl_seconds)
    except CacheUnavailableError:
        return CacheAsideResult(value=value, hit=False, used_fallback=True)
    return CacheAsideResult(value=value, hit=False, used_fallback=False)
