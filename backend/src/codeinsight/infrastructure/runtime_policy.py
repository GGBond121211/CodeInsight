"""运行时开发策略。

开发模式只放开本地调试阶段明确会阻塞反馈的门：自动确认修改预览、
以及在 Docker 校验环境不可用时跳过固定校验。它不是客户端可传入的参数，
也不会改变路径、基线、隔离 workspace、产物指纹或补丁范围校验。

所有开关默认关闭；生产环境即使误配置了开发开关，也会被强制关闭。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_PRODUCTION_ENVIRONMENTS = frozenset({"prod", "production", "release"})


def _flag(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class DevelopmentPolicy:
    """当前进程是否可以使用本地调试便利开关。"""

    environment: str
    enabled: bool
    auto_approve_changes: bool
    skip_sandbox_validation: bool

    @classmethod
    def from_environment(
        cls, environ: Mapping[str, str] | None = None
    ) -> DevelopmentPolicy:
        source = os.environ if environ is None else environ
        environment = source.get("CODEINSIGHT_ENV", "development").strip().lower()
        requested = _flag(source.get("CODEINSIGHT_DEV_MODE"))
        enabled = requested and environment not in _PRODUCTION_ENVIRONMENTS
        return cls(
            environment=environment,
            enabled=enabled,
            auto_approve_changes=enabled and _flag(
                source.get("CODEINSIGHT_DEV_AUTO_APPROVE")
            ),
            skip_sandbox_validation=enabled and _flag(
                source.get("CODEINSIGHT_DEV_SKIP_SANDBOX_VALIDATION")
            ),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "enabled": self.enabled,
            "auto_approve_changes": self.auto_approve_changes,
            "skip_sandbox_validation": self.skip_sandbox_validation,
        }
