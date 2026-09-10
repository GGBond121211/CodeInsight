"""Project chat-gateway endpoint policy."""

from __future__ import annotations

import os
from collections.abc import Mapping

from codeinsight.domain.errors import ModelConfigurationError

EXPECTED_CHAT_BASE_URL = "https://api.frontier-intelligence.tech/v1"


def configured_chat_base_url(environ: Mapping[str, str] | None = None) -> str:
    """Return the only approved endpoint for real chat-model requests.

    A missing value defaults to the project's configured gateway. An explicit
    different endpoint is rejected so a stale shell variable cannot silently
    send requests directly to another provider.
    """

    source = environ if environ is not None else os.environ
    configured = (source.get("CODEINSIGHT_BASE_URL") or "").strip().rstrip("/")
    if not configured:
        return EXPECTED_CHAT_BASE_URL
    if configured != EXPECTED_CHAT_BASE_URL:
        raise ModelConfigurationError(
            "CODEINSIGHT_BASE_URL 必须使用项目大模型网关 "
            f"{EXPECTED_CHAT_BASE_URL}，不能直接连接其他模型端点"
        )
    return EXPECTED_CHAT_BASE_URL
