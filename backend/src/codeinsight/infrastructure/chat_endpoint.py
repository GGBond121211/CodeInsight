"""Project chat-gateway endpoint policy."""

from __future__ import annotations

import os
from collections.abc import Mapping

from codeinsight.domain.errors import ModelConfigurationError

# 2026-09-11：旧中转站（api.frontier-intelligence.tech）额度耗尽，换成与 Embedding 同一家的
# 阿里云 MaaS 兼容端点。这里仍然只认一个端点：换供应商必须改这一行，改动会被 git 和
# tests/unit/infrastructure/test_openai_chat.py 记录，不允许通过环境变量静默漂移。
EXPECTED_CHAT_BASE_URL = "https://llm-690t13u7jmtcfr4l.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"


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
