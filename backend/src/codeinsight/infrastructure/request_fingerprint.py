"""Gateway 请求的规范化指纹。

指纹只用于缓存隔离、审计关联和命中分析；日志及 API 不保存原始 prompt。规范化
把 tools、response format 与消息一起纳入，避免「同一句话但工具协议不同」错误复用
同一份模型结果。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol


class GatewayRequestLike(Protocol):
    messages: tuple[Any, ...]
    scene: str
    prompt_version: str
    tools: tuple[Any, ...]
    response_format: object | None
    reserved_output_tokens: int

FINGERPRINT_VERSION = "gateway-request-v2"


@dataclass(frozen=True)
class RequestFingerprints:
    request: str
    stable_prefix: str


def request_fingerprints(request: GatewayRequestLike, *, model_id: str) -> RequestFingerprints:
    """为完整请求和「最后一条消息之前」的稳定前缀生成 SHA-256。"""
    full = _envelope(request, model_id=model_id, messages=request.messages)
    prefix = _envelope(request, model_id=model_id, messages=request.messages[:-1])
    return RequestFingerprints(_digest(full), _digest(prefix))


def _envelope(
    request: GatewayRequestLike,
    *,
    model_id: str,
    messages: tuple[dict[str, Any], ...] | tuple[Any, ...],
) -> dict[str, object]:
    return {
        "version": FINGERPRINT_VERSION,
        "model": model_id,
        "scene": request.scene,
        "prompt_version": request.prompt_version,
        "messages": messages,
        "tools": request.tools,
        "response_format": request.response_format,
        "reserved_output_tokens": request.reserved_output_tokens,
    }


def _digest(value: object) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
