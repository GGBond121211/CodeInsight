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

FINGERPRINT_VERSION = "gateway-request-v3"


@dataclass(frozen=True)
class RequestFingerprints:
    request: str
    stable_prefix: str
    dynamic_suffix: str
    surface: str


def request_fingerprints(request: GatewayRequestLike, *, model_id: str) -> RequestFingerprints:
    """分别给稳定前缀、动态后缀、消息形状和完整请求生成 SHA-256。

    稳定前缀是开头的连续 system 消息加上工具目录、响应格式和路由标识——
    这些在压缩前后都不该变化。动态后缀是其余消息；压缩只替换这一部分。
    把两者分开，才能回答「压缩之后前缀是否还在被复用」，而不是只看到一个
    总分不清是前缀命中还是后缀偶然相同的数字。

    ``surface`` 只描述消息形状（角色序列、工具名、边界 ID），不含正文：
    它回答的是「这次请求的结构和上次是不是同一个」，而不是内容是否相同。
    """
    full = _envelope(request, model_id=model_id, messages=request.messages)
    system_end = _leading_system_end(request.messages)
    invariants = _invariants(request, model_id=model_id)
    prefix = {**invariants, "messages": request.messages[:system_end]}
    suffix = {
        "messages": request.messages[system_end:],
        "reserved_output_tokens": request.reserved_output_tokens,
    }
    return RequestFingerprints(
        _digest(full),
        _digest(prefix),
        _digest(suffix),
        _digest(_surface(request, model_id=model_id)),
    )


def _leading_system_end(messages: tuple[Any, ...]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role", "")) != "system":
            break
        count += 1
    return count


def _invariants(request: GatewayRequestLike, *, model_id: str) -> dict[str, object]:
    return {
        "version": FINGERPRINT_VERSION,
        "model": model_id,
        "scene": request.scene,
        "prompt_version": request.prompt_version,
        "tools": request.tools,
        "response_format": request.response_format,
    }


def _surface(request: GatewayRequestLike, *, model_id: str) -> dict[str, object]:
    roles: list[str] = []
    tool_names: list[str] = []
    for message in request.messages:
        if not isinstance(message, dict):
            roles.append("unknown")
            continue
        roles.append(str(message.get("role", "unknown")))
        calls = message.get("tool_calls")
        if isinstance(calls, (list, tuple)):
            for call in calls:
                if isinstance(call, dict):
                    function = call.get("function")
                    if isinstance(function, dict):
                        tool_names.append(str(function.get("name", "")))
    return {
        "version": FINGERPRINT_VERSION,
        "model": model_id,
        "scene": request.scene,
        "prompt_version": request.prompt_version,
        "roles": roles,
        "tool_names": tool_names,
        "boundary_id": getattr(request, "compaction_boundary_id", None),
    }


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
