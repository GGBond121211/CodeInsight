"""Gateway Provider 适配器与确定性 Fake Provider。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

from codeinsight.agent.tool_loop import ToolCall
from codeinsight.infrastructure.gateway_errors import (
    AuthenticationGatewayError,
    BadRequestGatewayError,
    InvalidResponseGatewayError,
    NetworkGatewayError,
    PermissionGatewayError,
    RateLimitGatewayError,
    TimeoutGatewayError,
    UpstreamGatewayError,
)


@dataclass(frozen=True)
class ProviderRequest:
    model: str
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] = ()
    response_format: Mapping[str, object] | None = None
    max_output_tokens: int | None = None


@dataclass(frozen=True)
class ProviderResponse:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    model: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    finish_reason: str
    cache_miss_tokens: int = 0
    usage_source: str = "unknown"

    def __post_init__(self) -> None:
        if min(
            self.input_tokens,
            self.output_tokens,
            self.cached_input_tokens,
            self.cache_miss_tokens,
        ) < 0:
            raise ValueError("Provider usage 不能为负")
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cache hit tokens 不能超过 input tokens")
        if not self.usage_source.strip():
            raise ValueError("usage_source 不能为空")

    @property
    def effective_cache_miss_tokens(self) -> int:
        """返回与 prompt_tokens 对齐的未命中输入量。"""
        return self.cache_miss_tokens or max(
            0, self.input_tokens - self.cached_input_tokens
        )

    @property
    def cache_hit_ratio(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens


class ProviderAdapter(Protocol):
    def invoke(self, request: ProviderRequest) -> ProviderResponse: ...


class OpenAIProviderAdapter:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        payload: dict[str, object] = {
            "model": request.model,
            "messages": list(request.messages),
            "temperature": 0,
        }
        if request.tools:
            payload["tools"] = list(request.tools)
        if request.response_format is not None:
            payload["response_format"] = dict(request.response_format)
        if request.max_output_tokens is not None:
            payload["max_tokens"] = request.max_output_tokens
        try:
            response = self.client.chat.completions.create(**payload)
        except RateLimitError as error:
            raise RateLimitGatewayError("上游限流") from error
        except APITimeoutError as error:
            raise TimeoutGatewayError("上游超时") from error
        except AuthenticationError as error:
            raise AuthenticationGatewayError("上游鉴权失败") from error
        except PermissionDeniedError as error:
            raise PermissionGatewayError("上游拒绝访问") from error
        except BadRequestError as error:
            raise BadRequestGatewayError("上游拒绝请求") from error
        except APIConnectionError as error:
            raise NetworkGatewayError("上游网络错误") from error
        except APIStatusError as error:
            if error.status_code >= 500:
                raise UpstreamGatewayError("上游服务错误") from error
            raise BadRequestGatewayError("上游请求失败") from error

        choice = response.choices[0]
        message = choice.message
        tool_calls: list[ToolCall] = []
        for item in getattr(message, "tool_calls", None) or ():
            function = getattr(item, "function", None)
            if function is None:
                raise InvalidResponseGatewayError("tool_call 缺少 function")
            try:
                arguments = json.loads(getattr(function, "arguments", "{}"))
            except json.JSONDecodeError as error:
                raise InvalidResponseGatewayError("tool_call arguments 不是 JSON") from error
            if not isinstance(arguments, dict):
                raise InvalidResponseGatewayError("tool_call arguments 必须是 object")
            tool_calls.append(
                ToolCall(str(item.id), str(function.name), arguments)
            )
        usage = response.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        cached_tokens, cache_miss_tokens, usage_source = _cache_usage(usage, input_tokens)
        return ProviderResponse(
            content=getattr(message, "content", None),
            tool_calls=tuple(tool_calls),
            model=str(getattr(response, "model", request.model)),
            input_tokens=input_tokens,
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_input_tokens=cached_tokens,
            finish_reason=str(getattr(choice, "finish_reason", "unknown")),
            cache_miss_tokens=cache_miss_tokens,
            usage_source=usage_source,
        )


class FakeProviderAdapter:
    """每个模型按脚本依次返回结果或抛异常，不产生质量结论。"""

    def __init__(self, scripts: Mapping[str, list[ProviderResponse | Exception]]) -> None:
        self._scripts = {model: list(items) for model, items in scripts.items()}
        self.called_models: list[str] = []

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        self.called_models.append(request.model)
        queue = self._scripts.get(request.model, [])
        if not queue:
            raise UpstreamGatewayError(f"Fake Provider 未配置 {request.model} 的下一响应")
        result = queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class StaticFakeProviderAdapter:
    """仅供多进程 service smoke；固定成功，不参与质量评测。"""

    def invoke(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(
            content='{"status":"fake-service-ok"}',
            tool_calls=(),
            model=request.model,
            input_tokens=8,
            output_tokens=4,
            cached_input_tokens=0,
            finish_reason="stop",
        )


def _cache_usage(usage: object | None, input_tokens: int) -> tuple[int, int, str]:
    """读取 DeepSeek 原生 usage，并兼容 OpenAI-compatible details。

    DeepSeek 返回 ``prompt_cache_hit_tokens`` 与 ``prompt_cache_miss_tokens``；
    OpenAI-compatible 上游常把命中量放在 ``prompt_tokens_details.cached_tokens``。
    外部字段可能通过 SDK 的宽松对象透传，因此这里使用 ``getattr``，并把
    hit+miss 归一到 prompt_tokens，避免成本核算因上游字段不完整而虚高或为负。
    """
    if usage is None:
        return 0, 0, "unavailable"

    native_hit = getattr(usage, "prompt_cache_hit_tokens", None)
    native_miss = getattr(usage, "prompt_cache_miss_tokens", None)
    details = getattr(usage, "prompt_tokens_details", None)
    compatible_hit = getattr(details, "cached_tokens", None)

    if native_hit is not None or native_miss is not None:
        hit = int(native_hit or 0)
        miss = int(native_miss or 0)
        source = "provider_native"
    elif compatible_hit is not None:
        hit = int(compatible_hit or 0)
        miss = max(0, input_tokens - hit)
        source = "openai_compatible"
    else:
        hit = 0
        miss = input_tokens
        source = "inferred_uncached"

    hit = min(max(hit, 0), input_tokens)
    if hit + miss != input_tokens:
        miss = max(0, input_tokens - hit)
        source = f"{source}_normalized"
    return hit, miss, source
