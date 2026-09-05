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
        cached_tokens = 0
        if usage is not None:
            details = getattr(usage, "prompt_tokens_details", None)
            cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        return ProviderResponse(
            content=getattr(message, "content", None),
            tool_calls=tuple(tool_calls),
            model=str(getattr(response, "model", request.model)),
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            cached_input_tokens=cached_tokens,
            finish_reason=str(getattr(choice, "finish_reason", "unknown")),
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
