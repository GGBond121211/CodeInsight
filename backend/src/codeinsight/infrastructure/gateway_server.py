"""无会话状态的 OpenAI-compatible Gateway 服务。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from codeinsight.infrastructure.gateway_errors import GatewayError
from codeinsight.infrastructure.model_gateway import (
    GatewayRequest,
    ModelGateway,
    default_gateway_from_environment,
)


class GatewayChatRequest(BaseModel):
    messages: list[dict[str, object]]
    scene: str = "explain"
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = "default"
    user_id: str = "local"
    prompt_version: str = "gateway-api-v1"
    estimated_input_tokens: int = 0
    reserved_output_tokens: int = 1000
    tools: list[dict[str, object]] = Field(default_factory=list)
    response_format: dict[str, object] | None = None


def create_gateway_app(gateway: ModelGateway | None = None) -> FastAPI:
    selected = gateway
    app = FastAPI(title="CodeInsight Model Gateway", version="2.0-step7")

    def resolve_gateway() -> ModelGateway:
        nonlocal selected
        if selected is None:
            selected = default_gateway_from_environment()
        return selected

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "model-gateway"}

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(
            resolve_gateway().telemetry.metrics(), media_type="text/plain; version=0.0.4"
        )

    @app.get("/v1/models")
    def models() -> dict[str, object]:
        profiles = resolve_gateway().registry.all()
        return {
            "object": "list",
            "data": [
                {
                    "id": profile.model_id,
                    "object": "model",
                    "provider": profile.provider,
                    "quality_tier": profile.quality_tier,
                    "capabilities": sorted(profile.capabilities),
                    "price_version": profile.price_version,
                    "input_stars_per_million": str(profile.input_price_per_million),
                    "output_stars_per_million": str(profile.output_price_per_million),
                    "cached_input_stars_per_million": str(
                        profile.cached_input_price_per_million
                    ),
                }
                for profile in profiles
            ],
        }

    @app.post("/v1/chat/completions")
    def chat(request: GatewayChatRequest) -> dict[str, object]:
        capabilities = {"text"}
        if request.tools:
            capabilities.update({"tools", "structured_output"})
        if request.response_format is not None:
            capabilities.add("structured_output")
        try:
            result = resolve_gateway().complete(
                GatewayRequest(
                    request.request_id,
                    request.tenant_id,
                    request.user_id,
                    request.scene,
                    request.prompt_version,
                    tuple(_mapping(item) for item in request.messages),
                    request.estimated_input_tokens,
                    request.reserved_output_tokens,
                    frozenset(capabilities),
                    tuple(_mapping(item) for item in request.tools),
                    request.response_format,
                )
            )
        except GatewayError as error:
            status_code = 429 if error.code in {"RATE_LIMIT", "BUDGET_EXCEEDED"} else 502
            raise HTTPException(
                status_code, detail={"code": error.code, "message": str(error)}
            ) from error
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in result.tool_calls
        ]
        return {
            "id": f"chatcmpl-{request.request_id}",
            "object": "chat.completion",
            "model": result.model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.content,
                        "tool_calls": tool_calls,
                    },
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": result.input_tokens,
                "completion_tokens": result.output_tokens,
                "total_tokens": result.input_tokens + result.output_tokens,
            },
            "gateway": {
                "attempts": len(result.attempts),
                "cache_hit": result.cache_hit,
                "price_version": result.cost.price_version,
                "cost_stars": str(result.cost.total_stars),
            },
        }

    return app


def _mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return value


app = create_gateway_app()
