"""无会话状态的 OpenAI-compatible Gateway 服务。"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping

from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, Field

from codeinsight.domain.errors import ModelConfigurationError
from codeinsight.infrastructure.gateway_errors import GatewayError
from codeinsight.infrastructure.model_gateway import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    GatewayRequest,
    ModelGateway,
    default_gateway_from_environment,
    usage_record_as_dict,
)


class GatewayChatRequest(BaseModel):
    messages: list[dict[str, object]]
    scene: str = "explain"
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tenant_id: str = "default"
    user_id: str = "local"
    prompt_version: str = "gateway-api-v1"
    estimated_input_tokens: int = 0
    reserved_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    tools: list[dict[str, object]] = Field(default_factory=list)
    response_format: dict[str, object] | None = None
    run_id: str | None = None


def create_gateway_app(gateway: ModelGateway | None = None) -> FastAPI:
    selected = gateway
    app = FastAPI(title="CodeInsight Model Gateway", version="2.0.3")

    def resolve_gateway() -> ModelGateway:
        nonlocal selected
        if selected is None:
            selected = default_gateway_from_environment()
        return selected

    @app.get("/health")
    def health() -> dict[str, object]:
        try:
            gateway = resolve_gateway()
        except ModelConfigurationError as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "MODEL_CONFIGURATION", "message": str(error)},
            ) from error
        return {
            "status": "ok",
            "service": "model-gateway",
            "routes": {
                scene: {
                    "primary": route.primary_model_id,
                    "fallbacks": list(route.fallback_model_ids),
                }
                for scene, route in gateway.routes.items()
            },
        }

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

    @app.get("/v1/usage/summary")
    def usage_summary() -> dict[str, object]:
        """低敏用量汇总，供本地监控面板和排障使用。"""
        return resolve_gateway().usage_summary()

    @app.get("/v1/usage/calls")
    def usage_calls(
        limit: int = Query(default=100, ge=1, le=1_000),
    ) -> dict[str, object]:
        """最近 attempt 的明细；不返回 prompt、工具参数或模型正文。"""
        records = resolve_gateway().usage_records(limit=limit)
        return {
            "object": "list",
            "data": [usage_record_as_dict(record) for record in records],
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
                    request_id=request.request_id,
                    tenant_id=request.tenant_id,
                    user_id=request.user_id,
                    scene=request.scene,
                    prompt_version=request.prompt_version,
                    messages=tuple(_mapping(item) for item in request.messages),
                    estimated_input_tokens=request.estimated_input_tokens,
                    reserved_output_tokens=request.reserved_output_tokens,
                    required_capabilities=frozenset(capabilities),
                    tools=tuple(_mapping(item) for item in request.tools),
                    response_format=request.response_format,
                    run_id=request.run_id,
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
                "prompt_cache_hit_tokens": result.cached_input_tokens,
                "prompt_cache_miss_tokens": result.cache_miss_tokens,
                "cache_hit_ratio": (
                    result.cached_input_tokens / result.input_tokens
                    if result.input_tokens
                    else 0.0
                ),
                "usage_source": result.cost.usage_source,
            },
            "gateway": {
                "attempts": len(result.attempts),
                "cache_hit": result.cache_hit,
                "semantic_cache_hit": result.semantic_cache_hit,
                "provider_cache_read_tokens": result.cached_input_tokens,
                "provider_cache_miss_tokens": result.cache_miss_tokens,
                "price_version": result.cost.price_version,
                "cost_stars": str(result.cost.total_stars),
                "request_fingerprint": result.request_fingerprint,
                "stable_prefix_fingerprint": result.stable_prefix_fingerprint,
            },
        }

    return app


def _mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return value


app = create_gateway_app()
