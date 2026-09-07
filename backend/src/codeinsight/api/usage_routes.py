"""让主 API 暴露实际承载 Auto Answer 调用的 Gateway 用量视图。"""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Query

from codeinsight.infrastructure.model_gateway import (
    ModelGateway,
    default_gateway_from_environment,
    usage_record_as_dict,
)

GatewayFactory = Callable[[], ModelGateway]


def create_usage_router(
    gateway_factory: GatewayFactory = default_gateway_from_environment,
) -> APIRouter:
    """创建低敏用量查询接口，并复用主 API 的进程内 Gateway 单例。"""

    router = APIRouter(prefix="/api/v1")

    @router.get("/usage/summary")
    def usage_summary() -> dict[str, object]:
        return gateway_factory().usage_summary()

    @router.get("/usage/calls")
    def usage_calls(limit: int = Query(default=100, ge=1, le=1_000)) -> dict[str, object]:
        gateway = gateway_factory()
        return {
            "object": "list",
            "data": [
                usage_record_as_dict(record)
                for record in gateway.usage_records(limit=limit)
            ],
        }

    return router
