"""本地 CodeInsight 服务的 FastAPI 应用组装。"""

import os
from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeinsight.api.chat_routes import create_chat_router
from codeinsight.api.routes import create_change_router, create_router
from codeinsight.api.usage_routes import create_usage_router
from codeinsight.application.agent_run_dispatcher import AgentRunTransport
from codeinsight.application.change_service import ChangeService
from codeinsight.application.code_understanding_route import MCPClientFactory
from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.application.conversation_service import ConversationService
from codeinsight.application.validation_coordinator import ValidationTransport
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.model_gateway import GatewayChatModel as OpenAIChatModel
from codeinsight.infrastructure.model_gateway import ModelGateway, default_gateway_from_environment
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.reranker import OpenAITextReranker, Reranker

# 投递通道的开关。配了 broker 才跨进程投递；没配就保持进程内回调。
ENV_CELERY_BROKER_URL = "CODEINSIGHT_CELERY_BROKER_URL"


def _default_model_factory() -> OpenAIChatModel:
    return OpenAIChatModel.from_environment(context_assembler=ContextAssembler())


def _celery_transports() -> tuple[AgentRunTransport, ValidationTransport] | None:
    """按环境装配跨进程投递通道；没配 broker 时返回 None。

    默认不切：本机开发与单进程测试没有 Worker 进程在跑，切过去只会让每一轮永远停在
    QUEUED——「已排队但永远不会有人执行」正是这一层要避免的状态。所以跨进程执行是
    显式选择（配 broker），不是隐式默认。

    两条通道一起装配：Agent Run 的续跑与固定校验都必须投得出去，只切一条会让同一个
    流程分叉到两个地方执行。
    """

    if not os.environ.get(ENV_CELERY_BROKER_URL, "").strip():
        return None
    # 延迟导入：没配 broker 时不必为了组装应用去建一个 Celery app。
    from codeinsight.agent.worker_tasks import celery_app
    from codeinsight.infrastructure.celery_agent_dispatcher import (
        CeleryAgentRunTransport,
        CeleryValidationTransport,
    )

    return CeleryAgentRunTransport(celery_app), CeleryValidationTransport(celery_app)


def create_app(
    model_factory: Callable[[], OpenAIChatModel] = _default_model_factory,
    embedding_factory: Callable[[], OpenAIEmbeddingModel] = OpenAIEmbeddingModel.from_environment,
    change_service: ChangeService | None = None,
    reranker_factory: Callable[[], Reranker] = OpenAITextReranker.from_environment,
    usage_gateway_factory: Callable[[], ModelGateway] = default_gateway_from_environment,
    conversation_service: ConversationService | None = None,
    mcp_client_factory: MCPClientFactory | None = None,
    agent_run_transport: AgentRunTransport | None = None,
    validation_transport: ValidationTransport | None = None,
) -> FastAPI:
    """组装本地 HTTP 应用，导入时不读取模型配置。"""
    application = FastAPI(
        title="CodeInsight API",
        version="2.1.0",
        description="由 Router 引导、带可核验引用的仓库回答。",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:5174",
            "http://localhost:5174",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @application.middleware("http")
    async def trace_http_request(request, call_next):
        with get_telemetry().span(
            "api", "http_request", attributes={"http_method": request.method}
        ):
            return await call_next(request)

    selected_change_service = change_service
    if selected_change_service is None and conversation_service is not None:
        selected_change_service = conversation_service.change_service
    selected_change_service = selected_change_service or ChangeService()
    if conversation_service is None and agent_run_transport is None:
        # 调用方显式注入 transport 时一律以它为准（Worker 进程就是这样把自己的
        # broker 通道传进来）；只有完全没指定时才按环境决定走不走跨进程。
        assembled_transports = _celery_transports()
        if assembled_transports is not None:
            agent_run_transport, validation_transport = assembled_transports
    selected_conversation_service = conversation_service or ConversationService(
        model_factory,
        embedding_factory,
        reranker_factory=reranker_factory,
        change_service=selected_change_service,
        mcp_client_factory=mcp_client_factory,
        agent_run_transport=agent_run_transport,
        validation_transport=validation_transport,
    )
    application.state.codeinsight_conversation = selected_conversation_service

    application.include_router(
        create_router(
            model_factory,
            embedding_factory,
            reranker_factory,
            mcp_client_factory=mcp_client_factory,
        )
    )
    application.include_router(create_usage_router(usage_gateway_factory))
    application.include_router(create_change_router(selected_change_service))
    application.include_router(create_chat_router(selected_conversation_service))
    return application


app = create_app()
