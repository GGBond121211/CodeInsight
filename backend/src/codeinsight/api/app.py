"""本地 CodeInsight 服务的 FastAPI 应用组装。"""

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeinsight.api.chat_routes import create_chat_router
from codeinsight.api.routes import create_change_router, create_router
from codeinsight.api.usage_routes import create_usage_router
from codeinsight.application.change_service import ChangeService
from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.application.conversation_service import ConversationService
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.model_gateway import GatewayChatModel as OpenAIChatModel
from codeinsight.infrastructure.model_gateway import ModelGateway, default_gateway_from_environment
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.reranker import OpenAITextReranker, Reranker


def _default_model_factory() -> OpenAIChatModel:
    return OpenAIChatModel.from_environment(context_assembler=ContextAssembler())


def create_app(
    model_factory: Callable[[], OpenAIChatModel] = _default_model_factory,
    embedding_factory: Callable[[], OpenAIEmbeddingModel] = OpenAIEmbeddingModel.from_environment,
    change_service: ChangeService | None = None,
    reranker_factory: Callable[[], Reranker] = OpenAITextReranker.from_environment,
    usage_gateway_factory: Callable[[], ModelGateway] = default_gateway_from_environment,
    conversation_service: ConversationService | None = None,
) -> FastAPI:
    """组装本地 HTTP 应用，导入时不读取模型配置。"""
    application = FastAPI(
        title="CodeInsight API",
        version="2.0.3",
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
    selected_conversation_service = conversation_service or ConversationService(
        model_factory,
        embedding_factory,
        reranker_factory=reranker_factory,
        change_service=selected_change_service,
    )
    application.state.codeinsight_conversation = selected_conversation_service

    application.include_router(create_router(model_factory, embedding_factory, reranker_factory))
    application.include_router(create_usage_router(usage_gateway_factory))
    application.include_router(create_change_router(selected_change_service))
    application.include_router(create_chat_router(selected_conversation_service))
    return application


app = create_app()
