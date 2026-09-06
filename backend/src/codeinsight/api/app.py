"""本地 CodeInsight 服务的 FastAPI 应用组装。"""

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeinsight.api.routes import create_change_router, create_router
from codeinsight.application.change_service import ChangeService
from codeinsight.application.context_assembler import ContextAssembler
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.model_gateway import GatewayChatModel as OpenAIChatModel
from codeinsight.infrastructure.otel import get_telemetry
from codeinsight.infrastructure.reranker import OpenAITextReranker, Reranker


def _default_model_factory() -> OpenAIChatModel:
    return OpenAIChatModel.from_environment(context_assembler=ContextAssembler())


def create_app(
    model_factory: Callable[[], OpenAIChatModel] = _default_model_factory,
    embedding_factory: Callable[[], OpenAIEmbeddingModel] = OpenAIEmbeddingModel.from_environment,
    change_service: ChangeService | None = None,
    reranker_factory: Callable[[], Reranker] = OpenAITextReranker.from_environment,
) -> FastAPI:
    """组装本地 HTTP 应用，导入时不读取模型配置。"""
    application = FastAPI(
        title="CodeInsight API",
        version="2.0.1",
        description="由 Router 引导、带可核验引用的仓库回答。",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
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

    application.include_router(create_router(model_factory, embedding_factory, reranker_factory))
    application.include_router(create_change_router(change_service))
    return application


app = create_app()
