"""本地 CodeInsight 服务的 FastAPI 应用组装。"""

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from codeinsight.api.routes import create_router
from codeinsight.infrastructure.embeddings import OpenAIEmbeddingModel
from codeinsight.infrastructure.openai_chat import OpenAIChatModel


def create_app(
    model_factory: Callable[[], OpenAIChatModel] = OpenAIChatModel.from_environment,
    embedding_factory: Callable[[], OpenAIEmbeddingModel] = OpenAIEmbeddingModel.from_environment,
) -> FastAPI:
    """组装本地 HTTP 应用，导入时不读取模型配置。"""
    application = FastAPI(
        title="CodeInsight API",
        version="1.1.0",
        description="由 Router 引导、带可核验引用的仓库回答。",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    application.include_router(create_router(model_factory, embedding_factory))
    return application


app = create_app()
