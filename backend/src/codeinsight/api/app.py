"""FastAPI application composition for the local CodeInsight service."""

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
    """Compose a local HTTP app without reading model configuration at import time."""
    application = FastAPI(
        title="CodeInsight API",
        version="1.1.0",
        description="Router-guided repository answers with verifiable citations.",
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
