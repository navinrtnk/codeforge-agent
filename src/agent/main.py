"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from agent.agent_runs import router as agent_runs_router
from agent.config import Settings, get_settings
from agent.conversations import router as conversations_router
from agent.database import Database
from agent.embeddings import EmbeddingClient
from agent.models_api import ModelClient
from agent.observability import (
    ApiError,
    RequestLoggingMiddleware,
    configure_logging,
    http_error,
    validation_error,
)
from agent.repositories import router as repositories_router
from agent.reviews import router as reviews_router
from agent.usage import router as usage_router


def create_app(
    settings: Settings | None = None,
    *,
    model_client: ModelClient | None = None,
    embedding_client: EmbeddingClient | None = None,
) -> FastAPI:
    """Create and configure the FastAPI application."""
    application_settings = settings or get_settings()
    configure_logging()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        database = Database(application_settings.database_url)
        database.create_schema()
        application.state.database = database
        yield
        database.dispose()

    application = FastAPI(
        title="CodeForge Agent",
        version="0.1.0",
        lifespan=lifespan,
        responses={
            code: {"model": ApiError} for code in (400, 403, 404, 405, 409, 422, 500, 502, 503, 504)
        },
    )
    application.add_middleware(RequestLoggingMiddleware)
    application.add_exception_handler(HTTPException, http_error)
    application.add_exception_handler(RequestValidationError, validation_error)
    application.state.settings = application_settings
    application.state.model_client = model_client
    application.state.embedding_client = embedding_client
    application.state.index_locks = {}
    application.state.git_service = None

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Report whether the service is available."""
        return {"status": "ok"}

    application.include_router(repositories_router)
    application.include_router(agent_runs_router)
    application.include_router(conversations_router)
    application.include_router(reviews_router)
    application.include_router(usage_router)
    return application


app = create_app()
