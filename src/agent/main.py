"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from agent.config import Settings, get_settings
from agent.database import Database
from agent.repositories import router as repositories_router


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    application_settings = settings or get_settings()

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
    )

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Report whether the service is available."""
        return {"status": "ok"}

    application.include_router(repositories_router)
    return application


app = create_app()
