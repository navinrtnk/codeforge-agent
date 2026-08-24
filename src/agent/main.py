"""FastAPI application entry point."""

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title="CodeForge Agent",
        version="0.1.0",
    )

    @application.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        """Report whether the service is available."""
        return {"status": "ok"}

    return application


app = create_app()
