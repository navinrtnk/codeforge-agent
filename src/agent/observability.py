"""Small structured logger and request correlation without logging request content."""

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import UTC, datetime
from time import perf_counter

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
run_id: ContextVar[str | None] = ContextVar("run_id", default=None)
logger = logging.getLogger("codeforge")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "time": datetime.now(UTC).isoformat(),
                "level": record.levelname,
                "event": record.getMessage(),
                "request_id": request_id.get(),
                "run_id": run_id.get(),
                **{
                    key: getattr(record, key)
                    for key in ("status", "duration_ms")
                    if hasattr(record, key)
                },
            }
        )


def configure_logging() -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class ApiError(BaseModel):
    detail: str
    code: str
    request_id: str | None
    errors: list[dict[str, object]] = Field(default_factory=list)


def error_response(status: int, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ApiError(
            detail=message,
            code=f"http_{status}",
            request_id=request_id.get(),
        ).model_dump(),
    )


async def http_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, HTTPException)
    response = error_response(error.status_code, str(error.detail))
    if error.headers:
        response.headers.update(error.headers)
    return response


async def validation_error(request: Request, error: Exception) -> JSONResponse:
    assert isinstance(error, RequestValidationError)
    # Pydantic input/ctx fields may contain credentials or whole request bodies.
    errors = [{"location": list(item["loc"]), "type": item["type"]} for item in error.errors()]
    return JSONResponse(
        status_code=422,
        content=ApiError(
            detail="Request validation failed",
            code="http_422",
            request_id=request_id.get(),
            errors=errors,
        ).model_dump(),
    )


class RequestLoggingMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        try:
            correlation = str(uuid.UUID(headers.get(b"x-request-id", b"").decode("ascii")))
        except ValueError, UnicodeError:
            correlation = str(uuid.uuid4())
        token = request_id.set(correlation)
        started = perf_counter()
        status = 500
        sent = False

        async def respond(message: Message) -> None:
            nonlocal status, sent
            if message["type"] == "http.response.start":
                status = message["status"]
                sent = True
                message["headers"] = [
                    (key, value)
                    for key, value in message.get("headers", [])
                    if key.lower() != b"x-request-id"
                ] + [(b"x-request-id", correlation.encode())]
            await send(message)

        try:
            await self.app(scope, receive, respond)
        except Exception:
            logger.error("request.failed")
            if sent:
                raise
            await error_response(500, "Internal server error")(scope, receive, respond)
        finally:
            logger.info(
                "request.completed",
                extra={
                    "status": status,
                    "duration_ms": round((perf_counter() - started) * 1000, 3),
                },
            )
            request_id.reset(token)
