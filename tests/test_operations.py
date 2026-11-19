"""Operational contracts: safe errors, correlation, persisted usage, and upgrades."""

import asyncio
import io
import json
import logging
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.config import Settings
from agent.database import Database
from agent.main import create_app
from agent.models import ModelUsage
from agent.models_api import (
    FakeModelClient,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    TextContent,
    TokenUsage,
)
from agent.observability import JsonFormatter, logger, request_id, run_id
from agent.usage import ModelPrice, UsageModelClient


def configuration(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'state.db'}", workspace_root=tmp_path, _env_file=None
    )  # type: ignore[call-arg]


def response(text: str = "done") -> ModelResponse:
    return ModelResponse(
        "response", "fake", "test", (TextContent(text),), "end_turn", TokenUsage(100, 50)
    )


def test_errors_and_request_ids_are_consistent_and_do_not_leak_inputs(tmp_path: Path) -> None:
    app = create_app(configuration(tmp_path))

    @app.get("/broken")
    def broken() -> None:
        raise RuntimeError("password=do-not-log")

    @app.get("/limited")
    def limited() -> None:
        raise HTTPException(429, "Try later", headers={"Retry-After": "2"})

    correlation = str(uuid.uuid4())
    with TestClient(app) as client:
        for path, status in (("/missing", 404), ("/broken", 500), ("/limited", 429)):
            result = client.get(path, headers={"X-Request-ID": correlation})
            assert result.status_code == status
            assert result.headers["x-request-id"] == correlation
            assert result.json()["request_id"] == correlation
            assert result.json()["code"] == f"http_{status}"
            assert result.json()["errors"] == []
            assert "do-not-log" not in result.text
        assert client.get("/limited").headers["retry-after"] == "2"
        bad = client.post("/repositories", json={"name": {"password": "private-input"}})
        assert bad.status_code == 422 and bad.json()["errors"]
        assert (
            "private-input" not in bad.text and bad.json()["detail"] == "Request validation failed"
        )
        generated = client.get("/health", headers={"X-Request-ID": "invalid-id"})
        assert str(uuid.UUID(generated.headers["x-request-id"])) != "invalid-id"
        schema = client.get("/openapi.json").json()
        assert "ApiError" in schema["components"]["schemas"]
    assert request_id.get() is None and run_id.get() is None


def test_run_logging_and_usage_share_ids_and_survive_restart(tmp_path: Path) -> None:
    settings = configuration(tmp_path)
    settings.model_prices = {
        "fake:test": ModelPrice(input_per_million=Decimal("2"), output_per_million=Decimal("4"))
    }
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    correlation = str(uuid.uuid4())
    try:
        with TestClient(create_app(settings, model_client=FakeModelClient([response()]))) as client:
            registered = client.post("/repositories", json={"name": "test", "path": str(tmp_path)})
            result = client.post(
                f"/repositories/{registered.json()['id']}/runs",
                json={"task": "password=never-log-this"},
                headers={"X-Request-ID": correlation},
            )
            assert result.json()["status"] == "completed"
            run = result.json()["id"]
        events = [json.loads(line) for line in stream.getvalue().splitlines()]
        event = next(item for item in events if item["event"] == "model.completed")
        assert event["request_id"] == correlation and event["run_id"] == run
        assert "never-log-this" not in stream.getvalue()
        with TestClient(create_app(settings)) as restarted:
            group = restarted.get("/usage").json()["models"][0]
            assert group["calls"] == 1 and group["input_tokens"] == 100
            assert group["output_tokens"] == 50 and group["unpriced_calls"] == 0
            assert Decimal(group["estimated_cost_usd"]) == Decimal("0.0004")
        database = Database(settings.database_url)
        with database.session_factory() as session:
            usage = session.scalar(select(ModelUsage))
            assert usage and str(usage.agent_run_id) == run and usage.request_id == correlation
        database.dispose()
    finally:
        logger.removeHandler(handler)


def test_unpriced_and_invalid_responses_are_accounted_once(tmp_path: Path) -> None:
    # A returned response still incurs usage when the agent rejects its empty final answer.
    settings = configuration(tmp_path)
    with TestClient(create_app(settings, model_client=FakeModelClient([response("")]))) as client:
        repo = client.post("/repositories", json={"name": "test", "path": str(tmp_path)}).json()
        result = client.post(f"/repositories/{repo['id']}/runs", json={"task": "test"})
        assert result.json()["status"] == "failed"
        summary = client.get("/usage").json()["models"][0]
        assert summary["calls"] == summary["unpriced_calls"] == 1
        assert summary["estimated_cost_usd"] is None


def test_no_usage_is_invented_when_provider_fails(tmp_path: Path) -> None:
    database = Database(configuration(tmp_path).database_url)
    database.create_schema()
    database.create_schema()
    with database.session_factory() as session:
        tracker = UsageModelClient(FakeModelClient([]), session, {})
        with pytest.raises(ModelResponseError, match="no queued response"):
            asyncio.run(tracker.complete(ModelRequest((Message.text("user", "hello"),))))
        assert session.scalar(select(ModelUsage)) is None
    database.dispose()


@pytest.mark.parametrize("value", ["-1", "NaN", "Infinity"])
def test_invalid_prices_are_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        ModelPrice(input_per_million=Decimal(value), output_per_million=Decimal("1"))
