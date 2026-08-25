"""Tests for the service health endpoint."""

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app


def test_health_returns_ok() -> None:
    settings = Settings(database_url="sqlite://", _env_file=None)  # type: ignore[call-arg]

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
