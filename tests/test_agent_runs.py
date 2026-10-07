"""Tests for agent run API endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage


def final_client(text: str = "Repository inspected") -> FakeModelClient:
    return FakeModelClient(
        responses=[
            ModelResponse(
                id="response-1",
                provider="fake",
                model="test-model",
                content=(TextContent(text),),
                stop_reason="end_turn",
                usage=TokenUsage(input_tokens=12, output_tokens=4),
            )
        ]
    )


def create_test_client(
    tmp_path: Path,
    model_client: FakeModelClient | None,
) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        workspace_root=tmp_path,
        _env_file=None,
    )
    return TestClient(create_app(settings, model_client=model_client))


def register_repository(client: TestClient, tmp_path: Path) -> str:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    response = client.post(
        "/repositories",
        json={"name": "Example", "path": str(repository_path)},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def test_create_list_and_get_agent_run(tmp_path: Path) -> None:
    with create_test_client(tmp_path, final_client()) as client:
        repository_id = register_repository(client, tmp_path)
        created = client.post(
            f"/repositories/{repository_id}/runs",
            json={"task": "Summarize the repository"},
        )
        listed = client.get(f"/repositories/{repository_id}/runs")
        fetched = client.get(f"/repositories/{repository_id}/runs/{created.json()['id']}")

    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "completed"
    assert body["error_message"] is None
    assert [message["role"] for message in body["messages"]] == [
        "system",
        "user",
        "assistant",
    ]
    assert body["messages"][-1]["content"] == [{"type": "text", "text": "Repository inspected"}]
    assert listed.status_code == 200
    assert [run["id"] for run in listed.json()] == [body["id"]]
    assert fetched.status_code == 200
    assert fetched.json() == body


def test_create_agent_run_requires_provider_configuration(tmp_path: Path) -> None:
    with create_test_client(tmp_path, None) as client:
        repository_id = register_repository(client, tmp_path)
        response = client.post(
            f"/repositories/{repository_id}/runs",
            json={"task": "Inspect the repository"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "CODEFORGE_MODEL_NAME is required"}


def test_agent_run_endpoints_reject_unknown_resources(tmp_path: Path) -> None:
    with create_test_client(tmp_path, final_client()) as client:
        repository_id = register_repository(client, tmp_path)
        missing_repository = client.post(
            "/repositories/00000000-0000-0000-0000-000000000000/runs",
            json={"task": "Inspect"},
        )
        missing_run = client.get(
            f"/repositories/{repository_id}/runs/00000000-0000-0000-0000-000000000000"
        )

    assert missing_repository.status_code == 404
    assert missing_run.status_code == 404
    assert missing_run.json() == {"detail": "Agent run does not exist"}
