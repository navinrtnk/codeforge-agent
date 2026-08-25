"""Tests for repository registration endpoints."""

from pathlib import Path

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app


def create_test_client(database_path: Path) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{database_path}",
        workspace_root=database_path.parent,
        _env_file=None,
    )
    return TestClient(create_app(settings))


def test_register_and_list_repository(tmp_path: Path) -> None:
    repository_path = tmp_path / "example"
    repository_path.mkdir()

    with create_test_client(tmp_path / "test.db") as client:
        create_response = client.post(
            "/repositories",
            json={"name": "Example", "path": str(repository_path)},
        )
        list_response = client.get("/repositories")

    assert create_response.status_code == 201
    created_repository = create_response.json()
    assert created_repository["name"] == "Example"
    assert created_repository["path"] == str(repository_path.resolve())
    assert list_response.status_code == 200
    assert list_response.json() == [created_repository]


def test_register_repository_rejects_duplicate_path(tmp_path: Path) -> None:
    repository_path = tmp_path / "example"
    repository_path.mkdir()
    payload = {"name": "Example", "path": str(repository_path)}

    with create_test_client(tmp_path / "test.db") as client:
        assert client.post("/repositories", json=payload).status_code == 201
        response = client.post("/repositories", json=payload)

    assert response.status_code == 409
    assert response.json() == {"detail": "Repository path is already registered"}


def test_register_repository_rejects_missing_path(tmp_path: Path) -> None:
    with create_test_client(tmp_path / "test.db") as client:
        response = client.post(
            "/repositories",
            json={"name": "Missing", "path": str(tmp_path / "missing")},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Repository path does not exist"}


def test_register_repository_rejects_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.touch()

    with create_test_client(tmp_path / "test.db") as client:
        response = client.post(
            "/repositories",
            json={"name": "File", "path": str(file_path)},
        )

    assert response.status_code == 422
    assert response.json() == {"detail": "Repository path is not a directory"}


def test_list_repositories_supports_pagination(tmp_path: Path) -> None:
    first_path = tmp_path / "first"
    second_path = tmp_path / "second"
    first_path.mkdir()
    second_path.mkdir()

    with create_test_client(tmp_path / "test.db") as client:
        client.post("/repositories", json={"name": "First", "path": str(first_path)})
        client.post("/repositories", json={"name": "Second", "path": str(second_path)})
        response = client.get("/repositories", params={"offset": 1, "limit": 1})

    assert response.status_code == 200
    assert [repository["name"] for repository in response.json()] == ["Second"]
