"""Tests for repository-scoped code and symbol search."""

import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from agent.config import Settings
from agent.main import create_app


def create_test_client(database_path: Path) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{database_path}",
        workspace_root=database_path.parent,
        index_chunk_size_lines=20,
        _env_file=None,
    )
    return TestClient(create_app(settings))


def register_and_index(client: TestClient, repository_path: Path, name: str = "Example") -> str:
    response = client.post(
        "/repositories",
        json={"name": name, "path": str(repository_path)},
    )
    assert response.status_code == 201
    repository_id = str(response.json()["id"])
    assert client.post(f"/repositories/{repository_id}/index").status_code == 200
    return repository_id


def test_lexical_search_returns_ranked_repository_scoped_snippets(tmp_path: Path) -> None:
    first_repository = tmp_path / "first"
    second_repository = tmp_path / "second"
    first_repository.mkdir()
    second_repository.mkdir()
    (first_repository / "primary.py").write_text(
        "# repository index repository index\nvalue = 1\n",
        encoding="utf-8",
    )
    (first_repository / "secondary.md").write_text(
        "A repository index supports search.\n",
        encoding="utf-8",
    )
    (second_repository / "private.md").write_text(
        "private repository index\n",
        encoding="utf-8",
    )

    with create_test_client(tmp_path / "test.db") as client:
        first_id = register_and_index(client, first_repository, "First")
        register_and_index(client, second_repository, "Second")
        response = client.get(
            f"/repositories/{first_id}/search",
            params={"q": "repository index", "limit": 10},
        )

    assert response.status_code == 200
    results = response.json()
    assert [result["path"] for result in results] == ["primary.py", "secondary.md"]
    assert all("<mark>" in result["snippet"] for result in results)
    assert results[0]["score"] >= results[1]["score"]
    assert all(result["path"] != "private.md" for result in results)


def test_exact_search_supports_case_sensitivity_and_context_lines(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "service.py").write_text(
        "one\ntwo\nclass RepositoryIndexer:\n    pass\nfive\nsix\n",
        encoding="utf-8",
    )

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_and_index(client, repository)
        insensitive_response = client.get(
            f"/repositories/{repository_id}/search",
            params={"q": "repositoryindexer", "mode": "exact"},
        )
        sensitive_response = client.get(
            f"/repositories/{repository_id}/search",
            params={
                "q": "repositoryindexer",
                "mode": "exact",
                "case_sensitive": True,
            },
        )

    assert insensitive_response.status_code == 200
    result = insensitive_response.json()[0]
    assert result["start_line"] == 1
    assert result["end_line"] == 5
    assert "RepositoryIndexer" in result["snippet"]
    assert sensitive_response.json() == []


def test_search_index_removes_updated_and_deleted_content(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source_file = repository / "service.py"
    source_file.write_text("old_unique_term = True\n", encoding="utf-8")

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_and_index(client, repository)
        assert client.get(
            f"/repositories/{repository_id}/search", params={"q": "old_unique_term"}
        ).json()

        source_file.write_text("new_unique_term = True\n", encoding="utf-8")
        client.post(f"/repositories/{repository_id}/index")
        old_response = client.get(
            f"/repositories/{repository_id}/search", params={"q": "old_unique_term"}
        )
        new_response = client.get(
            f"/repositories/{repository_id}/search", params={"q": "new_unique_term"}
        )

        source_file.unlink()
        client.post(f"/repositories/{repository_id}/index")
        deleted_response = client.get(
            f"/repositories/{repository_id}/search", params={"q": "new_unique_term"}
        )

    assert old_response.json() == []
    assert new_response.json()
    assert deleted_response.json() == []


def test_symbol_search_returns_qualified_names_and_signatures(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "service.py").write_text(
        "class RepositoryIndexer:\n    def index(self, repository: str) -> None:\n        pass\n",
        encoding="utf-8",
    )

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_and_index(client, repository)
        response = client.get(
            f"/repositories/{repository_id}/symbols",
            params={"q": "index"},
        )

    assert response.status_code == 200
    results = response.json()
    assert [result["qualified_name"] for result in results] == [
        "RepositoryIndexer.index",
        "RepositoryIndexer",
    ]
    assert results[0]["kind"] == "method"
    assert results[0]["signature"] == "def index(self, repository: str) -> None"
    assert results[0]["start_line"] == 2


def test_symbol_search_refreshes_updated_and_deleted_files(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source_file = repository / "service.py"
    source_file.write_text("def old_symbol() -> None:\n    pass\n", encoding="utf-8")

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_and_index(client, repository)
        source_file.write_text("def new_symbol() -> None:\n    pass\n", encoding="utf-8")
        client.post(f"/repositories/{repository_id}/index")
        old_response = client.get(
            f"/repositories/{repository_id}/symbols", params={"q": "old_symbol"}
        )
        new_response = client.get(
            f"/repositories/{repository_id}/symbols", params={"q": "new_symbol"}
        )

        source_file.unlink()
        client.post(f"/repositories/{repository_id}/index")
        deleted_response = client.get(
            f"/repositories/{repository_id}/symbols", params={"q": "new_symbol"}
        )

    assert old_response.json() == []
    assert new_response.json()
    assert deleted_response.json() == []


def test_search_endpoints_validate_queries_and_repository(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_and_index(client, repository)
        blank_response = client.get(f"/repositories/{repository_id}/search", params={"q": "   "})
        punctuation_response = client.get(
            f"/repositories/{repository_id}/search", params={"q": "---"}
        )
        missing_response = client.get(
            f"/repositories/{uuid.uuid4()}/symbols", params={"q": "anything"}
        )

    assert blank_response.status_code == 422
    assert punctuation_response.status_code == 422
    assert missing_response.status_code == 404
