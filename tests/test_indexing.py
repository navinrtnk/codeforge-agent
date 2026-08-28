"""Tests for repository indexing."""

import hashlib
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from agent.indexing import chunk_text, detect_language, hash_content
from agent.main import create_app


@pytest.mark.parametrize(
    ("path", "language"),
    [
        ("src/main.py", "python"),
        ("web/app.tsx", "typescript"),
        ("config.yml", "yaml"),
        ("Dockerfile", "dockerfile"),
        ("Makefile", "makefile"),
        ("LICENSE", "unknown"),
        ("UPPER.PY", "python"),
    ],
)
def test_detect_language(path: str, language: str) -> None:
    assert detect_language(path) == language


def test_hash_content_returns_sha256_hex_digest() -> None:
    assert hash_content("hello") == hashlib.sha256(b"hello").hexdigest()
    assert len(hash_content("hello")) == 64


def test_chunk_text_preserves_content_and_line_numbers() -> None:
    content = "one\ntwo\nthree\nfour"

    chunks = chunk_text(content, chunk_size_lines=2)

    assert [(chunk.index, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (0, 1, 2),
        (1, 3, 4),
    ]
    assert "".join(chunk.content for chunk in chunks) == content


def test_chunk_text_handles_empty_content() -> None:
    assert chunk_text("", chunk_size_lines=10) == []


def test_chunk_text_rejects_nonpositive_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        chunk_text("content", chunk_size_lines=0)


def create_test_client(database_path: Path, chunk_size_lines: int = 2) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{database_path}",
        workspace_root=database_path.parent,
        index_chunk_size_lines=chunk_size_lines,
        _env_file=None,
    )
    return TestClient(create_app(settings))


def register_repository(client: TestClient, repository_path: Path) -> str:
    response = client.post(
        "/repositories",
        json={"name": "Example", "path": str(repository_path)},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def test_index_repository_and_report_status(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    (repository_path / "app.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (repository_path / "README.md").write_text("documentation\n", encoding="utf-8")
    (repository_path / "binary.dat").write_bytes(b"data\x00data")

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_repository(client, repository_path)
        index_response = client.post(f"/repositories/{repository_id}/index")
        status_response = client.get(f"/repositories/{repository_id}/index/status")

    assert index_response.status_code == 200
    assert index_response.json() == {
        "discovered": 3,
        "indexed": 2,
        "updated": 0,
        "skipped": 0,
        "deleted": 0,
        "failed": 1,
    }
    assert status_response.status_code == 200
    status = status_response.json()
    assert status["file_count"] == 2
    assert status["chunk_count"] == 3
    assert status["last_indexed_at"] is not None


def test_reindex_skips_unchanged_updates_changed_and_deletes_missing(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    app_path = repository_path / "app.py"
    readme_path = repository_path / "README.md"
    app_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    readme_path.write_text("documentation\n", encoding="utf-8")

    with create_test_client(tmp_path / "test.db") as client:
        repository_id = register_repository(client, repository_path)
        assert client.post(f"/repositories/{repository_id}/index").status_code == 200

        unchanged_response = client.post(f"/repositories/{repository_id}/index")
        app_path.write_text("one\nchanged\nthree\n", encoding="utf-8")
        readme_path.unlink()
        changed_response = client.post(f"/repositories/{repository_id}/index")
        status_response = client.get(f"/repositories/{repository_id}/index/status")

    assert unchanged_response.json() == {
        "discovered": 2,
        "indexed": 0,
        "updated": 0,
        "skipped": 2,
        "deleted": 0,
        "failed": 0,
    }
    assert changed_response.json() == {
        "discovered": 1,
        "indexed": 0,
        "updated": 1,
        "skipped": 0,
        "deleted": 1,
        "failed": 0,
    }
    assert status_response.json()["file_count"] == 1
    assert status_response.json()["chunk_count"] == 2


def test_reindex_rechunks_when_chunk_size_changes(tmp_path: Path) -> None:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    (repository_path / "app.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    database_path = tmp_path / "test.db"

    with create_test_client(database_path, chunk_size_lines=2) as client:
        repository_id = register_repository(client, repository_path)
        first_response = client.post(f"/repositories/{repository_id}/index")

    with create_test_client(database_path, chunk_size_lines=1) as client:
        second_response = client.post(f"/repositories/{repository_id}/index")
        status_response = client.get(f"/repositories/{repository_id}/index/status")

    assert first_response.json()["indexed"] == 1
    assert second_response.json()["updated"] == 1
    assert second_response.json()["skipped"] == 0
    assert status_response.json()["chunk_count"] == 3


def test_index_endpoints_reject_unknown_repository(tmp_path: Path) -> None:
    repository_id = uuid.uuid4()

    with create_test_client(tmp_path / "test.db") as client:
        index_response = client.post(f"/repositories/{repository_id}/index")
        status_response = client.get(f"/repositories/{repository_id}/index/status")

    assert index_response.status_code == 404
    assert status_response.status_code == 404
