"""Git snapshot integration tests for bounded prompts and fresh conversation context."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent.config import Settings
from agent.git_context import GitContextBuilder
from agent.git_service import GitService
from agent.main import create_app
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage
from agent.repository_access import RepositoryAccess
from agent.tools import serialize_json


def git(path: Path, *arguments: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        ["git", "-C", str(path), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    path = tmp_path / "repository"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Test Engineer")
    git(path, "config", "user.email", "test@example.com")
    for name in ("staged.py", "unstaged.py", "deleted.py", "original.py"):
        (path / name).write_text("value = 1\n", encoding="utf-8")
    git(path, "add", ".")
    git(path, "commit", "-m", "initial commit")
    return path


def test_dirty_snapshot_contains_status_both_patches_and_redacted_secrets(
    tmp_path: Path,
    repository: Path,
) -> None:
    (repository / "staged.py").write_text(
        "api_key=sk-abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8"
    )
    git(repository, "add", "staged.py")
    (repository / "unstaged.py").write_text("value = 2\n", encoding="utf-8")
    (repository / "deleted.py").unlink()
    git(repository, "mv", "original.py", "renamed.py")
    (repository / "untracked.txt").write_text("UNTRACKED_PRIVATE_CONTENT", encoding="utf-8")
    builder = GitContextBuilder(GitService(RepositoryAccess(tmp_path)))
    snapshot = asyncio.run(builder.build(repository))

    assert snapshot is not None
    assert snapshot["branch"] == "main" and snapshot["clean"] is False
    assert snapshot["total_changed_files"] == 5
    assert {file["path"] for file in snapshot["files"]} == {
        "staged.py",
        "unstaged.py",
        "deleted.py",
        "renamed.py",
        "untracked.txt",
    }
    assert "[REDACTED]" in snapshot["staged_patch"]["patch"]
    assert "+value = 2" in snapshot["unstaged_patch"]["patch"]
    assert "deleted.py" in snapshot["unstaged_patch"]["patch"]
    assert "abcdefghijklmnopqrstuvwxyz" not in serialize_json(snapshot)
    assert "UNTRACKED_PRIVATE_CONTENT" not in serialize_json(snapshot)
    assert len(serialize_json(snapshot).encode("utf-8")) <= 16_000


def test_clean_and_unborn_snapshots(tmp_path: Path, repository: Path) -> None:
    builder = GitContextBuilder(GitService(RepositoryAccess(tmp_path)))
    clean = asyncio.run(builder.build(repository))
    assert clean is not None and clean["clean"] is True
    assert clean["files"] == []
    assert clean["staged_patch"] == {"patch": "", "truncated": False, "available": True}
    unborn = tmp_path / "unborn"
    unborn.mkdir()
    git(unborn, "init", "-b", "main")
    (unborn / "first.txt").write_text("first", encoding="utf-8")
    snapshot = asyncio.run(builder.build(unborn))
    assert snapshot is not None and snapshot["head"] is None
    assert snapshot["files"] == [{"status": "??", "path": "first.txt"}]


def test_snapshot_limits_file_count_and_json_bytes(tmp_path: Path, repository: Path) -> None:
    for index in range(8):
        (repository / f"new-{index}.txt").write_text("new", encoding="utf-8")
    for name in ("staged.py", "unstaged.py"):
        (repository / name).write_text(('value = "\\特殊文字"\n') * 500, encoding="utf-8")
    git(repository, "add", "staged.py")
    builder = GitContextBuilder(GitService(RepositoryAccess(tmp_path)), max_bytes=1024, max_files=2)
    snapshot = asyncio.run(builder.build(repository))
    assert snapshot is not None
    assert len(serialize_json(snapshot).encode("utf-8")) <= 1024
    assert snapshot["files_truncated"] is True
    assert len(snapshot["files"]) <= 2
    assert snapshot["total_changed_files"] == 10
    assert snapshot["staged_patch"]["truncated"] is True
    assert snapshot["unstaged_patch"]["truncated"] is True


def test_plain_directory_and_missing_git_are_graceful(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    builder = GitContextBuilder(GitService(RepositoryAccess(tmp_path)))
    assert asyncio.run(builder.build(plain)) is None
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))
    assert asyncio.run(builder.build(repository)) is None


def test_large_git_output_keeps_status_and_reports_unavailable_patch(
    tmp_path: Path, repository: Path
) -> None:
    (repository / "unstaged.py").write_text("x" * 10_000, encoding="utf-8")
    builder = GitContextBuilder(GitService(RepositoryAccess(tmp_path), max_output_bytes=256))
    snapshot = asyncio.run(builder.build(repository))
    assert snapshot is not None and snapshot["clean"] is False
    assert snapshot["files"][0]["path"] == "unstaged.py"
    assert snapshot["unstaged_patch"]["available"] is False
    assert snapshot["unstaged_patch"]["reason"] == "Git patch inspection unavailable"


def test_snapshot_timeout_keeps_status_and_cancels_patch_collection(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (repository / "unstaged.py").write_text("changed", encoding="utf-8")
    service = GitService(RepositoryAccess(tmp_path))
    cancelled: list[bool] = []

    async def slow_diff(path: Path, *, staged: bool = False) -> dict[str, str]:
        try:
            await asyncio.sleep(10)
        finally:
            cancelled.append(True)
        return {"patch": "unreachable"}

    monkeypatch.setattr(service, "diff", slow_diff)
    snapshot = asyncio.run(GitContextBuilder(service, timeout_seconds=0.2).build(repository))
    assert snapshot is not None
    assert snapshot["files"][0]["path"] == "unstaged.py"
    assert snapshot["staged_patch"]["available"] is False
    assert snapshot["unstaged_patch"]["available"] is False
    assert cancelled == [True]


def answer() -> ModelResponse:
    return ModelResponse(
        "response", "fake", "test-model", (TextContent("Inspected"),), "end_turn", TokenUsage(10, 5)
    )


def test_run_includes_and_persists_redacted_snapshot(tmp_path: Path, repository: Path) -> None:
    (repository / "unstaged.py").write_text("password=hunter2\n", encoding="utf-8")
    model = FakeModelClient([answer()])
    settings = Settings(  # type: ignore[call-arg]
        database_url="sqlite://",
        workspace_root=tmp_path,
        git_context_max_bytes=1024,
        _env_file=None,
    )
    with TestClient(create_app(settings, model_client=model)) as client:
        repository_id = client.post(
            "/repositories", json={"name": "Example", "path": str(repository)}
        ).json()["id"]
        created = client.post(
            f"/repositories/{repository_id}/runs", json={"task": "Review changes"}
        )
        assert created.status_code == 201 and created.json()["status"] == "completed"
        stored = client.get(f"/repositories/{repository_id}/runs/{created.json()['id']}").json()
    audit = stored["messages"][1]["content"][1]
    assert audit["type"] == "repository_context" and audit["source"] == "git"
    assert audit["snapshot"]["files"][0]["path"] == "unstaged.py"
    assert "hunter2" not in json.dumps(stored)
    request = model.requests[0]
    assert request.messages[-1].role == "user"
    context_block = request.messages[-1].content[1]
    assert isinstance(context_block, TextContent)
    snapshot = json.loads(context_block.text.split("\n", 1)[1])
    assert snapshot == audit["snapshot"]
    assert len(serialize_json(snapshot).encode("utf-8")) <= 1024


def test_conversation_follow_up_refreshes_snapshot_without_replaying_old_one(
    tmp_path: Path,
    repository: Path,
) -> None:
    model = FakeModelClient([answer(), answer()])
    settings = Settings(database_url="sqlite://", workspace_root=tmp_path, _env_file=None)  # type: ignore[call-arg]
    (repository / "first.txt").write_text("first", encoding="utf-8")
    with TestClient(create_app(settings, model_client=model)) as client:
        repository_id = client.post(
            "/repositories", json={"name": "Example", "path": str(repository)}
        ).json()["id"]
        conversation_id = client.post(
            f"/repositories/{repository_id}/conversations", json={}
        ).json()["id"]
        url = f"/repositories/{repository_id}/conversations/{conversation_id}/messages"
        assert client.post(url, json={"content": "Inspect"}).status_code == 201
        (repository / "first.txt").unlink()
        (repository / "second.txt").write_text("second", encoding="utf-8")
        assert client.post(url, json={"content": "Inspect again"}).status_code == 201
        saved = client.get(url).json()
    old_user = model.requests[1].messages[1]
    assert old_user.content == (TextContent("Inspect"),)
    latest = str(model.requests[1].messages)
    assert "second.txt" in latest and "first.txt" not in latest
    assert saved[0]["content"][1]["snapshot"]["files"][0]["path"] == "first.txt"
