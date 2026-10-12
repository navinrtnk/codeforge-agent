"""Integration tests for local Git operations and preservation of working files."""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select

from agent.database import Database
from agent.git_service import GitError, GitService
from agent.git_tools import create_agent_tool_registry
from agent.models import AgentRun, Repository, ToolEvent
from agent.models_api import ToolCall
from agent.repository_access import RepositoryAccess
from agent.tools import ToolContext, ToolExecutor


def git(repository: Path, *arguments: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
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
    (path / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(path, "add", "app.py")
    git(path, "commit", "-m", "initial commit")
    return path


def service(tmp_path: Path, *, mutations: bool = True) -> GitService:
    return GitService(RepositoryAccess(tmp_path), allow_mutations=mutations)


def test_status_diff_history_and_branches(tmp_path: Path, repository: Path) -> None:
    client = service(tmp_path)
    (repository / "app.py").write_text("value = 2\n", encoding="utf-8")
    (repository / "new file.txt").write_text("new\n", encoding="utf-8")

    async def inspect() -> None:
        state = await client.status(repository)
        assert state["branch"] == "main"
        assert state["head"] == git(repository, "rev-parse", "HEAD")
        assert state["detached"] is False
        assert state["clean"] is False
        assert state["files"] == [
            {"status": " M", "path": "app.py"},
            {"status": "??", "path": "new file.txt"},
        ]
        assert "+value = 2" in (await client.diff(repository))["patch"]
        assert (await client.diff(repository, staged=True))["patch"] == ""
        assert (await client.log(repository))["commits"][0]["subject"] == "initial commit"
        assert (await client.branches(repository))["branches"] == ["main"]

    asyncio.run(inspect())


def test_branch_stage_commit_and_unstage_preserve_unrelated_changes(
    tmp_path: Path,
    repository: Path,
) -> None:
    client = service(tmp_path)

    async def workflow() -> None:
        assert (await client.create_branch(repository, "feature/test"))["created"] == "feature/test"
        assert (await client.checkout(repository, "feature/test"))["branch"] == "feature/test"
        (repository / "app.py").write_text("value = 2\n", encoding="utf-8")
        (repository / "unrelated.txt").write_text("user work\n", encoding="utf-8")
        await client.stage(repository, ["app.py"])
        assert "+value = 2" in (await client.diff(repository, staged=True))["patch"]
        await client.unstage(repository, ["app.py"])
        assert (await client.diff(repository, staged=True))["patch"] == ""
        assert (repository / "app.py").read_text() == "value = 2\n"
        await client.stage(repository, ["app.py"])
        committed = await client.commit(repository, "update value")
        assert committed["commit"] == git(repository, "rev-parse", "HEAD")
        assert git(repository, "show", "HEAD:app.py") == "value = 2"
        assert (repository / "unrelated.txt").read_text() == "user work\n"
        assert (await client.status(repository))["files"] == [
            {"status": "??", "path": "unrelated.txt"}
        ]

    asyncio.run(workflow())


def test_mutations_disabled_at_service_boundary(tmp_path: Path, repository: Path) -> None:
    client = service(tmp_path, mutations=False)
    with pytest.raises(GitError, match="disabled"):
        asyncio.run(client.stage(repository, ["app.py"]))
    assert git(repository, "status", "--porcelain") == ""


def test_checkout_refuses_dirty_tree_and_duplicate_branch(tmp_path: Path, repository: Path) -> None:
    client = service(tmp_path)
    (repository / "app.py").write_text("uncommitted\n", encoding="utf-8")
    asyncio.run(client.create_branch(repository, "feature"))
    with pytest.raises(GitError, match="clean"):
        asyncio.run(client.checkout(repository, "feature"))
    with pytest.raises(GitError, match="failed"):
        asyncio.run(client.create_branch(repository, "feature"))
    assert (repository / "app.py").read_text() == "uncommitted\n"
    assert git(repository, "branch", "--show-current") == "main"


@pytest.mark.parametrize("name", ["--orphan", "bad name", "HEAD", "../outside", "@{-1}"])
def test_invalid_branch_names(tmp_path: Path, repository: Path, name: str) -> None:
    with pytest.raises(GitError, match="Invalid local branch"):
        asyncio.run(service(tmp_path).create_branch(repository, name))


@pytest.mark.parametrize(
    "paths", [[], ["../outside"], ["/tmp/file"], ["."], [".git/config"], ["dir\\file"]]
)
def test_invalid_stage_paths(tmp_path: Path, repository: Path, paths: list[str]) -> None:
    with pytest.raises(GitError):
        asyncio.run(service(tmp_path).stage(repository, paths))
    assert git(repository, "diff", "--cached", "--name-only") == ""


def test_literal_paths_and_deleted_files(tmp_path: Path, repository: Path) -> None:
    client = service(tmp_path)
    (repository / "literal[1].txt").write_text("literal\n", encoding="utf-8")
    (repository / "literal1.txt").write_text("other\n", encoding="utf-8")
    (repository / "app.py").unlink()
    asyncio.run(client.stage(repository, ["literal[1].txt", "app.py"]))
    assert git(repository, "diff", "--cached", "--name-only").splitlines() == [
        "app.py",
        "literal[1].txt",
    ]


def test_symlink_outside_repository_rejected(tmp_path: Path, repository: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("private\n", encoding="utf-8")
    (repository / "link.txt").symlink_to(outside)
    with pytest.raises(GitError, match="relative files"):
        asyncio.run(service(tmp_path).stage(repository, ["link.txt"]))


def test_non_git_and_nested_registration_rejected(tmp_path: Path, repository: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    nested = repository / "nested"
    nested.mkdir()
    with pytest.raises(GitError, match="failed"):
        asyncio.run(service(tmp_path).status(plain))
    with pytest.raises(GitError, match="working tree root"):
        asyncio.run(service(tmp_path).status(nested))


def test_detached_head_refuses_commit_and_can_switch(tmp_path: Path, repository: Path) -> None:
    git(repository, "checkout", "--detach", "HEAD")
    client = service(tmp_path)
    assert asyncio.run(client.status(repository))["detached"] is True
    with pytest.raises(GitError, match="local branch"):
        asyncio.run(client.commit(repository, "detached change"))
    assert asyncio.run(client.checkout(repository, "main"))["detached"] is False


def test_unborn_repository_status_history_and_unstage(tmp_path: Path) -> None:
    path = tmp_path / "unborn"
    path.mkdir()
    git(path, "init", "-b", "main")
    git(path, "config", "user.name", "Test Engineer")
    git(path, "config", "user.email", "test@example.com")
    client = service(tmp_path)
    assert asyncio.run(client.status(path))["head"] is None
    assert asyncio.run(client.log(path))["commits"] == []
    with pytest.raises(GitError, match="initial commit"):
        asyncio.run(client.create_branch(path, "feature"))
    (path / "first.txt").write_text("first\n", encoding="utf-8")
    asyncio.run(client.stage(path, ["first.txt"]))
    (path / "first.txt").write_text("newer user work\n", encoding="utf-8")
    asyncio.run(client.unstage(path, ["first.txt"]))
    assert (path / "first.txt").read_text() == "newer user work\n"
    asyncio.run(client.stage(path, ["first.txt"]))
    assert asyncio.run(client.commit(path, "initial commit"))["commit"] is not None


def test_conflicts_and_active_operations_refuse_mutations(tmp_path: Path, repository: Path) -> None:
    client = service(tmp_path)
    git(repository, "checkout", "-b", "other")
    (repository / "app.py").write_text("other\n", encoding="utf-8")
    git(repository, "commit", "-am", "other")
    git(repository, "checkout", "main")
    (repository / "app.py").write_text("main\n", encoding="utf-8")
    git(repository, "commit", "-am", "main")
    with pytest.raises(subprocess.CalledProcessError):
        git(repository, "merge", "other")
    assert any(item["status"] == "UU" for item in asyncio.run(client.status(repository))["files"])
    with pytest.raises(GitError, match="merge conflicts"):
        asyncio.run(client.stage(repository, ["app.py"]))
    git(repository, "add", "app.py")
    with pytest.raises(GitError, match="active merge"):
        asyncio.run(client.commit(repository, "finish merge"))


def test_hooks_external_diff_and_filters_do_not_execute(tmp_path: Path, repository: Path) -> None:
    marker = tmp_path / "executed"
    hook = repository / ".git" / "hooks" / "pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n", encoding="utf-8")
    hook.chmod(0o755)
    git(repository, "config", "diff.external", f"touch {marker}")
    client = service(tmp_path)
    (repository / "app.py").write_text("changed\n", encoding="utf-8")
    assert "+changed" in asyncio.run(client.diff(repository))["patch"]
    asyncio.run(client.stage(repository, ["app.py"]))
    asyncio.run(client.commit(repository, "update without hooks"))
    assert not marker.exists()
    git(repository, "config", "filter.danger.clean", f"touch {marker}")
    with pytest.raises(GitError, match="configured filters"):
        asyncio.run(client.stage(repository, ["app.py"]))
    with pytest.raises(GitError, match="configured filters"):
        asyncio.run(client.status(repository))
    assert not marker.exists()


def test_environment_cannot_redirect_git(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path))
    assert asyncio.run(service(tmp_path).status(repository))["branch"] == "main"


def test_rename_status(tmp_path: Path, repository: Path) -> None:
    git(repository, "mv", "app.py", "renamed.py")
    state = asyncio.run(service(tmp_path).status(repository))
    assert state["files"] == [{"status": "R ", "path": "renamed.py", "original_path": "app.py"}]


def test_missing_git_is_reported(
    tmp_path: Path, repository: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "missing"))
    with pytest.raises(GitError, match="not installed"):
        asyncio.run(service(tmp_path).status(repository))


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ("import time; time.sleep(10)", "timed out"),
        ("import sys; sys.stdout.write('x' * 1000000)", "byte limit"),
    ],
)
def test_process_timeout_and_output_limit_cleanup(
    tmp_path: Path,
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    expected: str,
) -> None:
    original = asyncio.create_subprocess_exec
    processes: list[asyncio.subprocess.Process] = []

    async def launch(*arguments: str, **kwargs: Any) -> asyncio.subprocess.Process:
        process = await original(sys.executable, "-c", script, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    client = GitService(RepositoryAccess(tmp_path), timeout_seconds=0.5, max_output_bytes=256)
    with pytest.raises(GitError, match=expected):
        asyncio.run(client.status(repository))
    assert len(processes) == 1
    assert processes[0].returncode is not None


def test_failed_commit_preserves_head_index_and_working_files(
    tmp_path: Path, repository: Path
) -> None:
    client = service(tmp_path)
    original_head = git(repository, "rev-parse", "HEAD")
    (repository / "app.py").write_text("staged change\n", encoding="utf-8")
    asyncio.run(client.stage(repository, ["app.py"]))
    original_index = git(repository, "write-tree")
    git(repository, "config", "--unset", "user.name")
    git(repository, "config", "--unset", "user.email")
    git(repository, "config", "user.useConfigOnly", "true")
    with pytest.raises(GitError, match="failed"):
        asyncio.run(client.commit(repository, "should fail"))
    assert git(repository, "rev-parse", "HEAD") == original_head
    assert git(repository, "write-tree") == original_index
    assert (repository / "app.py").read_text() == "staged change\n"


def test_deleted_directory_cannot_stage_multiple_files(tmp_path: Path, repository: Path) -> None:
    (repository / "directory").mkdir()
    (repository / "directory" / "one.txt").write_text("one\n", encoding="utf-8")
    git(repository, "add", "directory")
    git(repository, "commit", "-m", "add directory")
    (repository / "directory" / "one.txt").unlink()
    (repository / "directory").rmdir()
    with pytest.raises(GitError, match="one tracked file"):
        asyncio.run(service(tmp_path).stage(repository, ["directory"]))
    assert git(repository, "diff", "--cached", "--name-only") == ""


def test_git_metadata_outside_workspace_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    path = workspace / "repository"
    path.mkdir(parents=True)
    git(path, "init", "--separate-git-dir", str(tmp_path / "outside.git"))
    client = GitService(RepositoryAccess(workspace))
    with pytest.raises(GitError, match="metadata is outside"):
        asyncio.run(client.status(path))


def test_tool_registry_gates_mutations(tmp_path: Path) -> None:
    read_only = create_agent_tool_registry(service(tmp_path, mutations=False))
    enabled = create_agent_tool_registry(service(tmp_path))
    assert {tool.name for tool in read_only.definitions()} == {
        "list_files",
        "read_file",
        "search_code",
        "git_status",
        "git_diff",
        "git_log",
        "git_branches",
    }
    assert {tool.name for tool in enabled.definitions()} - {
        tool.name for tool in read_only.definitions()
    } == {"git_create_branch", "git_checkout", "git_stage", "git_unstage", "git_commit"}


def test_git_tools_validate_and_persist_redacted_events(tmp_path: Path, repository: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    (repository / "app.py").write_text("api_key=sk-abcdefghijklmnopqrstuvwxyz\n", encoding="utf-8")
    git_client = service(tmp_path)
    executor = ToolExecutor(create_agent_tool_registry(git_client))

    async def execute() -> None:
        with database.session_factory() as session:
            stored = Repository(name="Test", path=str(repository))
            run = AgentRun(repository=stored, task="Review Git changes")
            session.add(stored)
            session.commit()
            context = ToolContext(stored, session, git_client.access)
            for sequence, call in enumerate(
                [
                    ToolCall("status", "git_status", {}),
                    ToolCall("diff", "git_diff", {}),
                    ToolCall("invalid", "git_stage", {"paths": [], "command": "status"}),
                    ToolCall("blocked", "git_stage", {"paths": ["../outside"]}),
                ],
                start=1,
            ):
                result = await executor.execute(
                    call, context, agent_run_id=run.id, sequence_number=sequence
                )
                assert result.is_error is (sequence >= 3)
                assert "abcdefghijklmnopqrstuvwxyz" not in result.content
                if sequence == 1:
                    assert json.loads(result.content)["branch"] == "main"
            events = list(session.scalars(select(ToolEvent).order_by(ToolEvent.sequence_number)))
            assert len(events) == 4
            assert events[1].result is not None
            assert "abcdefghijklmnopqrstuvwxyz" not in str(events[1].result)
            assert events[2].is_error is True
            assert git(repository, "diff", "--cached", "--name-only") == ""

    asyncio.run(execute())
    database.dispose()
