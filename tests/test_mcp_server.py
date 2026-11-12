"""MCP server contracts exercised through the official client and real stdio transport."""

import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from mcp import Client, StdioServerParameters
from sqlalchemy import select

from agent.config import Settings
from agent.database import Database
from agent.indexing import RepositoryIndexer
from agent.mcp_server import create_mcp_server
from agent.models import AgentRun, PytestExecution, Repository, ToolEvent
from agent.repository_access import RepositoryAccess


def settings_for(tmp_path: Path, **overrides: Any) -> Settings:
    return Settings(database_url="sqlite://", workspace_root=tmp_path, _env_file=None, **overrides)  # type: ignore[call-arg]


def fixture_repository(tmp_path: Path, database: Database) -> Repository:
    path = tmp_path / "repository"
    path.mkdir()
    (path / "app.py").write_text("value = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "add", "app.py"], check=True)
    (path / "app.py").write_text("value = 2\n", encoding="utf-8")
    with database.session_factory() as session:
        repository = Repository(name="MCP", path=str(path))
        session.add(repository)
        session.commit()
        RepositoryIndexer(RepositoryAccess(tmp_path, ignore_patterns=(".git",))).index(
            session, repository
        )
        return repository


def test_server_lists_only_native_tools_and_returns_audited_repository_results(
    tmp_path: Path,
) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)

    async def exercise() -> None:
        async with Client(create_mcp_server(settings_for(tmp_path), database=database)) as client:
            definitions = await client.list_tools()
            assert {tool.name for tool in definitions.tools} == {
                "search_code",
                "read_file",
                "git_diff",
            }
            assert all(
                "repository_id" in tool.input_schema["required"] for tool in definitions.tools
            )
            arguments = {"repository_id": str(repository.id)}
            read = await client.call_tool("read_file", {**arguments, "path": "app.py"})
            assert read.structured_content is not None
            assert read.structured_content["content"] == "value = 2\n" and not read.is_error
            search = await client.call_tool("search_code", {**arguments, "query": "value"})
            assert search.structured_content is not None
            assert search.structured_content["results"][0]["path"] == "app.py"
            diff = await client.call_tool("git_diff", arguments)
            assert (
                diff.structured_content is not None
                and "+value = 2" in diff.structured_content["patch"]
            )

    asyncio.run(exercise())
    with database.session_factory() as session:
        assert len(list(session.scalars(select(AgentRun)))) == 3
        assert len(list(session.scalars(select(ToolEvent)))) == 3
        assert all(
            run.status == "completed" and run.completed_at
            for run in session.scalars(select(AgentRun))
        )
    database.dispose()


@pytest.mark.skipif(os.getenv("CODEFORGE_DOCKER_TESTS") != "1", reason="Docker opt-in required")
def test_packaged_mcp_entry_point_in_docker() -> None:
    name = f"codeforge-mcp-test-{uuid.uuid4().hex}"

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command="docker",
            args=[
                "run",
                "--rm",
                "-i",
                "--name",
                name,
                "--network=none",
                "--read-only",
                "--tmpfs",
                "/data:rw,nosuid,nodev,size=32m,mode=1777",
                "--env",
                "CODEFORGE_WORKSPACE_ROOT=/workspace",
                "codeforge-agent:local",
                "codeforge-mcp",
            ],
        )
        async with Client(parameters, read_timeout_seconds=10) as client:
            tools = await client.list_tools()
            assert {tool.name for tool in tools.tools} == {"read_file", "search_code", "git_diff"}

    try:
        asyncio.run(exercise())
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


@pytest.mark.parametrize(
    ("writes", "tests"), [(False, False), (False, True), (True, False), (True, True)]
)
def test_test_execution_requires_both_server_operator_flags(
    tmp_path: Path, writes: bool, tests: bool
) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)
    (Path(repository.path) / "test_app.py").write_text("def test_pass():\n    assert True\n")

    async def exercise() -> None:
        server = create_mcp_server(
            settings_for(tmp_path, code_writes_enabled=writes, test_execution_enabled=tests),
            database=database,
        )
        async with Client(server) as client:
            definitions = await client.list_tools()
            assert ("run_pytest" in {tool.name for tool in definitions.tools}) is (writes and tests)
            if writes and tests:
                result = await client.call_tool("run_pytest", {"repository_id": str(repository.id)})
                assert (
                    result.structured_content is not None
                    and result.structured_content["verified"] is True
                )

    asyncio.run(exercise())
    with database.session_factory() as session:
        assert bool(session.scalar(select(PytestExecution))) is (writes and tests)
    database.dispose()


def test_server_rejects_bad_arguments_unknown_repositories_and_escaping_paths(
    tmp_path: Path,
) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)
    (tmp_path / "private.txt").write_text("private-host-content")

    async def exercise() -> None:
        async with Client(create_mcp_server(settings_for(tmp_path), database=database)) as client:
            bad = await client.call_tool(
                "read_file", {"repository_id": str(repository.id), "path": "../private.txt"}
            )
            assert bad.is_error and "private-host-content" not in str(bad)
            missing = await client.call_tool(
                "read_file",
                {"repository_id": "00000000-0000-0000-0000-000000000000", "path": "app.py"},
            )
            assert missing.is_error
            invalid = await client.call_tool(
                "read_file", {"repository_id": "invalid", "path": "app.py"}
            )
            assert invalid.is_error
            disabled = await client.call_tool("run_pytest", {"repository_id": str(repository.id)})
            assert disabled.is_error

    asyncio.run(exercise())
    database.dispose()


def test_server_rechecks_workspace_before_returning_indexed_content(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)
    workspace = tmp_path / "restricted"
    workspace.mkdir()

    async def exercise() -> None:
        async with Client(create_mcp_server(settings_for(workspace), database=database)) as client:
            result = await client.call_tool(
                "search_code", {"repository_id": str(repository.id), "query": "value"}
            )
            assert result.is_error and "value = 2" not in str(result)

    asyncio.run(exercise())
    database.dispose()


def test_stdio_entry_point_uses_registered_database_without_a_model_key(tmp_path: Path) -> None:
    database = Database(f"sqlite:///{tmp_path / 'mcp.db'}")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)
    database.dispose()

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "agent.mcp_server"],
            cwd=str(tmp_path),
            env={
                "CODEFORGE_DATABASE_URL": f"sqlite:///{tmp_path / 'mcp.db'}",
                "CODEFORGE_WORKSPACE_ROOT": str(tmp_path),
            },
        )
        async with Client(parameters, read_timeout_seconds=5) as client:
            result = await client.call_tool(
                "read_file", {"repository_id": str(repository.id), "path": "app.py"}
            )
            assert not result.is_error
            assert json.loads(result.content[0].text)["content"] == "value = 2\n"  # type: ignore[union-attr]

    asyncio.run(exercise())


def test_server_test_timeout_persists_unverified_test_evidence(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = fixture_repository(tmp_path, database)
    (Path(repository.path) / "test_slow.py").write_text(
        "import time\ndef test_slow():\n    time.sleep(5)\n"
    )

    async def exercise() -> None:
        server = create_mcp_server(
            settings_for(
                tmp_path,
                code_writes_enabled=True,
                test_execution_enabled=True,
                agent_timeout_seconds=0.3,
            ),
            database=database,
        )
        async with Client(server) as client:
            result = await client.call_tool("run_pytest", {"repository_id": str(repository.id)})
            assert result.is_error

    asyncio.run(exercise())
    with database.session_factory() as session:
        run = session.scalar(select(AgentRun))
        execution = session.scalar(select(PytestExecution))
        assert run is not None and run.status == "failed" and "timed out" in str(run.error_message)
        assert execution is not None and not execution.verified
        assert execution.status == "cancelled"
    database.dispose()
