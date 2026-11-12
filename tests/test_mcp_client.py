"""Real fake-server integrations for allowlists, authority, schemas, and session cleanup."""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.agent_runner import AgentRunner
from agent.config import Settings
from agent.database import Database
from agent.main import create_app
from agent.mcp_client import MCPConnectionError, connect_external_tools, validate_schema
from agent.mcp_config import MCPServerConfig
from agent.models import AgentRun, Repository
from agent.models_api import FakeModelClient, ModelResponse, TextContent, TokenUsage, ToolCall
from agent.repository_access import RepositoryAccess
from agent.tools import ToolRegistry

FAKE_SERVER = Path(__file__).parent / "fixtures" / "fake_mcp_server.py"


def response(*content: TextContent | ToolCall) -> ModelResponse:
    return ModelResponse(
        "response",
        "fake",
        "test",
        content,
        "tool_use" if any(isinstance(item, ToolCall) for item in content) else "end_turn",
        TokenUsage(1, 1),
    )


def client_for(
    tmp_path: Path, model: FakeModelClient, configuration: dict[str, Any], **settings: Any
) -> TestClient:
    return TestClient(
        create_app(
            Settings(  # type: ignore[call-arg]
                database_url="sqlite://",
                workspace_root=tmp_path,
                mcp_servers=configuration,
                _env_file=None,
                **settings,
            ),
            model_client=model,
        )
    )


def configuration(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    return {
        "docs": {
            "command": sys.executable,
            "args": [str(FAKE_SERVER), "--state", str(tmp_path / "state.json")],
            "allowed_tools": ["echo"],
            "read_only": True,
            **overrides,
        }
    }


def register(client: TestClient, tmp_path: Path) -> str:
    created = client.post("/repositories", json={"name": "MCP", "path": str(tmp_path)})
    assert created.status_code == 201
    return f"/repositories/{created.json()['id']}"


def test_imported_tools_are_namespaced_validated_redacted_and_audited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CODEFORGE_OPENAI_API_KEY", "must-not-inherit")
    model = FakeModelClient(
        [
            response(ToolCall("echo", "mcp_docs__echo", {"text": "password=private-value"})),
            response(TextContent("Finished.")),
        ]
    )
    with client_for(
        tmp_path, model, configuration(tmp_path, env={"MCP_TEST_VALUE": "explicit"})
    ) as client:
        body = client.post(register(client, tmp_path) + "/runs", json={"task": "Echo"}).json()
    assert body["status"] == "completed"
    event = body["tool_events"][0]
    assert event["is_error"] is False
    assert event["result"]["structured_content"]["model_key_in_env"] is False
    assert event["result"]["structured_content"]["explicit_env"] == "explicit"
    assert "private-value" not in str(body)
    names = {tool.name for tool in model.requests[0].tools}
    assert "mcp_docs__echo" in names and "mcp_docs__blocked" not in names
    imported = next(tool for tool in model.requests[0].tools if tool.name == "mcp_docs__echo")
    assert imported.input_schema["required"] == ["text"] and imported.strict is False
    state = json.loads((tmp_path / "state.json").read_text())
    assert state["calls"] == ["echo"]
    if os.name == "posix":
        assert not Path(f"/proc/{state['pid']}").exists()


@pytest.mark.parametrize("mode", ["read_only", "write"])
def test_server_annotations_cannot_grant_write_authority(tmp_path: Path, mode: str) -> None:
    model = FakeModelClient(
        [
            response(ToolCall("guess", "mcp_docs__echo", {"text": "hello"})),
            response(TextContent("Done")),
        ]
    )
    with client_for(
        tmp_path, model, configuration(tmp_path, read_only=False), code_writes_enabled=True
    ) as client:
        body = client.post(
            register(client, tmp_path) + "/runs", json={"task": "Echo", "mode": mode}
        ).json()
    assert body["tool_events"][0]["is_error"] is (mode == "read_only")
    assert (tmp_path / "state.json").exists() is (mode == "write")


@pytest.mark.parametrize(
    "arguments", [{"text": 5}, {"wrong": "field"}, {"text": "valid", "extra": 1}]
)
def test_bad_tool_arguments_do_not_reach_remote_server(
    tmp_path: Path, arguments: dict[str, Any]
) -> None:
    model = FakeModelClient(
        [response(ToolCall("bad", "mcp_docs__echo", arguments)), response(TextContent("Done"))]
    )
    with client_for(tmp_path, model, configuration(tmp_path)) as client:
        body = client.post(register(client, tmp_path) + "/runs", json={"task": "Echo"}).json()
    assert body["tool_events"][0]["is_error"] is True
    assert "calls" not in json.loads((tmp_path / "state.json").read_text())


def test_guessed_non_allowlisted_tool_is_never_dispatched(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            response(ToolCall("blocked", "mcp_docs__blocked", {"text": "hello"})),
            response(TextContent("Done")),
        ]
    )
    with client_for(tmp_path, model, configuration(tmp_path)) as client:
        body = client.post(
            register(client, tmp_path) + "/runs", json={"task": "Try blocked tool"}
        ).json()
    assert body["tool_events"][0]["is_error"] is True
    assert "calls" not in json.loads((tmp_path / "state.json").read_text())


@pytest.mark.parametrize("mode", ["missing", "duplicate", "external_ref", "loop"])
def test_invalid_discovery_fails_closed_and_closes_connection(tmp_path: Path, mode: str) -> None:
    registry = ToolRegistry()
    config = MCPServerConfig.model_validate(configuration(tmp_path)["docs"])
    config.args.extend(["--mode", mode])

    async def exercise() -> None:
        with pytest.raises((MCPConnectionError, ExceptionGroup)) as error:
            async with connect_external_tools(registry, {"docs": config}, allow_writes=False):
                raise AssertionError("Invalid discovery succeeded")
        pending = [error.value]
        while pending:
            current = pending.pop()
            if isinstance(current, ExceptionGroup):
                pending.extend(current.exceptions)
            else:
                assert isinstance(current, MCPConnectionError)
        assert registry.definitions() == ()

    asyncio.run(exercise())
    state = json.loads((tmp_path / "state.json").read_text())
    if os.name == "posix":
        assert not Path(f"/proc/{state['pid']}").exists()


def test_discovery_follows_pages_and_removes_temporary_tools_after_use(tmp_path: Path) -> None:
    registry = ToolRegistry()
    config = MCPServerConfig.model_validate(configuration(tmp_path)["docs"])
    config.args.extend(["--mode", "paginated"])
    config.allowed_tools = ["echo", "blocked"]

    async def exercise() -> None:
        async with connect_external_tools(registry, {"docs": config}, allow_writes=False):
            assert {tool.name for tool in registry.definitions()} == {
                "mcp_docs__echo",
                "mcp_docs__blocked",
            }
        assert registry.definitions() == ()

    asyncio.run(exercise())


@pytest.mark.parametrize("value", ["error", "slow", "disconnect"])
def test_remote_errors_timeouts_and_disconnects_are_persisted_as_tool_errors(
    tmp_path: Path, value: str
) -> None:
    model = FakeModelClient(
        [
            response(ToolCall("failure", "mcp_docs__echo", {"text": value})),
            response(TextContent("Done")),
        ]
    )
    with client_for(tmp_path, model, configuration(tmp_path, timeout_seconds=0.5)) as client:
        result = client.post(register(client, tmp_path) + "/runs", json={"task": "Call tool"})
    assert result.status_code == 201
    assert result.json()["tool_events"][0]["is_error"] is True


def test_empty_allowlist_does_not_launch_a_process(tmp_path: Path) -> None:
    model = FakeModelClient([response(TextContent("Done"))])
    with client_for(tmp_path, model, configuration(tmp_path, allowed_tools=[])) as client:
        result = client.post(register(client, tmp_path) + "/runs", json={"task": "Inspect"})
    assert result.json()["status"] == "completed" and not (tmp_path / "state.json").exists()


@pytest.mark.parametrize("cancel", [False, True])
def test_overall_timeout_and_cancellation_close_sessions_and_persist_failure(
    tmp_path: Path, cancel: bool
) -> None:
    database = Database("sqlite://")
    database.create_schema()
    registry = ToolRegistry()
    model = FakeModelClient([response(ToolCall("slow", "mcp_docs__echo", {"text": "slow"}))])
    config = MCPServerConfig.model_validate(configuration(tmp_path)["docs"])
    runner = AgentRunner(
        model,
        registry,
        RepositoryAccess(tmp_path),
        mcp_servers={"docs": config},
        timeout_seconds=10 if cancel else 0.8,
    )

    async def exercise() -> None:
        with database.session_factory() as session:
            repository = Repository(name="MCP", path=str(tmp_path))
            session.add(repository)
            session.commit()
            task = asyncio.create_task(runner.run(session, repository, "Call external tool"))
            if cancel:
                for _ in range(200):
                    state = tmp_path / "state.json"
                    if state.exists() and json.loads(state.read_text()).get("calls"):
                        break
                    await asyncio.sleep(0.01)
                else:
                    raise AssertionError("External tool never started")
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                await task
            run = session.scalar(select(AgentRun))
            assert run is not None and run.status == "failed"
            assert run.error_message == (
                "Agent run was cancelled" if cancel else "Agent run timed out"
            )
            assert registry.definitions() == ()

    asyncio.run(exercise())
    state = json.loads((tmp_path / "state.json").read_text())
    if os.name == "posix":
        assert not Path(f"/proc/{state['pid']}").exists()
    database.dispose()


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "array"},
        {"type": "object", "$id": "https://outside.test/schema"},
        {"type": "object", "$ref": "file:///etc/passwd"},
        {"type": "object", "properties": 4},
    ],
)
def test_untrusted_schema_references_and_invalid_shapes_are_rejected(
    schema: dict[str, Any],
) -> None:
    with pytest.raises(MCPConnectionError):
        validate_schema(schema)


def test_local_schema_references_work_without_remote_resolution() -> None:
    validator = validate_schema(
        {
            "type": "object",
            "$defs": {"text": {"type": "string"}},
            "properties": {"text": {"$ref": "#/$defs/text"}},
        }
    )
    validator.validate({"text": "hello"})


@pytest.mark.parametrize(
    "config",
    [
        {"command": ""},
        {"transport": "streamable_http"},
        {"command": "python", "url": "http://localhost"},
        {"command": "python", "allowed_tools": ["echo", "echo"]},
        {"command": "python", "allowed_tools": ["invalid.name"]},
        {"command": "python", "timeout_seconds": 0},
    ],
)
def test_invalid_connection_policy_is_rejected(config: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MCPServerConfig.model_validate(config)


def test_streamable_http_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    state = tmp_path / "http.json"
    with subprocess.Popen(
        [
            sys.executable,
            str(FAKE_SERVER),
            "--http",
            "--state",
            str(state),
            "--http-token",
            "test-token",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as server:
        try:
            for _ in range(100):
                if state.exists() and "port" in json.loads(state.read_text()):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("Fake HTTP server did not start")
            port = json.loads(state.read_text())["port"]
            model = FakeModelClient(
                [
                    response(ToolCall("http", "mcp_docs__echo", {"text": "hello"})),
                    response(TextContent("Done")),
                ]
            )
            config = {
                "docs": {
                    "transport": "streamable_http",
                    "url": f"http://127.0.0.1:{port}/mcp",
                    "headers": {"Authorization": "Bearer test-token"},
                    "allowed_tools": ["echo"],
                    "read_only": True,
                }
            }
            with client_for(tmp_path, model, config) as client:
                result = client.post(register(client, tmp_path) + "/runs", json={"task": "Echo"})
            assert result.json()["status"] == "completed", result.text
            assert (
                result.json()["tool_events"][0]["result"]["structured_content"]["echo"] == "hello"
            )
        finally:
            server.terminate()
            server.wait(timeout=5)
