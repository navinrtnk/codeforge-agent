"""Tests for constrained repository tool execution."""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.database import Database
from agent.indexing import RepositoryIndexer
from agent.models import AgentRun, Repository, ToolEvent
from agent.models_api import JsonObject, ToolCall
from agent.repository_access import RepositoryAccess
from agent.repository_tools import create_repository_tool_registry
from agent.tools import ToolContext, ToolExecutor, ToolRegistry, redact_json_object


def execute_tool(
    executor: ToolExecutor,
    call: ToolCall,
    context: ToolContext,
    run: AgentRun,
    sequence_number: int,
) -> dict[str, Any]:
    result = asyncio.run(
        executor.execute(
            call,
            context,
            agent_run_id=run.id,
            sequence_number=sequence_number,
        )
    )
    return {"result": result, "content": json.loads(result.content)}


def create_context(tmp_path: Path, session: Session) -> tuple[ToolContext, AgentRun]:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    repository = Repository(name="Example", path=str(repository_path))
    run = AgentRun(repository=repository, task="Inspect the repository")
    session.add(repository)
    session.commit()
    access = RepositoryAccess(
        tmp_path,
        ignore_patterns=(".git", "__pycache__", "*.pyc"),
    )
    return ToolContext(repository=repository, session=session, access=access), run


def test_registry_exposes_stable_strict_tool_definitions() -> None:
    registry = create_repository_tool_registry()

    definitions = registry.definitions()

    assert [definition.name for definition in definitions] == [
        "list_files",
        "read_file",
        "search_code",
    ]
    assert all(definition.strict for definition in definitions)
    read_file = next(item for item in definitions if item.name == "read_file")
    assert read_file.input_schema["additionalProperties"] is False
    assert "path" in read_file.input_schema["required"]


def test_registry_rejects_duplicate_tool_names() -> None:
    registry = ToolRegistry()
    tool = create_repository_tool_registry().get("read_file")
    assert tool is not None
    registry.register(tool)

    with pytest.raises(ValueError, match="already registered"):
        registry.register(tool)


def test_repository_tools_list_read_search_and_persist_events(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        context, run = create_context(tmp_path, session)
        repository_path = Path(context.repository.path)
        (repository_path / "app.py").write_text(
            "one\nneedle = True\nthree\n",
            encoding="utf-8",
        )
        (repository_path / "README.md").write_text("documentation\n", encoding="utf-8")
        (repository_path / ".git").mkdir()
        (repository_path / ".git" / "config").write_text("ignored", encoding="utf-8")
        RepositoryIndexer(context.access, chunk_size_lines=20).index(session, context.repository)
        executor = ToolExecutor(create_repository_tool_registry())

        listed = execute_tool(
            executor,
            ToolCall("call-1", "list_files", {"limit": 10}),
            context,
            run,
            1,
        )
        read = execute_tool(
            executor,
            ToolCall(
                "call-2",
                "read_file",
                {"path": "app.py", "start_line": 2, "end_line": 2},
            ),
            context,
            run,
            2,
        )
        searched = execute_tool(
            executor,
            ToolCall(
                "call-3",
                "search_code",
                {"query": "needle", "mode": "exact"},
            ),
            context,
            run,
            3,
        )

        assert [file["path"] for file in listed["content"]["files"]] == [
            "README.md",
            "app.py",
        ]
        assert read["content"]["content"] == "needle = True\n"
        assert read["content"]["start_line"] == 2
        assert searched["content"]["results"][0]["path"] == "app.py"
        events = list(session.scalars(select(ToolEvent).order_by(ToolEvent.sequence_number)))
        assert [event.tool_call_id for event in events] == ["call-1", "call-2", "call-3"]
        assert [event.tool_name for event in events] == [
            "list_files",
            "read_file",
            "search_code",
        ]
        assert all(event.duration_ms is not None and event.duration_ms >= 0 for event in events)
        assert all(event.is_error is False for event in events)

    database.dispose()


def test_executor_rejects_unknown_and_malformed_calls(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        context, run = create_context(tmp_path, session)
        executor = ToolExecutor(create_repository_tool_registry())

        unknown = execute_tool(
            executor,
            ToolCall("call-1", "shell", {"command": "whoami"}),
            context,
            run,
            1,
        )
        malformed = execute_tool(
            executor,
            ToolCall("call-2", "read_file", {"unexpected": "value"}),
            context,
            run,
            2,
        )
        traversal = execute_tool(
            executor,
            ToolCall("call-3", "read_file", {"path": "../secret.txt"}),
            context,
            run,
            3,
        )

        assert unknown["result"].is_error is True
        assert unknown["content"] == {"error": "Unknown tool: shell"}
        assert malformed["result"].is_error is True
        assert "Invalid arguments" in malformed["content"]["error"]
        assert "unexpected" in malformed["content"]["error"]
        assert traversal["result"].is_error is True
        assert traversal["content"] == {"error": "File path is outside the repository"}
        assert session.scalar(select(ToolEvent).where(ToolEvent.sequence_number == 1)) is not None

    database.dispose()


class EchoArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]


class EchoTool:
    name = "echo"
    description = "Echo test data"
    arguments_model = EchoArguments

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        validated = EchoArguments.model_validate(arguments)
        return {
            "echo": validated.payload,
            "authorization": "Bearer very-secret-credential",
            "note": "token=another-secret-value",
        }


class LargeTool:
    name = "large"
    description = "Return large test data"
    arguments_model = EchoArguments

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        return {"content": "x" * 2_000}


def test_executor_redacts_arguments_results_and_model_output(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        context, run = create_context(tmp_path, session)
        registry = ToolRegistry()
        registry.register(EchoTool())
        executor = ToolExecutor(registry)

        executed = execute_tool(
            executor,
            ToolCall(
                "call-1",
                "echo",
                {
                    "payload": {
                        "api_key": "sk-abcdefghijklmnopqrstuvwxyz",
                        "message": "password=hunter2",
                    }
                },
            ),
            context,
            run,
            1,
        )

        serialized = executed["result"].content
        assert "abcdefghijklmnopqrstuvwxyz" not in serialized
        assert "hunter2" not in serialized
        assert "very-secret-credential" not in serialized
        assert "another-secret-value" not in serialized
        event = session.scalar(select(ToolEvent))
        assert event is not None
        assert event.arguments["payload"]["api_key"] == "[REDACTED]"
        assert event.result is not None
        assert event.result["authorization"] == "[REDACTED]"

    database.dispose()


def test_executor_bounds_returned_and_persisted_output(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        context, run = create_context(tmp_path, session)
        registry = ToolRegistry()
        registry.register(LargeTool())
        executor = ToolExecutor(registry, max_output_bytes=256)

        executed = execute_tool(
            executor,
            ToolCall("call-1", "large", {"payload": {}}),
            context,
            run,
            1,
        )

        assert len(executed["result"].content.encode("utf-8")) <= 256
        assert executed["content"]["truncated"] is True
        assert executed["content"]["original_bytes"] > 256
        event = session.scalar(select(ToolEvent))
        assert event is not None
        assert event.result == executed["content"]

    database.dispose()


def test_executor_rejects_mismatched_agent_run_repository(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        context, _ = create_context(tmp_path, session)
        other_path = tmp_path / "other"
        other_path.mkdir()
        other = Repository(name="Other", path=str(other_path))
        other_run = AgentRun(repository=other, task="Other task")
        session.add(other)
        session.commit()
        executor = ToolExecutor(create_repository_tool_registry())

        with pytest.raises(ValueError, match="does not belong"):
            asyncio.run(
                executor.execute(
                    ToolCall("call-1", "list_files", {}),
                    context,
                    agent_run_id=other_run.id,
                    sequence_number=1,
                )
            )

    database.dispose()


def test_redact_json_object_handles_nested_common_secret_formats() -> None:
    redacted = redact_json_object(
        {
            "nested": [
                {"password": "visible"},
                "Bearer abc.def.ghi",
                "ghp_abcdefghijklmnopqrstuvwxyz1234",
            ]
        }
    )

    assert redacted == {
        "nested": [
            {"password": "[REDACTED]"},
            "[REDACTED]",
            "[REDACTED]",
        ]
    }
