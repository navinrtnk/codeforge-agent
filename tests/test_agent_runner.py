"""Tests for bounded agent orchestration and conversation persistence."""

import asyncio
from pathlib import Path

from sqlalchemy import select

from agent.agent_runner import AgentRunner
from agent.database import Database
from agent.models import AgentMessage, AgentMessageRole, AgentRunStatus, Repository, ToolEvent
from agent.models_api import (
    FakeModelClient,
    ModelRequest,
    ModelResponse,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolResult,
)
from agent.repository_access import RepositoryAccess
from agent.repository_tools import create_repository_tool_registry


def model_response(
    response_id: str,
    *content: TextContent | ToolCall,
) -> ModelResponse:
    return ModelResponse(
        id=response_id,
        provider="fake",
        model="test-model",
        content=content,
        stop_reason=(
            "tool_use" if any(isinstance(item, ToolCall) for item in content) else "end_turn"
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=5),
    )


def create_repository(tmp_path: Path, database: Database) -> Repository:
    repository_path = tmp_path / "repository"
    repository_path.mkdir()
    (repository_path / "app.py").write_text("value = 42\n", encoding="utf-8")
    with database.session_factory() as session:
        repository = Repository(name="Example", path=str(repository_path))
        session.add(repository)
        session.commit()
        session.expunge(repository)
    return repository


def test_runner_executes_tools_and_persists_conversation(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    client = FakeModelClient(
        responses=[
            model_response(
                "response-1",
                TextContent("I will inspect the file."),
                ToolCall("call-1", "read_file", {"path": "app.py"}),
            ),
            model_response("response-2", TextContent("The value is 42.")),
        ]
    )
    runner = AgentRunner(
        client,
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        run = asyncio.run(runner.run(session, stored_repository, "What is the value?"))

        assert run.status is AgentRunStatus.COMPLETED
        assert run.error_message is None
        assert run.completed_at is not None
        messages = list(
            session.scalars(select(AgentMessage).order_by(AgentMessage.sequence_number))
        )
        assert [message.role for message in messages] == [
            AgentMessageRole.SYSTEM,
            AgentMessageRole.USER,
            AgentMessageRole.ASSISTANT,
            AgentMessageRole.USER,
            AgentMessageRole.ASSISTANT,
        ]
        assert messages[2].response_id == "response-1"
        assert messages[2].input_tokens == 10
        assert messages[4].content == [{"type": "text", "text": "The value is 42."}]
        event = session.scalar(select(ToolEvent))
        assert event is not None
        assert event.tool_name == "read_file"
        assert event.result is not None
        assert event.result["content"] == "value = 42\n"

    assert len(client.requests) == 2
    assert [tool.name for tool in client.requests[0].tools] == [
        "list_files",
        "read_file",
        "search_code",
    ]
    tool_result = client.requests[1].messages[-1].content[0]
    assert isinstance(tool_result, ToolResult)
    assert "value = 42" in tool_result.content
    database.dispose()


def test_runner_fails_after_iteration_limit(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    client = FakeModelClient(
        responses=[model_response("response-1", ToolCall("call-1", "list_files", {}))]
    )
    runner = AgentRunner(
        client,
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
        max_iterations=1,
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        run = asyncio.run(runner.run(session, stored_repository, "Keep inspecting"))

        assert run.status is AgentRunStatus.FAILED
        assert run.error_message == "Agent exceeded the maximum of 1 model iterations"
        assert run.completed_at is not None
        assert len(run.tool_events) == 1

    database.dispose()


def test_runner_persists_model_failure(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    runner = AgentRunner(
        FakeModelClient(responses=[]),
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        run = asyncio.run(runner.run(session, stored_repository, "Inspect the project"))

        assert run.status is AgentRunStatus.FAILED
        assert run.error_message == "Fake model client has no queued response"

    database.dispose()


def test_runner_rejects_empty_final_response(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    runner = AgentRunner(
        FakeModelClient(responses=[model_response("response-1")]),
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        run = asyncio.run(runner.run(session, stored_repository, "Inspect the project"))

        assert run.status is AgentRunStatus.FAILED
        assert run.error_message == "Model returned no final answer"

    database.dispose()


class SlowModelClient:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        await asyncio.sleep(0.05)
        return model_response("response-1", TextContent("Too late"))


def test_runner_enforces_timeout(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    runner = AgentRunner(
        SlowModelClient(),
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
        timeout_seconds=0.001,
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        run = asyncio.run(runner.run(session, stored_repository, "Inspect the project"))

        assert run.status is AgentRunStatus.FAILED
        assert run.error_message == "Agent run timed out"

    database.dispose()


def test_runner_rejects_blank_task(tmp_path: Path) -> None:
    database = Database("sqlite://")
    database.create_schema()
    repository = create_repository(tmp_path, database)
    runner = AgentRunner(
        FakeModelClient(responses=[]),
        create_repository_tool_registry(),
        RepositoryAccess(tmp_path),
    )

    with database.session_factory() as session:
        stored_repository = session.get(Repository, repository.id)
        assert stored_repository is not None
        try:
            asyncio.run(runner.run(session, stored_repository, "  "))
        except ValueError as error:
            assert str(error) == "Agent task must not be blank"
        else:
            raise AssertionError("Blank task was accepted")

    database.dispose()
