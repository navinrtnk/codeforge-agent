"""Tests for database model persistence."""

from sqlalchemy import select

from agent.database import Database
from agent.models import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentRunStatus,
    Repository,
    ToolEvent,
)


def test_models_and_relationships_are_persisted() -> None:
    database = Database("sqlite://")
    database.create_schema()

    with database.session_factory() as session:
        repository = Repository(name="Example", path="/tmp/example")
        run = AgentRun(repository=repository, task="Fix the failing test")
        event = ToolEvent(
            run=run,
            sequence_number=1,
            tool_call_id="call-1",
            tool_name="read_file",
            arguments={"path": "src/example.py"},
            result={"content": "pass"},
            duration_ms=12.5,
        )
        message = AgentMessage(
            run=run,
            sequence_number=1,
            role=AgentMessageRole.USER,
            content=[{"type": "text", "text": "Fix the failing test"}],
        )
        session.add(repository)
        session.commit()

        stored_repository = session.scalar(select(Repository))
        stored_run = session.scalar(select(AgentRun))
        stored_event = session.scalar(select(ToolEvent))

        assert stored_repository is not None
        assert stored_repository.runs == [run]
        assert stored_run is not None
        assert stored_run.status is AgentRunStatus.PENDING
        assert stored_run.tool_events == [event]
        assert stored_run.messages == [message]
        assert stored_event is not None
        assert stored_event.arguments == {"path": "src/example.py"}
        assert stored_event.is_error is False

    database.dispose()


def test_deleting_repository_cascades_to_agent_activity() -> None:
    database = Database("sqlite://")
    database.create_schema()

    with database.session_factory() as session:
        repository = Repository(name="Example", path="/tmp/example")
        run = AgentRun(repository=repository, task="Review the change")
        run.tool_events.append(
            ToolEvent(
                sequence_number=1,
                tool_call_id="call-1",
                tool_name="git_diff",
                arguments={},
            )
        )
        run.messages.append(
            AgentMessage(
                sequence_number=1,
                role=AgentMessageRole.USER,
                content=[{"type": "text", "text": "Review the change"}],
            )
        )
        session.add(repository)
        session.commit()
        session.delete(repository)
        session.commit()

        assert session.scalar(select(Repository)) is None
        assert session.scalar(select(AgentRun)) is None
        assert session.scalar(select(ToolEvent)) is None
        assert session.scalar(select(AgentMessage)) is None

    database.dispose()
