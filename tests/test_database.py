"""Tests for database model persistence."""

from sqlalchemy import select, text

from agent.database import Database
from agent.models import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentRunMode,
    AgentRunStatus,
    AgentRunType,
    Conversation,
    ConversationTurn,
    PytestExecution,
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
        assert stored_run.mode is AgentRunMode.READ_ONLY
        assert stored_run.tool_events == [event]
        assert stored_run.messages == [message]
        assert stored_event is not None
        assert stored_event.arguments == {"path": "src/example.py"}
        assert stored_event.is_error is False

    database.dispose()


def test_existing_database_receives_read_only_mode_without_losing_runs() -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        repository = Repository(name="Existing", path="/tmp/existing")
        run = AgentRun(repository=repository, task="Existing task")
        session.add(repository)
        session.commit()
        run_id = run.id
    # Simulate the exact pre-Epic-11 schema.
    with database.engine.begin() as connection:
        connection.execute(text("ALTER TABLE agent_runs DROP COLUMN mode"))
    database.create_schema()
    database.create_schema()
    with database.session_factory() as session:
        saved = session.get(AgentRun, run_id)
        assert saved is not None
        assert saved.mode is AgentRunMode.READ_ONLY
        assert saved.task == "Existing task"
    database.dispose()


def test_deleting_repository_cascades_to_agent_activity() -> None:
    database = Database("sqlite://")
    database.create_schema()

    with database.session_factory() as session:
        repository = Repository(name="Example", path="/tmp/example")
        run = AgentRun(repository=repository, task="Review the change")
        conversation = Conversation(repository=repository, title="Review discussion")
        conversation.turns.append(ConversationTurn(run=run, sequence_number=1))
        run.test_results.append(
            PytestExecution(
                tool_call_id="test-call",
                command=["python", "-m", "pytest"],
                targets=[],
                config_file=None,
                status="passed",
                counts={"passed": 1},
                summary_complete=True,
                verified=True,
                exit_code=0,
                stdout="1 passed in 0.01s",
                stderr="",
                timed_out=False,
                stdout_truncated=False,
                stderr_truncated=False,
                error_message=None,
                duration_ms=10,
            )
        )
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
        assert session.scalar(select(Conversation)) is None
        assert session.scalar(select(ConversationTurn)) is None
        assert session.scalar(select(PytestExecution)) is None

    database.dispose()


def test_existing_runs_receive_repair_defaults_without_data_loss() -> None:
    database = Database("sqlite://")
    database.create_schema()
    with database.session_factory() as session:
        run = AgentRun(repository=Repository(name="Existing", path="/existing"), task="Inspect")
        session.add(run)
        session.commit()
        run_id = run.id
    with database.engine.begin() as connection:
        for name in ("run_type", "repair_attempts", "verified"):
            connection.execute(text(f"ALTER TABLE agent_runs DROP COLUMN {name}"))
    database.create_schema()
    database.create_schema()
    with database.session_factory() as session:
        saved = session.get(AgentRun, run_id)
        assert saved is not None and saved.task == "Inspect"
        assert saved.run_type is AgentRunType.TASK
        assert saved.repair_attempts == 0 and saved.verified is False
    database.dispose()
