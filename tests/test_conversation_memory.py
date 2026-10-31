"""Boundary tests for atomic turn retention, summary budgets, and interrupted requests."""

import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.agent_runner import AgentRunner
from agent.conversation_memory import (
    ConversationBusyError,
    ConversationMemory,
    history_cost,
)
from agent.database import Database
from agent.models import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationTurn,
    Repository,
)
from agent.models_api import (
    FakeModelClient,
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolResult,
    validate_model_request,
)
from agent.repository_access import RepositoryAccess
from agent.repository_tools import create_repository_tool_registry


def response(text: str) -> ModelResponse:
    return ModelResponse(
        "summary", "fake", "test-model", (TextContent(text),), "end_turn", TokenUsage(10, 5)
    )


def setup(tmp_path: Path) -> tuple[Database, Repository, Conversation]:
    path = tmp_path / "repo"
    path.mkdir()
    database = Database(f"sqlite:///{tmp_path / 'test.db'}")
    database.create_schema()
    with database.session_factory() as session:
        repository = Repository(name="Test", path=str(path))
        conversation = Conversation(repository=repository, title="Discussion")
        session.add(repository)
        session.commit()
        session.expunge_all()
    return database, repository, conversation


def store_turn(
    session: Session, conversation: Conversation, number: int, messages: tuple[Message, ...]
) -> None:
    from agent.agent_runner import _serialize_content

    run = AgentRun(
        repository_id=conversation.repository_id,
        task="Previous task",
        status=AgentRunStatus.COMPLETED,
    )
    for index, message in enumerate(messages, start=1):
        run.messages.append(
            AgentMessage(
                sequence_number=index,
                role=AgentMessageRole(message.role),
                content=_serialize_content(message.content),
            )
        )
    session.add(ConversationTurn(conversation_id=conversation.id, run=run, sequence_number=number))
    session.commit()


def test_trim_keeps_a_complete_recent_tool_turn_and_reuses_summary(tmp_path: Path) -> None:
    database, _, original = setup(tmp_path)
    recent = (
        Message.text("user", "Read a.py"),
        Message("assistant", (ToolCall("call", "read_file", {"path": "a.py"}),)),
        Message("user", (ToolResult("call", '{"content":"value=42"}'),)),
        Message.text("assistant", "Value is 42"),
    )
    model = FakeModelClient([response("Previous request prefers concise answers")])
    memory = ConversationMemory(model, history_token_budget=1500)
    with database.session_factory() as session:
        conversation = session.get(Conversation, original.id)
        assert conversation is not None
        store_turn(
            session,
            conversation,
            1,
            (Message.text("user", "old" * 2000), Message.text("assistant", "Old answer")),
        )
        store_turn(session, conversation, 2, recent)
        prepared = asyncio.run(memory.prepare_history(session, conversation))
        assert prepared[1:] == recent
        assert history_cost(prepared) <= 1500
        assert conversation.summary_through_turn == 1
        assert asyncio.run(memory.prepare_history(session, conversation)) == prepared
        assert len(model.requests) == 1
    database.dispose()


class SummaryClient:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        validate_model_request(request)
        self.requests.append(request)
        return response("記憶" * 200)


def test_summary_batches_consume_full_transcript_with_bounded_unicode_context(
    tmp_path: Path,
) -> None:
    database, _, original = setup(tmp_path)
    model = SummaryClient()
    memory = ConversationMemory(model, history_token_budget=256, summary_input_token_budget=256)
    with database.session_factory() as session:
        conversation = session.get(Conversation, original.id)
        assert conversation is not None
        store_turn(
            session,
            conversation,
            1,
            (
                Message.text("user", "偏好" * 200),
                Message.text("assistant", "LAST_EXCERPT"),
            ),
        )
        prepared = asyncio.run(memory.prepare_history(session, conversation))
        assert history_cost(prepared) <= 256
        assert conversation.summary is not None
        assert "記憶" in conversation.summary
        assert conversation.summary_through_turn == 1
    assert len(model.requests) > 1
    assert "LAST_EXCERPT" in str(model.requests[-1].messages)
    for request in model.requests:
        source = request.messages[1].content[0]
        assert isinstance(source, TextContent)
        assert len(source.text.encode("utf-8")) <= 256
        assert request.tools == ()
    database.dispose()


@pytest.mark.parametrize(
    "bad_response",
    [
        response(""),
        replace(response("Truncated"), stop_reason="max_tokens"),
        replace(response("Tool call"), content=(ToolCall("call", "read_file", {}),)),
    ],
)
def test_invalid_summaries_do_not_advance_checkpoint(
    tmp_path: Path, bad_response: ModelResponse
) -> None:
    database, _, original = setup(tmp_path)
    memory = ConversationMemory(FakeModelClient([bad_response]), history_token_budget=256)
    with database.session_factory() as session:
        conversation = session.get(Conversation, original.id)
        assert conversation is not None
        store_turn(
            session,
            conversation,
            1,
            (Message.text("user", "long" * 500), Message.text("assistant", "Answer")),
        )
        with pytest.raises(ModelResponseError, match="complete conversation summary"):
            asyncio.run(memory.prepare_history(session, conversation))
        assert conversation.summary is None
        assert conversation.summary_through_turn == 0
    database.dispose()


class BlockingClient:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.entered.set()
        await asyncio.Event().wait()
        raise AssertionError("Unreachable")


def test_concurrent_append_is_rejected_and_cancellation_releases_claim(tmp_path: Path) -> None:
    database, repository, original = setup(tmp_path)

    async def exercise() -> None:
        model = BlockingClient()
        memory = ConversationMemory(model)
        runner = AgentRunner(model, create_repository_tool_registry(), RepositoryAccess(tmp_path))
        with (
            database.session_factory() as first_session,
            database.session_factory() as second_session,
        ):
            conversation = first_session.get(Conversation, original.id)
            other = second_session.get(Conversation, original.id)
            first_repository = first_session.get(Repository, repository.id)
            second_repository = second_session.get(Repository, repository.id)
            assert conversation is not None and other is not None
            assert first_repository is not None and second_repository is not None
            pending = asyncio.create_task(
                memory.append(first_session, conversation, first_repository, "Task", runner)
            )
            try:
                await asyncio.wait_for(model.entered.wait(), timeout=2)
            except TimeoutError:
                if pending.done():
                    await pending
                pending.cancel()
                raise
            with pytest.raises(ConversationBusyError):
                await memory.append(second_session, other, second_repository, "Concurrent", runner)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        with database.session_factory() as session:
            saved = session.get(Conversation, original.id)
            run = session.scalar(select(AgentRun))
            turn = session.scalar(select(ConversationTurn))
            assert saved is not None and saved.is_running is False
            assert run is not None and run.status is AgentRunStatus.FAILED
            assert run.error_message == "Agent run was cancelled"
            assert turn is not None and turn.run_id == run.id

    asyncio.run(exercise())
    database.dispose()
