"""Database models for repositories and agent activity."""

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Uuid as SqlUuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


class Base(DeclarativeBase):
    """Base class for all database models."""


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Store timestamps portably and always restore UTC-aware values."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class AgentRunStatus(enum.StrEnum):
    """Lifecycle states for an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentRunMode(enum.StrEnum):
    """Explicit authority selected by the caller for one run."""

    READ_ONLY = "read_only"
    WRITE = "write"


class AgentRunType(enum.StrEnum):
    """Workflow enforced independently of the model's instructions."""

    TASK = "task"
    BUG_FIX = "bug_fix"


class AgentMessageRole(enum.StrEnum):
    """Roles stored in an agent conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Repository(Base):
    """A source repository known to CodeForge Agent."""

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(),
        default=utc_now,
        onupdate=utc_now,
    )

    runs: Mapped[list[AgentRun]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    indexed_files: Mapped[list[IndexedFile]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )
    conversations: Mapped[list[Conversation]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )


class Conversation(Base):
    """A continuing, repository-scoped discussion with compacted older history."""

    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
    )
    title: Mapped[str] = mapped_column(String(255))
    summary: Mapped[str | None] = mapped_column(Text)
    summary_through_turn: Mapped[int] = mapped_column(Integer, default=0)
    is_running: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now, onupdate=utc_now)

    repository: Mapped[Repository] = relationship(back_populates="conversations")
    turns: Mapped[list[ConversationTurn]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationTurn.sequence_number",
    )


class ConversationTurn(Base):
    """Associate a run and its existing message records with an ordered conversation turn."""

    __tablename__ = "conversation_turns"
    __table_args__ = (UniqueConstraint("conversation_id", "sequence_number"),)

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        index=True,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        unique=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer)

    conversation: Mapped[Conversation] = relationship(back_populates="turns")
    run: Mapped[AgentRun] = relationship()


class IndexedFile(Base):
    """The latest indexed representation of one repository file."""

    __tablename__ = "indexed_files"
    __table_args__ = (UniqueConstraint("repository_id", "path"),)

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
    )
    path: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    language: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    chunk_size_lines: Mapped[int] = mapped_column(Integer)
    indexed_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    repository: Mapped[Repository] = relationship(back_populates="indexed_files")
    chunks: Mapped[list[CodeChunk]] = relationship(
        back_populates="indexed_file",
        cascade="all, delete-orphan",
        order_by="CodeChunk.chunk_index",
    )
    symbols: Mapped[list[CodeSymbol]] = relationship(
        back_populates="indexed_file",
        cascade="all, delete-orphan",
        order_by="CodeSymbol.start_line",
    )


class CodeChunk(Base):
    """A deterministic line-based section of an indexed file."""

    __tablename__ = "code_chunks"
    __table_args__ = (UniqueConstraint("indexed_file_id", "chunk_index"),)

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    indexed_file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("indexed_files.id", ondelete="CASCADE"),
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)

    indexed_file: Mapped[IndexedFile] = relationship(back_populates="chunks")


class CodeSymbol(Base):
    """A named Python declaration extracted from an indexed file."""

    __tablename__ = "code_symbols"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    indexed_file_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("indexed_files.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), index=True)
    qualified_name: Mapped[str] = mapped_column(Text, index=True)
    kind: Mapped[str] = mapped_column(String(32))
    signature: Mapped[str] = mapped_column(Text)
    start_line: Mapped[int] = mapped_column(Integer)
    end_line: Mapped[int] = mapped_column(Integer)

    indexed_file: Mapped[IndexedFile] = relationship(back_populates="symbols")


class AgentRun(Base):
    """One agent task executed against a repository."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
    )
    task: Mapped[str] = mapped_column(Text)
    run_type: Mapped[AgentRunType] = mapped_column(
        Enum(AgentRunType, native_enum=False), default=AgentRunType.TASK, server_default="TASK"
    )
    repair_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    verified: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0")
    mode: Mapped[AgentRunMode] = mapped_column(
        Enum(AgentRunMode, native_enum=False),
        default=AgentRunMode.READ_ONLY,
        server_default="READ_ONLY",
    )
    status: Mapped[AgentRunStatus] = mapped_column(
        Enum(AgentRunStatus, native_enum=False),
        default=AgentRunStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())

    repository: Mapped[Repository] = relationship(back_populates="runs")
    tool_events: Mapped[list[ToolEvent]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ToolEvent.sequence_number",
    )
    messages: Mapped[list[AgentMessage]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="AgentMessage.sequence_number",
    )
    test_results: Mapped[list[PytestExecution]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="(PytestExecution.created_at, PytestExecution.id)",
    )
    analysis_results: Mapped[list[AnalysisExecution]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="(AnalysisExecution.created_at, AnalysisExecution.id)",
    )


class AnalysisExecution(Base):
    """Bounded, sanitized static diagnostics independent of conversational truncation."""

    __tablename__ = "analysis_executions"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True
    )
    tool_call_id: Mapped[str] = mapped_column(String(255))
    tool_name: Mapped[str] = mapped_column(String(32))
    result: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    run: Mapped[AgentRun] = relationship(back_populates="analysis_results")


class PytestExecution(Base):
    """One bounded, sanitized pytest result associated with an agent run."""

    __tablename__ = "pytest_executions"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    tool_call_id: Mapped[str] = mapped_column(String(255))
    command: Mapped[list[str]] = mapped_column(JSON)
    targets: Mapped[list[str]] = mapped_column(JSON)
    config_file: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(32))
    counts: Mapped[dict[str, int]] = mapped_column(JSON)
    summary_complete: Mapped[bool] = mapped_column(Boolean)
    verified: Mapped[bool] = mapped_column(Boolean)
    exit_code: Mapped[int | None] = mapped_column(Integer)
    stdout: Mapped[str] = mapped_column(Text)
    stderr: Mapped[str] = mapped_column(Text)
    timed_out: Mapped[bool] = mapped_column(Boolean)
    stdout_truncated: Mapped[bool] = mapped_column(Boolean)
    stderr_truncated: Mapped[bool] = mapped_column(Boolean)
    error_message: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    run: Mapped[AgentRun] = relationship(back_populates="test_results")


class AgentMessage(Base):
    """One persisted message in an agent run conversation."""

    __tablename__ = "agent_messages"
    __table_args__ = (UniqueConstraint("agent_run_id", "sequence_number"),)

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer)
    role: Mapped[AgentMessageRole] = mapped_column(Enum(AgentMessageRole, native_enum=False))
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSON)
    response_id: Mapped[str | None] = mapped_column(String(255))
    provider: Mapped[str | None] = mapped_column(String(32))
    model: Mapped[str | None] = mapped_column(String(255))
    stop_reason: Mapped[str | None] = mapped_column(String(32))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    run: Mapped[AgentRun] = relationship(back_populates="messages")


class ToolEvent(Base):
    """A persisted tool invocation from an agent run."""

    __tablename__ = "tool_events"
    __table_args__ = (UniqueConstraint("agent_run_id", "sequence_number"),)

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer)
    tool_call_id: Mapped[str] = mapped_column(String(255))
    tool_name: Mapped[str] = mapped_column(String(255))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), default=utc_now)

    run: Mapped[AgentRun] = relationship(back_populates="tool_events")
