"""Database models for repositories and agent activity."""

import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy import Uuid as SqlUuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""


def utc_now() -> datetime:
    """Return the current time in UTC."""
    return datetime.now(UTC)


class AgentRunStatus(enum.StrEnum):
    """Lifecycle states for an agent run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Repository(Base):
    """A source repository known to CodeForge Agent."""

    __tablename__ = "repositories"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
    )

    runs: Mapped[list[AgentRun]] = relationship(
        back_populates="repository",
        cascade="all, delete-orphan",
    )


class AgentRun(Base):
    """One agent task executed against a repository."""

    __tablename__ = "agent_runs"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    repository_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
    )
    task: Mapped[str] = mapped_column(Text)
    status: Mapped[AgentRunStatus] = mapped_column(
        Enum(AgentRunStatus, native_enum=False),
        default=AgentRunStatus.PENDING,
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    repository: Mapped[Repository] = relationship(back_populates="runs")
    tool_events: Mapped[list[ToolEvent]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ToolEvent.sequence_number",
    )


class ToolEvent(Base):
    """A persisted tool invocation from an agent run."""

    __tablename__ = "tool_events"

    id: Mapped[uuid.UUID] = mapped_column(SqlUuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(Integer)
    tool_name: Mapped[str] = mapped_column(String(255))
    arguments: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    is_error: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[float | None] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)

    run: Mapped[AgentRun] = relationship(back_populates="tool_events")
