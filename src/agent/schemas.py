"""API request and response schemas."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.models import AgentMessageRole, AgentRunMode, AgentRunStatus


class RepositoryCreate(BaseModel):
    """Data required to register a repository."""

    name: str = Field(min_length=1, max_length=255)
    path: Path


class RepositoryResponse(BaseModel):
    """Public representation of a registered repository."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    path: str
    created_at: datetime
    updated_at: datetime


class IndexingResponse(BaseModel):
    """Counts produced by a repository indexing run."""

    discovered: int
    indexed: int
    updated: int
    skipped: int
    deleted: int
    failed: int


class IndexStatusResponse(BaseModel):
    """Current persisted index statistics."""

    file_count: int
    chunk_count: int
    last_indexed_at: datetime | None


class SearchResultResponse(BaseModel):
    """One matching source chunk."""

    path: str
    language: str
    start_line: int
    end_line: int
    snippet: str
    score: float


class SymbolSearchResultResponse(BaseModel):
    """One matching extracted symbol."""

    path: str
    language: str
    name: str
    qualified_name: str
    kind: str
    signature: str
    start_line: int
    end_line: int


class AgentRunCreate(BaseModel):
    """A task to execute against a registered repository."""

    task: str = Field(min_length=1, max_length=20_000)
    mode: AgentRunMode = AgentRunMode.READ_ONLY


class ConversationCreate(BaseModel):
    """Create a named discussion for one registered repository."""

    title: str = Field(default="Repository conversation", min_length=1, max_length=255)


class ConversationMessageCreate(BaseModel):
    """A follow-up task to append to a conversation."""

    content: str = Field(min_length=1, max_length=20_000)
    mode: AgentRunMode = AgentRunMode.READ_ONLY

    @field_validator("content")
    @classmethod
    def reject_blank_content(cls, value: str) -> str:
        """Reject whitespace before allocating a run or claiming a conversation."""
        if not value.strip():
            raise ValueError("Message must not be blank")
        return value


class ConversationResponse(BaseModel):
    """Conversation metadata and its current compacted history."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repository_id: uuid.UUID
    title: str
    summary: str | None
    summary_through_turn: int
    is_running: bool
    created_at: datetime
    updated_at: datetime


class AgentMessageResponse(BaseModel):
    """A persisted message from an agent conversation."""

    model_config = ConfigDict(from_attributes=True)

    sequence_number: int
    role: AgentMessageRole
    content: list[dict[str, Any]]
    response_id: str | None
    provider: str | None
    model: str | None
    stop_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    created_at: datetime


class ConversationMessageResponse(AgentMessageResponse):
    """An original persisted message with conversation ordering and run provenance."""

    run_id: uuid.UUID
    turn_number: int


class ToolEventResponse(BaseModel):
    """A sanitized persisted tool invocation."""

    model_config = ConfigDict(from_attributes=True)

    sequence_number: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None
    is_error: bool
    duration_ms: float | None
    created_at: datetime


class PytestExecutionResponse(BaseModel):
    """Structured test evidence, independent of truncated conversational tool output."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tool_call_id: str
    command: list[str]
    targets: list[str]
    config_file: str | None
    status: str
    counts: dict[str, int]
    summary_complete: bool
    verified: bool
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    error_message: str | None
    duration_ms: float
    created_at: datetime


class AgentRunResponse(BaseModel):
    """Public representation of an agent run and its conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repository_id: uuid.UUID
    task: str
    mode: AgentRunMode
    status: AgentRunStatus
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    messages: list[AgentMessageResponse]
    tool_events: list[ToolEventResponse]
    test_results: list[PytestExecutionResponse]
