"""API request and response schemas."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agent.models import AgentMessageRole, AgentRunStatus


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


class AgentRunResponse(BaseModel):
    """Public representation of an agent run and its conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repository_id: uuid.UUID
    task: str
    status: AgentRunStatus
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    messages: list[AgentMessageResponse]
    tool_events: list[ToolEventResponse]
