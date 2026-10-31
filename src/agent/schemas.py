"""API request and response schemas."""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from agent.models import AgentMessageRole, AgentRunMode, AgentRunStatus, AgentRunType


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


class AgentTaskCreate(BaseModel):
    """A nonblank repository task."""

    task: str = Field(min_length=1, max_length=20_000)

    @field_validator("task")
    @classmethod
    def reject_blank_task(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Agent task must not be blank")
        return value


class AgentRunCreate(AgentTaskCreate):
    """A task to execute against a registered repository."""

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


class AnalysisExecutionResponse(BaseModel):
    """Persisted structured static analysis evidence."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    tool_call_id: str
    tool_name: str
    result: dict[str, Any]
    created_at: datetime


class AgentRunResponse(BaseModel):
    """Public representation of an agent run and its conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    repository_id: uuid.UUID
    task: str
    mode: AgentRunMode
    run_type: AgentRunType
    repair_attempts: int
    verified: bool
    status: AgentRunStatus
    error_message: str | None
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    messages: list[AgentMessageResponse]
    tool_events: list[ToolEventResponse]
    test_results: list[PytestExecutionResponse]
    analysis_results: list[AnalysisExecutionResponse]


class ReviewCreate(BaseModel):
    """Select unstaged changes, staged changes, or committed changes since a base ref."""

    model_config = ConfigDict(extra="forbid")

    staged: bool = False
    base_ref: str | None = Field(default=None, min_length=1, max_length=255)

    @model_validator(mode="after")
    def validate_selection(self) -> ReviewCreate:
        if self.staged and self.base_ref is not None:
            raise ValueError("Select staged changes or a base ref, not both")
        if self.base_ref is not None and (
            self.base_ref.startswith("-")
            or any(character.isspace() or character == "\0" for character in self.base_ref)
        ):
            raise ValueError("Invalid base ref")
        return self


class ReviewFinding(BaseModel):
    """One actionable issue anchored to a changed line in the old or new file."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    severity: Literal["critical", "high", "medium", "low"]
    file: Annotated[str, StringConstraints(strip_whitespace=False)] = Field(
        min_length=1, max_length=1024
    )
    line: int = Field(ge=1)
    side: Literal["old", "new"]
    explanation: str = Field(min_length=1, max_length=4000)
    suggestion: str = Field(min_length=1, max_length=4000)


class ReviewReport(BaseModel):
    """Structured model output validated before returning a review."""

    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)

    summary: str = Field(min_length=1, max_length=4000)
    risk_level: Literal["low", "medium", "high", "unknown"]
    risk_assessment: str = Field(min_length=1, max_length=4000)
    findings: list[ReviewFinding] = Field(max_length=100)


class ReviewResponse(ReviewReport):
    """Review results plus explicit coverage and validation information."""

    reviewed_files: list[str]
    skipped_files: list[str]
    discarded_findings: int
