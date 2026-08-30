"""API request and response schemas."""

import uuid
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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
