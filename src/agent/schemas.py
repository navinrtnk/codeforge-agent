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
