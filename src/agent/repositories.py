"""Repository registration API."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent.database import get_database_session
from agent.models import Repository
from agent.schemas import RepositoryCreate, RepositoryResponse

router = APIRouter(prefix="/repositories", tags=["repositories"])
SessionDependency = Annotated[Session, Depends(get_database_session)]


@router.post("", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
def register_repository(payload: RepositoryCreate, session: SessionDependency) -> Repository:
    """Register an existing repository directory."""
    path = _resolve_directory(payload.path)
    repository = Repository(name=payload.name, path=str(path))
    session.add(repository)

    try:
        session.commit()
    except IntegrityError as error:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Repository path is already registered",
        ) from error

    session.refresh(repository)
    return repository


@router.get("", response_model=list[RepositoryResponse])
def list_repositories(
    session: SessionDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[Repository]:
    """List registered repositories in creation order."""
    statement = select(Repository).order_by(Repository.created_at, Repository.id)
    return list(session.scalars(statement.offset(offset).limit(limit)))


def _resolve_directory(path: Path) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository path does not exist",
        )
    if not resolved_path.is_dir():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Repository path is not a directory",
        )
    return resolved_path
