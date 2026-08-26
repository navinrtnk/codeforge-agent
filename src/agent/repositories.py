"""Repository registration API."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent.database import get_database_session
from agent.models import Repository
from agent.repository_access import (
    PathOutsideWorkspaceError,
    RepositoryAccess,
    RepositoryNotFoundError,
    RepositoryPathError,
)
from agent.schemas import RepositoryCreate, RepositoryResponse

router = APIRouter(prefix="/repositories", tags=["repositories"])
SessionDependency = Annotated[Session, Depends(get_database_session)]


def get_repository_access(request: Request) -> RepositoryAccess:
    """Build repository access from the active application settings."""
    settings = request.app.state.settings
    return RepositoryAccess(
        workspace_root=settings.workspace_root,
        ignore_patterns=settings.repository_ignore_patterns,
        max_file_size_bytes=settings.max_file_size_bytes,
    )


RepositoryAccessDependency = Annotated[RepositoryAccess, Depends(get_repository_access)]


@router.post("", response_model=RepositoryResponse, status_code=status.HTTP_201_CREATED)
def register_repository(
    payload: RepositoryCreate,
    session: SessionDependency,
    access: RepositoryAccessDependency,
) -> Repository:
    """Register an existing repository directory."""
    try:
        path = access.resolve_repository(payload.path)
    except PathOutsideWorkspaceError as error:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(error),
        ) from error
    except RepositoryNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(error),
        ) from error
    except RepositoryPathError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error
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
