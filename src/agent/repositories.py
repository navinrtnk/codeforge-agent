"""Repository registration API."""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from agent.database import get_database_session
from agent.indexing import IndexingResult, IndexStatus, RepositoryIndexer, get_index_status
from agent.models import Repository
from agent.repository_access import (
    PathOutsideWorkspaceError,
    RepositoryAccess,
    RepositoryNotFoundError,
    RepositoryPathError,
)
from agent.schemas import (
    IndexingResponse,
    IndexStatusResponse,
    RepositoryCreate,
    RepositoryResponse,
    SearchResultResponse,
    SymbolSearchResultResponse,
)
from agent.search import (
    InvalidSearchQueryError,
    SearchResult,
    SymbolSearchResult,
    exact_search,
    lexical_search,
    symbol_search,
)

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


@router.post("/{repository_id}/index", response_model=IndexingResponse)
def index_repository(
    repository_id: uuid.UUID,
    session: SessionDependency,
    access: RepositoryAccessDependency,
    request: Request,
) -> IndexingResult:
    """Incrementally index the current repository contents."""
    repository = _get_repository(session, repository_id)
    indexer = RepositoryIndexer(
        access,
        chunk_size_lines=request.app.state.settings.index_chunk_size_lines,
    )
    return indexer.index(session, repository)


@router.get("/{repository_id}/index/status", response_model=IndexStatusResponse)
def repository_index_status(
    repository_id: uuid.UUID,
    session: SessionDependency,
) -> IndexStatus:
    """Return current index file and chunk counts."""
    _get_repository(session, repository_id)
    return get_index_status(session, repository_id)


@router.get("/{repository_id}/search", response_model=list[SearchResultResponse])
def search_repository(
    repository_id: uuid.UUID,
    session: SessionDependency,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    mode: Annotated[Literal["lexical", "exact"], Query()] = "lexical",
    case_sensitive: Annotated[bool, Query()] = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[SearchResult]:
    """Search indexed repository chunks using lexical or exact matching."""
    _get_repository(session, repository_id)
    try:
        if mode == "exact":
            return exact_search(
                session,
                repository_id,
                q,
                case_sensitive=case_sensitive,
                limit=limit,
            )
        return lexical_search(session, repository_id, q, limit=limit)
    except InvalidSearchQueryError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


@router.get("/{repository_id}/symbols", response_model=list[SymbolSearchResultResponse])
def search_repository_symbols(
    repository_id: uuid.UUID,
    session: SessionDependency,
    q: Annotated[str, Query(min_length=1, max_length=500)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list[SymbolSearchResult]:
    """Search persisted Python symbols by name or qualified name."""
    _get_repository(session, repository_id)
    try:
        return symbol_search(session, repository_id, q, limit=limit)
    except InvalidSearchQueryError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error


def _get_repository(session: Session, repository_id: uuid.UUID) -> Repository:
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository is not registered",
        )
    return repository
