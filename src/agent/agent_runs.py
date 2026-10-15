"""Agent run execution and conversation inspection API."""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from agent.agent_runner import AgentRunner
from agent.database import get_database_session
from agent.git_service import GitService
from agent.git_tools import create_agent_tool_registry
from agent.model_factory import create_model_client
from agent.models import AgentRun, Repository
from agent.models_api import ModelClient, ModelConfigurationError
from agent.repositories import RepositoryAccessDependency
from agent.repository_access import RepositoryAccess
from agent.schemas import AgentRunCreate, AgentRunResponse

router = APIRouter(prefix="/repositories/{repository_id}/runs", tags=["agent runs"])
SessionDependency = Annotated[Session, Depends(get_database_session)]


def get_model_client(request: Request) -> ModelClient:
    """Return an injected client or lazily create the configured provider."""
    client: ModelClient | None = request.app.state.model_client
    if client is None:
        try:
            client = create_model_client(request.app.state.settings)
        except ModelConfigurationError as error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(error),
            ) from error
        request.app.state.model_client = client
    return client


ModelClientDependency = Annotated[ModelClient, Depends(get_model_client)]


@router.post("", response_model=AgentRunResponse, status_code=status.HTTP_201_CREATED)
async def create_agent_run(
    repository_id: uuid.UUID,
    payload: AgentRunCreate,
    request: Request,
    session: SessionDependency,
    access: RepositoryAccessDependency,
    model_client: ModelClientDependency,
) -> AgentRun:
    """Execute one bounded agent task against a registered repository."""
    repository = _get_repository(session, repository_id)
    runner = build_agent_runner(request, access, model_client)
    return await runner.run(session, repository, payload.task)


def build_agent_runner(
    request: Request,
    access: RepositoryAccess,
    model_client: ModelClient,
) -> AgentRunner:
    """Build the same configured runner for standalone and conversation tasks."""
    settings = request.app.state.settings
    git_service: GitService | None = request.app.state.git_service
    if git_service is None:
        git_service = GitService(
            access,
            allow_mutations=settings.git_allow_mutations,
            timeout_seconds=settings.git_timeout_seconds,
            max_output_bytes=settings.git_max_output_bytes,
        )
        request.app.state.git_service = git_service
    return AgentRunner(
        model_client,
        create_agent_tool_registry(git_service),
        access,
        max_iterations=settings.agent_max_iterations,
        max_output_tokens=settings.agent_max_output_tokens,
        timeout_seconds=settings.agent_timeout_seconds,
        max_tool_output_bytes=settings.max_tool_output_bytes,
        git_service=git_service,
        git_context_max_bytes=settings.git_context_max_bytes,
        git_context_max_files=settings.git_context_max_files,
        git_context_timeout_seconds=settings.git_context_timeout_seconds,
    )


@router.get("", response_model=list[AgentRunResponse])
def list_agent_runs(
    repository_id: uuid.UUID,
    session: SessionDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[AgentRun]:
    """List agent runs for one repository, newest first."""
    _get_repository(session, repository_id)
    statement = (
        select(AgentRun)
        .where(AgentRun.repository_id == repository_id)
        .order_by(AgentRun.created_at.desc(), AgentRun.id.desc())
        .offset(offset)
        .limit(limit)
    )
    return list(session.scalars(statement))


@router.get("/{run_id}", response_model=AgentRunResponse)
def get_agent_run(
    repository_id: uuid.UUID,
    run_id: uuid.UUID,
    session: SessionDependency,
) -> AgentRun:
    """Return a run with its persisted messages and tool events."""
    run = session.get(AgentRun, run_id)
    if run is None or run.repository_id != repository_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent run does not exist",
        )
    return run


def _get_repository(session: Session, repository_id: uuid.UUID) -> Repository:
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Repository is not registered",
        )
    return repository
