"""Read-only code review API for local Git changes."""

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from agent.agent_runs import ModelClientDependency, SessionDependency, get_git_service
from agent.git_service import GitError
from agent.models import Repository
from agent.models_api import ModelError
from agent.repositories import RepositoryAccessDependency
from agent.repository_access import RepositoryAccessError
from agent.review_service import ReviewService
from agent.schemas import ReviewCreate, ReviewResponse

router = APIRouter(prefix="/repositories/{repository_id}/reviews", tags=["code reviews"])


@router.post("", response_model=ReviewResponse)
async def create_review(
    repository_id: uuid.UUID,
    payload: ReviewCreate,
    request: Request,
    session: SessionDependency,
    access: RepositoryAccessDependency,
    model_client: ModelClientDependency,
) -> ReviewResponse:
    """Review selected Git changes without editing files or publishing comments."""
    repository = session.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository is not registered")
    settings = request.app.state.settings
    service = ReviewService(
        get_git_service(request, access),
        model_client,
        max_files=settings.review_max_files,
        max_context_bytes=settings.review_max_context_bytes,
        max_output_tokens=settings.agent_max_output_tokens,
    )
    try:
        async with asyncio.timeout(settings.agent_timeout_seconds):
            return await service.review(Path(repository.path), payload)
    except RepositoryAccessError as error:
        raise HTTPException(status_code=403, detail="Repository is not accessible") from error
    except GitError as error:
        raise HTTPException(
            status_code=422, detail="Unable to collect the selected review diff"
        ) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="Code review timed out") from error
    except ModelError as error:
        raise HTTPException(
            status_code=502, detail="Unable to generate a valid code review"
        ) from error
