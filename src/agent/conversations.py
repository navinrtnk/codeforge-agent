"""Create conversations, continue their tasks, and inspect full message history."""

import uuid
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import select

from agent.agent_runs import (
    ModelClientDependency,
    SessionDependency,
    _get_repository,
    build_agent_runner,
)
from agent.conversation_memory import ConversationBusyError, ConversationMemory
from agent.models import AgentMessage, AgentRun, Conversation, ConversationTurn
from agent.models_api import ModelError
from agent.repositories import RepositoryAccessDependency
from agent.schemas import (
    AgentMessageResponse,
    AgentRunResponse,
    ConversationCreate,
    ConversationMessageCreate,
    ConversationMessageResponse,
    ConversationResponse,
)

router = APIRouter(prefix="/repositories/{repository_id}/conversations", tags=["conversations"])


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
def create_conversation(
    repository_id: uuid.UUID,
    payload: ConversationCreate,
    session: SessionDependency,
) -> Conversation:
    """Create an empty repository conversation without contacting a model."""
    _get_repository(session, repository_id)
    conversation = Conversation(repository_id=repository_id, title=payload.title)
    session.add(conversation)
    session.commit()
    return conversation


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    repository_id: uuid.UUID,
    session: SessionDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[Conversation]:
    """List this repository's discussions in stable creation order."""
    _get_repository(session, repository_id)
    return list(
        session.scalars(
            select(Conversation)
            .where(Conversation.repository_id == repository_id)
            .order_by(Conversation.created_at, Conversation.id)
            .offset(offset)
            .limit(limit)
        )
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
def get_conversation(
    repository_id: uuid.UUID,
    conversation_id: uuid.UUID,
    session: SessionDependency,
) -> Conversation:
    """Inspect metadata and the latest persisted summary."""
    return _get_conversation(repository_id, conversation_id, session)


@router.get("/{conversation_id}/messages", response_model=list[ConversationMessageResponse])
def list_messages(
    repository_id: uuid.UUID,
    conversation_id: uuid.UUID,
    session: SessionDependency,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> list[ConversationMessageResponse]:
    """Return original messages, including failed turns, without trimming audit history."""
    _get_conversation(repository_id, conversation_id, session)
    rows = session.execute(
        select(AgentMessage, ConversationTurn.sequence_number)
        .join(ConversationTurn, ConversationTurn.run_id == AgentMessage.agent_run_id)
        .where(ConversationTurn.conversation_id == conversation_id, AgentMessage.role != "system")
        .order_by(ConversationTurn.sequence_number, AgentMessage.sequence_number)
        .offset(offset)
        .limit(limit)
    )
    return [
        ConversationMessageResponse(
            **{
                **AgentMessageResponse.model_validate(message).model_dump(),
                "run_id": message.agent_run_id,
                "turn_number": number,
            },
        )
        for message, number in rows
    ]


@router.post(
    "/{conversation_id}/messages",
    response_model=AgentRunResponse,
    status_code=status.HTTP_201_CREATED,
)
async def append_message(
    repository_id: uuid.UUID,
    conversation_id: uuid.UUID,
    payload: ConversationMessageCreate,
    request: Request,
    session: SessionDependency,
    access: RepositoryAccessDependency,
    model_client: ModelClientDependency,
) -> AgentRun:
    """Execute a follow-up with recent context and a summary of older dialogue."""
    conversation = _get_conversation(repository_id, conversation_id, session)
    repository = _get_repository(session, repository_id)
    settings = request.app.state.settings
    memory = ConversationMemory(
        model_client,
        history_token_budget=settings.conversation_history_token_budget,
        summary_max_tokens=settings.conversation_summary_max_tokens,
        summary_input_token_budget=settings.conversation_summary_input_token_budget,
        timeout_seconds=settings.agent_timeout_seconds,
    )
    try:
        return await memory.append(
            session,
            conversation,
            repository,
            payload.content,
            build_agent_runner(request, access, model_client, mode=payload.mode),
        )
    except ConversationBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except TimeoutError as error:
        raise HTTPException(status_code=504, detail="Conversation summary timed out") from error
    except ModelError as error:
        raise HTTPException(
            status_code=502, detail="Conversation summary generation failed"
        ) from error


def _get_conversation(
    repository_id: uuid.UUID,
    conversation_id: uuid.UUID,
    session: SessionDependency,
) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.repository_id != repository_id:
        raise HTTPException(status_code=404, detail="Conversation does not exist")
    return conversation
