"""Bounded model and tool orchestration for one repository task."""

import asyncio
from datetime import UTC, datetime
from typing import cast

from sqlalchemy.orm import Session

from agent.models import (
    AgentMessage,
    AgentMessageRole,
    AgentRun,
    AgentRunStatus,
    Repository,
)
from agent.models_api import (
    Message,
    MessageContent,
    ModelClient,
    ModelRequest,
    ModelResponse,
    TextContent,
    ToolCall,
    ToolResult,
)
from agent.repository_access import RepositoryAccess
from agent.tools import ToolContext, ToolExecutor, ToolRegistry, redact_json_object

SYSTEM_PROMPT = """You are CodeForge Agent, a repository analysis assistant.
Use only the provided read-only tools to inspect the active repository.
Never claim to have read code that was not returned by a tool.
Give a concise final answer grounded in repository evidence."""


class AgentExecutionLimitError(Exception):
    """Raised internally when a run exhausts its model-turn budget."""


class AgentRunner:
    """Execute and persist a bounded model-tool conversation."""

    def __init__(
        self,
        model_client: ModelClient,
        registry: ToolRegistry,
        access: RepositoryAccess,
        *,
        max_iterations: int = 8,
        max_output_tokens: int = 4096,
        timeout_seconds: float = 120.0,
        max_tool_output_bytes: int = 100_000,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("Maximum agent iterations must be positive")
        if max_output_tokens < 1:
            raise ValueError("Maximum model output tokens must be positive")
        if timeout_seconds <= 0:
            raise ValueError("Agent timeout must be positive")
        self.model_client = model_client
        self.registry = registry
        self.access = access
        self.max_iterations = max_iterations
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self.tool_executor = ToolExecutor(
            registry,
            max_output_bytes=max_tool_output_bytes,
        )

    async def run(self, session: Session, repository: Repository, task: str) -> AgentRun:
        """Create and execute an agent run, persisting success or failure."""
        if not task.strip():
            raise ValueError("Agent task must not be blank")

        run = AgentRun(
            repository=repository,
            task=task,
            status=AgentRunStatus.RUNNING,
            started_at=datetime.now(UTC),
        )
        session.add(run)
        session.commit()

        messages = [Message.text("system", SYSTEM_PROMPT), Message.text("user", task)]
        next_message_sequence = 1
        for message in messages:
            self._persist_message(session, run, next_message_sequence, message)
            next_message_sequence += 1

        try:
            async with asyncio.timeout(self.timeout_seconds):
                await self._execute_loop(
                    session,
                    repository,
                    run,
                    messages,
                    next_message_sequence,
                )
        except TimeoutError:
            self._fail_run(session, run, "Agent run timed out")
        except Exception as error:
            sanitized = redact_json_object({"message": str(error)})["message"]
            self._fail_run(session, run, cast(str, sanitized))
        return run

    async def _execute_loop(
        self,
        session: Session,
        repository: Repository,
        run: AgentRun,
        messages: list[Message],
        next_message_sequence: int,
    ) -> None:
        tool_sequence = 1
        context = ToolContext(repository=repository, session=session, access=self.access)

        for _ in range(self.max_iterations):
            response = await self.model_client.complete(
                ModelRequest(
                    messages=tuple(messages),
                    tools=self.registry.definitions(),
                    max_output_tokens=self.max_output_tokens,
                )
            )
            assistant = Message(role="assistant", content=response.content)
            messages.append(assistant)
            self._persist_message(
                session,
                run,
                next_message_sequence,
                assistant,
                response=response,
            )
            next_message_sequence += 1

            if not response.tool_calls:
                if not response.text.strip():
                    raise ValueError("Model returned no final answer")
                run.status = AgentRunStatus.COMPLETED
                run.completed_at = datetime.now(UTC)
                session.commit()
                return

            results: list[ToolResult] = []
            for call in response.tool_calls:
                result = await self.tool_executor.execute(
                    call,
                    context,
                    agent_run_id=run.id,
                    sequence_number=tool_sequence,
                )
                results.append(result)
                tool_sequence += 1

            tool_message = Message(role="user", content=tuple(results))
            messages.append(tool_message)
            self._persist_message(
                session,
                run,
                next_message_sequence,
                tool_message,
            )
            next_message_sequence += 1

        raise AgentExecutionLimitError(
            f"Agent exceeded the maximum of {self.max_iterations} model iterations"
        )

    @staticmethod
    def _persist_message(
        session: Session,
        run: AgentRun,
        sequence_number: int,
        message: Message,
        *,
        response: ModelResponse | None = None,
    ) -> None:
        stored = AgentMessage(
            run=run,
            sequence_number=sequence_number,
            role=AgentMessageRole(message.role),
            content=_serialize_content(message.content),
            response_id=response.id if response else None,
            provider=response.provider if response else None,
            model=response.model if response else None,
            stop_reason=response.stop_reason if response else None,
            input_tokens=response.usage.input_tokens if response else None,
            output_tokens=response.usage.output_tokens if response else None,
        )
        session.add(stored)
        session.commit()

    @staticmethod
    def _fail_run(session: Session, run: AgentRun, message: str) -> None:
        session.rollback()
        run.status = AgentRunStatus.FAILED
        run.error_message = message
        run.completed_at = datetime.now(UTC)
        session.commit()


def _serialize_content(content: tuple[MessageContent, ...]) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for block in content:
        if isinstance(block, TextContent):
            item = redact_json_object({"type": "text", "text": block.text})
        elif isinstance(block, ToolCall):
            item = {
                "type": "tool_call",
                "id": block.id,
                "name": block.name,
                "arguments": redact_json_object(block.arguments),
            }
        else:
            item = redact_json_object(
                {
                    "type": "tool_result",
                    "tool_call_id": block.tool_call_id,
                    "content": block.content,
                    "is_error": block.is_error,
                }
            )
        serialized.append(item)
    return serialized
