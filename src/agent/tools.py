"""Constrained tool registration, execution, persistence, and redaction."""

import json
import re
import uuid
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from agent.models import AgentRun, Repository, ToolEvent
from agent.models_api import JsonObject, ToolCall, ToolDefinition, ToolResult
from agent.repository_access import RepositoryAccess

REDACTED = "[REDACTED]"
SENSITIVE_KEYS = ("api_key", "apikey", "authorization", "cookie", "password", "secret", "token")
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"(?i)\b(api[_-]?key|password|secret|token)\s*([:=])\s*([^\s,;]+)"),
)


class ToolExecutionError(Exception):
    """A safe error that may be returned to the model."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Trusted resources supplied by the application, never by the model."""

    repository: Repository
    session: Session
    access: RepositoryAccess


class Tool(Protocol):
    """Interface implemented by registered application tools."""

    name: str
    description: str

    @property
    def arguments_model(self) -> type[BaseModel]:
        """Return the Pydantic model used to validate arguments."""
        ...

    async def execute(self, context: ToolContext, arguments: BaseModel) -> JsonObject:
        """Execute with already validated arguments."""
        ...


class ToolRegistry:
    """An explicit allowlist of tools available to models."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a uniquely named tool."""
        if not tool.name:
            raise ValueError("Tool name must not be blank")
        if tool.name in self._tools:
            raise ValueError(f"Tool is already registered: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Return a tool only when it was explicitly registered."""
        return self._tools.get(name)

    def definitions(self) -> tuple[ToolDefinition, ...]:
        """Return model-facing definitions in stable name order."""
        return tuple(
            ToolDefinition(
                name=tool.name,
                description=tool.description,
                input_schema=tool.arguments_model.model_json_schema(),
                strict=True,
            )
            for tool in sorted(self._tools.values(), key=lambda item: item.name)
        )


class ToolExecutor:
    """Validate, execute, redact, bound, and persist one tool call."""

    def __init__(self, registry: ToolRegistry, max_output_bytes: int = 100_000) -> None:
        if max_output_bytes < 256:
            raise ValueError("Maximum tool output size must be at least 256 bytes")
        self.registry = registry
        self.max_output_bytes = max_output_bytes

    async def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        agent_run_id: uuid.UUID,
        sequence_number: int,
    ) -> ToolResult:
        """Execute one requested tool and record a sanitized event."""
        if sequence_number < 1:
            raise ValueError("Tool sequence number must be positive")
        run = context.session.get(AgentRun, agent_run_id)
        if run is None:
            raise ValueError("Agent run does not exist")
        if run.repository_id != context.repository.id:
            raise ValueError("Agent run does not belong to the active repository")

        started_at = perf_counter()
        tool = self.registry.get(call.name)
        is_error = False

        if tool is None:
            is_error = True
            result: JsonObject = {"error": f"Unknown tool: {call.name}"}
        else:
            try:
                arguments = tool.arguments_model.model_validate(call.arguments)
                result = await tool.execute(context, arguments)
            except ValidationError as error:
                is_error = True
                result = {"error": _validation_message(error)}
            except ToolExecutionError as error:
                is_error = True
                result = {"error": str(error)}
            except Exception:
                is_error = True
                result = {"error": "Tool execution failed"}

        duration_ms = (perf_counter() - started_at) * 1000
        sanitized_arguments = redact_json_object(call.arguments)
        sanitized_result = redact_json_object(result)
        bounded_result = bound_json_object(sanitized_result, self.max_output_bytes)
        context.session.add(
            ToolEvent(
                agent_run_id=agent_run_id,
                sequence_number=sequence_number,
                tool_call_id=call.id,
                tool_name=call.name,
                arguments=sanitized_arguments,
                result=bounded_result,
                is_error=is_error,
                duration_ms=duration_ms,
            )
        )
        context.session.commit()
        return ToolResult(
            tool_call_id=call.id,
            content=serialize_json(bounded_result),
            is_error=is_error,
        )


def redact_json_object(value: JsonObject) -> JsonObject:
    """Recursively redact secrets while preserving the JSON object shape."""
    redacted = _redact(value)
    if not isinstance(redacted, dict):
        raise TypeError("Expected a JSON object")
    return redacted


def bound_json_object(value: JsonObject, max_bytes: int) -> JsonObject:
    """Replace oversized JSON with a bounded preview envelope."""
    serialized = serialize_json(value)
    encoded = serialized.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    preview_budget = max(0, max_bytes - 100)
    preview = encoded[:preview_budget].decode("utf-8", errors="ignore")
    envelope: JsonObject = {
        "truncated": True,
        "original_bytes": len(encoded),
        "preview": preview,
    }
    while len(serialize_json(envelope).encode("utf-8")) > max_bytes and envelope["preview"]:
        envelope["preview"] = str(envelope["preview"])[:-1]
    return envelope


def serialize_json(value: JsonObject) -> str:
    """Serialize tool data deterministically for model consumption."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _redact(value: Any, key: str | None = None) -> Any:
    if key is not None and any(sensitive in key.lower() for sensitive in SENSITIVE_KEYS):
        return REDACTED
    if isinstance(value, dict):
        return {item_key: _redact(item_value, item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for pattern in SECRET_PATTERNS:
            if pattern.groups == 3:
                redacted = pattern.sub(r"\1\2" + REDACTED, redacted)
            else:
                redacted = pattern.sub(REDACTED, redacted)
        return redacted
    return value


def _validation_message(error: ValidationError) -> str:
    details = []
    for item in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in item["loc"]) or "arguments"
        details.append(f"{location}: {item['msg']}")
    return "Invalid arguments: " + "; ".join(details)
