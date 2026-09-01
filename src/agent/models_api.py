"""Provider-neutral model messages, tools, responses, and errors."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

type JsonObject = dict[str, Any]
type MessageRole = Literal["system", "user", "assistant"]
type StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "refusal",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class TextContent:
    """A text message block."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-requested tool invocation."""

    id: str
    name: str
    arguments: JsonObject


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The application-provided result of a tool invocation."""

    tool_call_id: str
    content: str
    is_error: bool = False


type MessageContent = TextContent | ToolCall | ToolResult
type ResponseContent = TextContent | ToolCall


@dataclass(frozen=True, slots=True)
class Message:
    """One provider-neutral conversation message."""

    role: MessageRole
    content: tuple[MessageContent, ...]

    @classmethod
    def text(cls, role: MessageRole, text: str) -> Message:
        """Create a message containing one text block."""
        return cls(role=role, content=(TextContent(text),))


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A JSON-schema-backed tool available to a model."""

    name: str
    description: str
    input_schema: JsonObject
    strict: bool = True


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """A normalized request sent to a model provider."""

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    max_output_tokens: int = 4096


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Normalized token usage reported by a provider."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        """Return input and output tokens combined."""
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """A normalized model response."""

    id: str
    provider: Literal["openai", "anthropic", "fake"]
    model: str
    content: tuple[ResponseContent, ...]
    stop_reason: StopReason
    usage: TokenUsage

    @property
    def text(self) -> str:
        """Join all returned text blocks."""
        return "".join(block.text for block in self.content if isinstance(block, TextContent))

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        """Return all tool calls in response order."""
        return tuple(block for block in self.content if isinstance(block, ToolCall))


class ModelClient(Protocol):
    """Interface implemented by every model provider."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate one normalized response."""
        ...


class ModelError(Exception):
    """Base error for model client failures."""


class ModelConfigurationError(ModelError):
    """Raised when a provider cannot be constructed from configuration."""


class ModelResponseError(ModelError):
    """Raised when a provider returns an invalid or unsupported response."""


class ModelRequestError(ModelError):
    """Raised when a normalized request violates the client contract."""


class ModelProviderError(ModelError):
    """A normalized provider SDK or API failure."""

    def __init__(
        self,
        provider: str,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
        self.status_code = status_code


def normalize_provider_error(provider: str, error: Exception) -> ModelProviderError:
    """Convert common SDK error attributes into one provider-neutral error."""
    status_code = getattr(error, "status_code", None)
    retryable = (
        error.__class__.__name__ in {"APIConnectionError", "APITimeoutError", "RateLimitError"}
        or status_code in {408, 409, 429}
        or isinstance(status_code, int)
        and status_code >= 500
    )
    return ModelProviderError(
        provider,
        str(error),
        retryable=retryable,
        status_code=status_code if isinstance(status_code, int) else None,
    )


def validate_model_request(request: ModelRequest) -> None:
    """Validate provider-neutral invariants before translating a request."""
    if not request.messages:
        raise ModelRequestError("At least one message is required")
    if request.max_output_tokens < 1:
        raise ModelRequestError("Maximum output tokens must be positive")
    tool_names: set[str] = set()
    for tool in request.tools:
        if not tool.name:
            raise ModelRequestError("Tool names must not be blank")
        if tool.name in tool_names:
            raise ModelRequestError(f"Duplicate tool name: {tool.name}")
        if tool.input_schema.get("type") != "object":
            raise ModelRequestError(f"Tool schema must describe an object: {tool.name}")
        tool_names.add(tool.name)

    for message in request.messages:
        if not message.content:
            raise ModelRequestError("Messages must contain at least one block")
        for block in message.content:
            if isinstance(block, ToolCall) and message.role != "assistant":
                raise ModelRequestError("Tool calls must appear in assistant messages")
            if isinstance(block, ToolResult) and message.role != "user":
                raise ModelRequestError("Tool results must appear in user messages")
            if message.role == "system" and not isinstance(block, TextContent):
                raise ModelRequestError("System messages may contain only text")


@dataclass(slots=True)
class FakeModelClient:
    """A deterministic queued-response model client for tests."""

    responses: list[ModelResponse]
    requests: list[ModelRequest] = field(default_factory=list)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the next queued response."""
        validate_model_request(request)
        self.requests.append(request)
        if not self.responses:
            raise ModelResponseError("Fake model client has no queued response")
        return self.responses.pop(0)
