"""Anthropic Messages API adapter."""

from typing import Any

import anthropic

from agent.models_api import (
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
    normalize_provider_error,
    validate_model_request,
)


class AnthropicModelClient:
    """Translate normalized model requests to the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str, *, client: Any | None = None) -> None:
        self.model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate and normalize one Anthropic response."""
        validate_model_request(request)
        system, messages = _anthropic_messages(request.messages)
        arguments: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": [_anthropic_tool(tool) for tool in request.tools],
            "max_tokens": request.max_output_tokens,
        }
        if system:
            arguments["system"] = system

        try:
            response = await self._client.messages.create(**arguments)
        except Exception as error:
            raise normalize_provider_error("anthropic", error) from error
        return _parse_anthropic_response(response)


def _anthropic_messages(messages: tuple[Message, ...]) -> tuple[str, list[dict[str, Any]]]:
    system: list[str] = []
    translated: list[dict[str, Any]] = []
    for message in messages:
        blocks: list[dict[str, Any]] = []
        for block in message.content:
            if isinstance(block, TextContent):
                if message.role == "system":
                    system.append(block.text)
                else:
                    blocks.append({"type": "text", "text": block.text})
            elif isinstance(block, ToolCall):
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.arguments,
                    }
                )
            elif isinstance(block, ToolResult):
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.tool_call_id,
                        "content": block.content,
                        "is_error": block.is_error,
                    }
                )
        if blocks:
            translated.append({"role": message.role, "content": blocks})
    return "\n\n".join(system), translated


def _anthropic_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "strict": tool.strict,
    }


def _parse_anthropic_response(response: Any) -> ModelResponse:
    content: list[TextContent | ToolCall] = []
    for block in response.content:
        if block.type == "text":
            content.append(TextContent(block.text))
        elif block.type == "tool_use":
            content.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

    stop_reasons: dict[str, StopReason] = {
        "end_turn": "end_turn",
        "tool_use": "tool_use",
        "max_tokens": "max_tokens",
        "stop_sequence": "stop_sequence",
        "refusal": "refusal",
    }
    return ModelResponse(
        id=response.id,
        provider="anthropic",
        model=response.model,
        content=tuple(content),
        stop_reason=stop_reasons.get(response.stop_reason, "unknown"),
        usage=TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        ),
    )
