"""OpenAI Responses API adapter."""

import json
from typing import Any

import openai

from agent.models_api import (
    Message,
    ModelRequest,
    ModelResponse,
    ModelResponseError,
    StopReason,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
    normalize_provider_error,
    validate_model_request,
)


class OpenAIModelClient:
    """Translate normalized model requests to the OpenAI Responses API."""

    def __init__(self, api_key: str, model: str, *, client: Any | None = None) -> None:
        self.model = model
        self._client = client or openai.AsyncOpenAI(api_key=api_key)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Generate and normalize one OpenAI response."""
        validate_model_request(request)
        instructions, input_items = _openai_input(request.messages)
        arguments: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "tools": [_openai_tool(tool) for tool in request.tools],
            "max_output_tokens": request.max_output_tokens,
            "store": False,
        }
        if instructions:
            arguments["instructions"] = instructions

        try:
            response = await self._client.responses.create(**arguments)
        except Exception as error:
            raise normalize_provider_error("openai", error) from error
        return _parse_openai_response(response)


def _openai_input(messages: tuple[Message, ...]) -> tuple[str, list[dict[str, Any]]]:
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextContent):
                if message.role == "system":
                    instructions.append(block.text)
                else:
                    items.append({"role": message.role, "content": block.text})
            elif isinstance(block, ToolCall):
                items.append(
                    {
                        "type": "function_call",
                        "call_id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    }
                )
            elif isinstance(block, ToolResult):
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": block.tool_call_id,
                        "output": block.content,
                    }
                )
    return "\n\n".join(instructions), items


def _openai_tool(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.input_schema,
        "strict": tool.strict,
    }


def _parse_openai_response(response: Any) -> ModelResponse:
    if response.status == "failed":
        message = response.error.message if response.error is not None else "unknown error"
        raise ModelResponseError(f"OpenAI response failed: {message}")

    content: list[TextContent | ToolCall] = []
    for item in response.output:
        if item.type == "message":
            for block in item.content:
                if block.type == "output_text":
                    content.append(TextContent(block.text))
        elif item.type == "function_call":
            try:
                arguments = json.loads(item.arguments)
            except (TypeError, json.JSONDecodeError) as error:
                raise ModelResponseError("OpenAI returned malformed tool arguments") from error
            if not isinstance(arguments, dict):
                raise ModelResponseError("OpenAI tool arguments must be a JSON object")
            content.append(ToolCall(id=item.call_id, name=item.name, arguments=arguments))

    stop_reason: StopReason = (
        "tool_use" if any(isinstance(block, ToolCall) for block in content) else "end_turn"
    )
    if response.status == "incomplete":
        incomplete_reason = (
            response.incomplete_details.reason if response.incomplete_details is not None else None
        )
        stop_reason = "max_tokens" if incomplete_reason == "max_output_tokens" else "unknown"
    usage = response.usage
    return ModelResponse(
        id=response.id,
        provider="openai",
        model=response.model,
        content=tuple(content),
        stop_reason=stop_reason,
        usage=TokenUsage(
            input_tokens=usage.input_tokens if usage is not None else 0,
            output_tokens=usage.output_tokens if usage is not None else 0,
        ),
    )
