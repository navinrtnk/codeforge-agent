"""Tests for the Anthropic Messages API adapter."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agent.anthropic_client import AnthropicModelClient
from agent.models_api import (
    Message,
    ModelProviderError,
    ModelRequest,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def model_request() -> ModelRequest:
    return ModelRequest(
        messages=(
            Message.text("system", "You are a coding agent."),
            Message.text("user", "Inspect the file."),
            Message(
                "assistant",
                (
                    TextContent("Checking."),
                    ToolCall("toolu-1", "read_file", {"path": "README.md"}),
                ),
            ),
            Message("user", (ToolResult("toolu-1", "contents", is_error=False),)),
        ),
        tools=(
            ToolDefinition(
                "read_file",
                "Read a repository file",
                {"type": "object", "properties": {"path": {"type": "string"}}},
            ),
        ),
        max_output_tokens=512,
    )


def test_anthropic_adapter_translates_request_and_response() -> None:
    response = SimpleNamespace(
        id="msg-1",
        model="claude-test",
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=11, output_tokens=6),
        content=[
            SimpleNamespace(type="text", text="I will inspect it."),
            SimpleNamespace(
                type="tool_use",
                id="toolu-2",
                name="read_file",
                input={"path": "src/app.py"},
            ),
        ],
    )
    messages = FakeMessages(response)
    client = AnthropicModelClient(
        "test-key",
        "claude-test",
        client=SimpleNamespace(messages=messages),
    )

    result = asyncio.run(client.complete(model_request()))

    call = messages.calls[0]
    assert call["model"] == "claude-test"
    assert call["system"] == "You are a coding agent."
    assert call["max_tokens"] == 512
    assert call["tools"][0]["input_schema"]["type"] == "object"
    assert call["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "Inspect the file."}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "read_file",
                    "input": {"path": "README.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu-1",
                    "content": "contents",
                    "is_error": False,
                }
            ],
        },
    ]
    assert result.provider == "anthropic"
    assert result.text == "I will inspect it."
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].arguments == {"path": "src/app.py"}
    assert result.usage.total_tokens == 17


def test_anthropic_adapter_normalizes_nonretryable_provider_errors() -> None:
    authentication_error = type(
        "AuthenticationError",
        (Exception,),
        {"status_code": 401},
    )("invalid key")
    messages = FakeMessages(error=authentication_error)
    client = AnthropicModelClient(
        "test-key",
        "claude-test",
        client=SimpleNamespace(messages=messages),
    )

    with pytest.raises(ModelProviderError) as error_info:
        asyncio.run(client.complete(model_request()))

    assert error_info.value.provider == "anthropic"
    assert error_info.value.retryable is False
    assert error_info.value.status_code == 401
