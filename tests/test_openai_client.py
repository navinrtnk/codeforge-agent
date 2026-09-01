"""Tests for the OpenAI Responses API adapter."""

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from agent.models_api import (
    Message,
    ModelProviderError,
    ModelRequest,
    ModelResponseError,
    TextContent,
    ToolCall,
    ToolDefinition,
    ToolResult,
)
from agent.openai_client import OpenAIModelClient


class FakeResponses:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def openai_response(*, arguments: str = '{"path":"src/app.py"}') -> Any:
    return SimpleNamespace(
        id="resp-1",
        model="gpt-test",
        status="completed",
        error=None,
        incomplete_details=None,
        usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="output_text", text="I will inspect it.")],
            ),
            SimpleNamespace(
                type="function_call",
                call_id="call-2",
                name="read_file",
                arguments=arguments,
            ),
        ],
    )


def model_request() -> ModelRequest:
    return ModelRequest(
        messages=(
            Message.text("system", "You are a coding agent."),
            Message.text("user", "Inspect the file."),
            Message(
                "assistant",
                (
                    TextContent("Checking."),
                    ToolCall("call-1", "read_file", {"path": "README.md"}),
                ),
            ),
            Message("user", (ToolResult("call-1", "contents"),)),
        ),
        tools=(
            ToolDefinition(
                name="read_file",
                description="Read a repository file",
                input_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
        ),
        max_output_tokens=512,
    )


def test_openai_adapter_translates_request_and_response() -> None:
    responses = FakeResponses(openai_response())
    client = OpenAIModelClient(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=responses),
    )

    result = asyncio.run(client.complete(model_request()))

    call = responses.calls[0]
    assert call["model"] == "gpt-test"
    assert call["instructions"] == "You are a coding agent."
    assert call["max_output_tokens"] == 512
    assert call["store"] is False
    assert call["tools"][0]["parameters"]["required"] == ["path"]
    assert call["input"] == [
        {"role": "user", "content": "Inspect the file."},
        {"role": "assistant", "content": "Checking."},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": '{"path": "README.md"}',
        },
        {"type": "function_call_output", "call_id": "call-1", "output": "contents"},
    ]
    assert result.provider == "openai"
    assert result.text == "I will inspect it."
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].arguments == {"path": "src/app.py"}
    assert result.usage.total_tokens == 19


def test_openai_adapter_rejects_malformed_tool_arguments() -> None:
    responses = FakeResponses(openai_response(arguments="not-json"))
    client = OpenAIModelClient(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(ModelResponseError, match="malformed tool arguments"):
        asyncio.run(client.complete(model_request()))


def test_openai_adapter_normalizes_retryable_provider_errors() -> None:
    rate_limit_error = type("RateLimitError", (Exception,), {"status_code": 429})("slow down")
    responses = FakeResponses(error=rate_limit_error)
    client = OpenAIModelClient(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(ModelProviderError) as error_info:
        asyncio.run(client.complete(model_request()))

    assert error_info.value.provider == "openai"
    assert error_info.value.retryable is True
    assert error_info.value.status_code == 429


def test_openai_adapter_normalizes_incomplete_response() -> None:
    response = openai_response()
    response.status = "incomplete"
    response.incomplete_details = SimpleNamespace(reason="max_output_tokens")
    responses = FakeResponses(response)
    client = OpenAIModelClient(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=responses),
    )

    result = asyncio.run(client.complete(model_request()))

    assert result.stop_reason == "max_tokens"


def test_openai_adapter_rejects_failed_response() -> None:
    response = openai_response()
    response.status = "failed"
    response.error = SimpleNamespace(message="provider failed")
    responses = FakeResponses(response)
    client = OpenAIModelClient(
        "test-key",
        "gpt-test",
        client=SimpleNamespace(responses=responses),
    )

    with pytest.raises(ModelResponseError, match="provider failed"):
        asyncio.run(client.complete(model_request()))
