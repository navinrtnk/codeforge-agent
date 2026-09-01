"""Contract tests for provider-neutral model clients."""

import asyncio

import pytest

from agent.models_api import (
    FakeModelClient,
    Message,
    ModelClient,
    ModelRequest,
    ModelRequestError,
    ModelResponse,
    ModelResponseError,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


def complete(client: ModelClient, request: ModelRequest) -> ModelResponse:
    return asyncio.run(client.complete(request))


def test_fake_client_returns_queued_response_and_records_request() -> None:
    response = ModelResponse(
        id="fake-1",
        provider="fake",
        model="fake-model",
        content=(TextContent("done"),),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=4, output_tokens=2),
    )
    client = FakeModelClient([response])
    request = ModelRequest(messages=(Message.text("user", "hello"),))

    result = complete(client, request)

    assert result == response
    assert result.text == "done"
    assert result.tool_calls == ()
    assert result.usage.total_tokens == 6
    assert client.requests == [request]


def test_fake_client_fails_when_response_queue_is_empty() -> None:
    client = FakeModelClient([])

    with pytest.raises(ModelResponseError, match="no queued response"):
        complete(client, ModelRequest(messages=(Message.text("user", "hello"),)))


@pytest.mark.parametrize(
    "invalid_request",
    [
        ModelRequest(messages=()),
        ModelRequest(messages=(Message.text("user", "hello"),), max_output_tokens=0),
        ModelRequest(messages=(Message(role="user", content=()),)),
        ModelRequest(messages=(Message("user", (ToolCall("call-1", "read_file", {}),)),)),
        ModelRequest(messages=(Message("assistant", (ToolResult("call-1", "result"),)),)),
        ModelRequest(
            messages=(Message.text("user", "hello"),),
            tools=(ToolDefinition("bad", "bad schema", {"type": "string"}),),
        ),
    ],
)
def test_all_clients_reject_invalid_normalized_requests(invalid_request: ModelRequest) -> None:
    with pytest.raises(ModelRequestError):
        complete(FakeModelClient([]), invalid_request)


def test_response_helpers_preserve_multiple_tool_calls() -> None:
    response = ModelResponse(
        id="fake-2",
        provider="fake",
        model="fake-model",
        content=(
            TextContent("checking"),
            ToolCall("call-1", "read_file", {"path": "a.py"}),
            ToolCall("call-2", "read_file", {"path": "b.py"}),
        ),
        stop_reason="tool_use",
        usage=TokenUsage(1, 1),
    )

    assert response.text == "checking"
    assert [call.id for call in response.tool_calls] == ["call-1", "call-2"]
