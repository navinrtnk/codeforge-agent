"""Conversation API tests with persisted history and deterministic model responses."""

import uuid
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import select

from agent.config import Settings
from agent.conversation_memory import history_cost
from agent.main import create_app
from agent.models import Conversation
from agent.models_api import (
    FakeModelClient,
    ModelResponse,
    TextContent,
    TokenUsage,
    ToolCall,
    ToolResult,
)


def answer(text: str) -> ModelResponse:
    return ModelResponse(
        id=str(uuid.uuid4()),
        provider="fake",
        model="test-model",
        content=(TextContent(text),),
        stop_reason="end_turn",
        usage=TokenUsage(10, 5),
    )


def create_client(tmp_path: Path, model: FakeModelClient, budget: int = 8192) -> TestClient:
    settings = Settings(  # type: ignore[call-arg]
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        workspace_root=tmp_path,
        conversation_history_token_budget=budget,
        _env_file=None,
    )
    return TestClient(create_app(settings, model_client=model))


def register(client: TestClient, tmp_path: Path, name: str = "repo") -> str:
    path = tmp_path / name
    path.mkdir()
    (path / "app.py").write_text("value = 42\n", encoding="utf-8")
    response = client.post("/repositories", json={"name": name, "path": str(path)})
    assert response.status_code == 201
    return str(response.json()["id"])


def create(client: TestClient, repository_id: str) -> str:
    response = client.post(
        f"/repositories/{repository_id}/conversations", json={"title": "Discuss code"}
    )
    assert response.status_code == 201
    assert response.json()["summary"] is None
    return f"/repositories/{repository_id}/conversations/{response.json()['id']}"


def test_follow_up_reloads_history_across_application_restarts(tmp_path: Path) -> None:
    first_model = FakeModelClient([answer("The value is 42.")])
    with create_client(tmp_path, first_model) as client:
        repository_id = register(client, tmp_path)
        url = create(client, repository_id)
        response = client.post(url + "/messages", json={"content": "What is the value?"})
        assert response.status_code == 201
        first_run = response.json()["id"]

    second_model = FakeModelClient([answer("It is an integer.")])
    with create_client(tmp_path, second_model) as client:
        response = client.post(url + "/messages", json={"content": "What type is it?"})
        assert response.status_code == 201
        assert response.json()["status"] == "completed"
        # Loaded context is not duplicated into the new run's audit messages.
        assert len(response.json()["messages"]) == 3
        listed = client.get(url + "/messages").json()
        assert [item["turn_number"] for item in listed] == [1, 1, 2, 2]
        assert listed[0]["run_id"] == first_run
        assert client.get(url + "/messages", params={"offset": 2, "limit": 1}).json() == [listed[2]]
        assert client.get(url).json()["is_running"] is False
        assert len(client.get(f"/repositories/{repository_id}/conversations").json()) == 1
    request = second_model.requests[0]
    assert [message.role for message in request.messages] == ["system", "user", "assistant", "user"]
    assert request.messages[1].content == (TextContent("What is the value?"),)
    assert request.messages[2].content == (TextContent("The value is 42."),)


def test_conversation_and_repository_isolation(tmp_path: Path) -> None:
    model = FakeModelClient([answer("Private answer"), answer("Separate answer")])
    with create_client(tmp_path, model) as client:
        repository_id = register(client, tmp_path)
        other_id = register(client, tmp_path, "other")
        url = create(client, repository_id)
        assert client.post(url + "/messages", json={"content": "Private task"}).status_code == 201
        other_url = create(client, other_id)
        assert (
            client.post(other_url + "/messages", json={"content": "Other task"}).status_code == 201
        )
        wrong_url = url.replace(repository_id, other_id)
        assert client.get(wrong_url).status_code == 404
        assert client.get(wrong_url + "/messages").status_code == 404
        assert (
            client.post(wrong_url + "/messages", json={"content": "Leak history"}).status_code
            == 404
        )
        third_url = create(client, repository_id)
        assert client.get(third_url + "/messages").json() == []
    assert all("Private" not in str(message) for message in model.requests[1].messages)


def test_history_preserves_complete_tool_exchange(tmp_path: Path) -> None:
    tool_response = ModelResponse(
        id="tool-response",
        provider="fake",
        model="test-model",
        content=(ToolCall("read-call", "read_file", {"path": "app.py"}),),
        stop_reason="tool_use",
        usage=TokenUsage(10, 5),
    )
    model = FakeModelClient([tool_response, answer("The value is 42"), answer("It is an integer")])
    with create_client(tmp_path, model) as client:
        url = create(client, register(client, tmp_path))
        assert (
            client.post(url + "/messages", json={"content": "Read app.py"}).json()["status"]
            == "completed"
        )
        assert (
            client.post(url + "/messages", json={"content": "What type?"}).json()["status"]
            == "completed"
        )
    blocks = [block for message in model.requests[2].messages for block in message.content]
    calls = [block.id for block in blocks if isinstance(block, ToolCall)]
    results = [block.tool_call_id for block in blocks if isinstance(block, ToolResult)]
    assert calls == results == ["read-call"]


def test_summaries_compact_old_turns_and_persist_without_deleting_messages(tmp_path: Path) -> None:
    model = FakeModelClient(
        [
            answer("first " * 200),
            answer("Keep answers brief"),
            answer("second " * 200),
            answer("Brief answers; explain validation next"),
            answer("Final response"),
        ]
    )
    with create_client(tmp_path, model, budget=512) as client:
        url = create(client, register(client, tmp_path))
        assert (
            client.post(url + "/messages", json={"content": "Keep answers brief"}).status_code
            == 201
        )
        assert (
            client.post(url + "/messages", json={"content": "Explain validation"}).status_code
            == 201
        )
        assert client.post(url + "/messages", json={"content": "Continue"}).status_code == 201
        metadata = client.get(url).json()
        assert metadata["summary_through_turn"] == 2
        assert metadata["summary"] == "Brief answers; explain validation next"
        assert len(client.get(url + "/messages").json()) == 6
        assert client.get(url).json()["is_running"] is False
    assert len(model.requests) == 5
    assert "Keep answers brief" in str(model.requests[3].messages)
    for index in (2, 4):
        assert "Untrusted past summary" in str(model.requests[index].messages[1])
        assert history_cost(model.requests[index].messages[1:-1]) <= 512


def test_summary_failure_keeps_previous_state_and_releases_claim(tmp_path: Path) -> None:
    model = FakeModelClient([answer("long answer " * 100)])
    with create_client(tmp_path, model, budget=256) as client:
        url = create(client, register(client, tmp_path))
        client.post(url + "/messages", json={"content": "First task"})
        assert client.post(url + "/messages", json={"content": "Follow up"}).status_code == 502
        metadata = client.get(url).json()
        assert metadata["summary"] is None
        assert metadata["summary_through_turn"] == 0
        assert metadata["is_running"] is False
        assert len(client.get(url + "/messages").json()) == 2
        model.responses.extend([answer("Previous task"), answer("Recovered")])
        assert (
            client.post(url + "/messages", json={"content": "Retry"}).json()["status"]
            == "completed"
        )


def test_failed_run_is_auditable_but_not_replayed(tmp_path: Path) -> None:
    model = FakeModelClient([])
    with create_client(tmp_path, model) as client:
        url = create(client, register(client, tmp_path))
        failed = client.post(url + "/messages", json={"content": "Failed task"}).json()
        assert failed["status"] == "failed"
        model.responses.append(answer("Success"))
        assert (
            client.post(url + "/messages", json={"content": "New task"}).json()["status"]
            == "completed"
        )
        messages = client.get(url + "/messages").json()
        assert messages[0]["content"] == [{"type": "text", "text": "Failed task"}]
        assert messages[-1]["turn_number"] == 2
    assert "Failed task" not in str(model.requests[-1].messages)


def test_busy_conversation_and_blank_input_rejected(tmp_path: Path) -> None:
    model = FakeModelClient([])
    with create_client(tmp_path, model) as client:
        url = create(client, register(client, tmp_path))
        assert client.post(url + "/messages", json={"content": "  "}).status_code == 422
        with cast(FastAPI, client.app).state.database.session_factory() as session:
            conversation = session.scalar(select(Conversation))
            assert conversation is not None
            conversation.is_running = True
            session.commit()
        response = client.post(url + "/messages", json={"content": "Concurrent task"})
        assert response.status_code == 409
        assert client.get(url).json()["is_running"] is True
    assert model.requests == []
